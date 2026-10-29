import asyncio
import json
import math
from typing import cast

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .config import EmbeddingProfile
from .content import parts_to_api
from .contracts import (
    EmbeddedItem,
    EmbeddingRequest,
    EmbeddingResult,
    ModelIdentity,
    Text,
    Usage,
)
from .errors import ErrorKind, ModelClientError


class _ProviderModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        strict=True,
        allow_inf_nan=False,
    )


class _ProviderEmbedding(_ProviderModel):
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class _ProviderUsage(_ProviderModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class _ProviderResponse(_ProviderModel):
    data: list[_ProviderEmbedding] = Field(min_length=1)
    model: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
    )
    usage: _ProviderUsage | None = None


def _unique_object(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value

    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}.")


def _load_json(value: bytes) -> JsonValue:
    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )


def _http_error(cause: APIStatusError) -> ModelClientError:
    status = cause.status_code

    if status in {401, 403}:
        kind = ErrorKind.AUTHENTICATION
    elif status == 429:
        kind = ErrorKind.RATE_LIMIT
    elif status in {408, 504}:
        kind = ErrorKind.TIMEOUT
    elif status in {400, 404, 413, 422}:
        kind = ErrorKind.INVALID_REQUEST
    else:
        kind = ErrorKind.PROVIDER_ERROR

    body = cause.body
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None

    permanent_limit = code in (
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
    )
    retryable = (status in {408, 409, 429} or status >= 500) and not permanent_limit

    return ModelClientError(
        kind,
        f"The embedding endpoint returned HTTP {status}.",
        retryable=retryable,
        status_code=status,
        request_id=(cause.request_id or "").strip() or None,
        details={
            "retry_after": cause.response.headers.get("retry-after"),
        },
    )


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        profile: EmbeddingProfile,
        sdk: AsyncOpenAI,
    ) -> None:
        self._profile = profile
        self._sdk = sdk.with_options(
            base_url=profile.base_url,
            max_retries=0,
        )

    def _request_options(
        self,
        request: EmbeddingRequest,
    ) -> dict[str, JsonValue]:
        options: dict[str, JsonValue] = {}

        if request.dimensions is not None:
            if not self._profile.supports_dimensions:
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "The selected embedding profile does not support dimensions.",
                )

            options["dimensions"] = request.dimensions

        if self._profile.purpose_mode == "input_type":
            options["input_type"] = (
                "search_query" if request.purpose == "query" else "search_document"
            )

        if request.instruction is not None:
            field = self._profile.instruction_field

            if field is None:
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "The selected embedding profile has no instruction parameter.",
                )

            options[field] = request.instruction

        return options

    def _inputs(
        self,
        request: EmbeddingRequest,
    ) -> list[str] | list[dict[str, JsonValue]]:
        if self._profile.input_format == "content":
            return [
                {
                    "content": parts_to_api(
                        item.parts,
                        supports_images=self._profile.supports_images,
                    ),
                }
                for item in request.items
            ]

        texts: list[str] = []

        for item in request.items:
            if len(item.parts) != 1 or not isinstance(item.parts[0], Text):
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "Text input format requires exactly one Text part per item.",
                )

            texts.append(item.parts[0].text)

        return texts

    def _vector(
        self,
        values: list[float],
    ) -> tuple[float, ...]:
        if self._profile.normalization == "none":
            return tuple(values)

        # Scaling avoids overflow and underflow when calculating the norm.
        scale = max(abs(value) for value in values)

        if scale == 0:
            raise ValueError("Cannot L2-normalize a zero vector.")

        scaled = tuple(value / scale for value in values)
        norm = math.hypot(*scaled)

        return tuple(value / norm for value in scaled)

    def _result(
        self,
        body: bytes,
        request: EmbeddingRequest,
        request_id: str | None,
        http_status: int,
    ) -> EmbeddingResult:
        try:
            response = _ProviderResponse.model_validate(_load_json(body))
            ordered = sorted(
                response.data,
                key=lambda item: item.index,
            )

            if [item.index for item in ordered] != list(range(len(request.items))):
                raise ValueError("Embedding indexes must match the input items.")

            dimensions = len(ordered[0].embedding)
            expected_dimensions = (
                request.dimensions
                if request.dimensions is not None
                else self._profile.expected_dimensions
            )

            if expected_dimensions is not None and dimensions != expected_dimensions:
                raise ValueError("Unexpected embedding dimensions.")

            embeddings: list[EmbeddedItem] = []
            for source, returned in zip(
                request.items,
                ordered,
                strict=True,
            ):
                if len(returned.embedding) != dimensions:
                    raise ValueError("Embedding vectors have inconsistent dimensions.")

                embeddings.append(
                    EmbeddedItem(
                        id=source.id,
                        vector=self._vector(returned.embedding),
                    )
                )

            usage = response.usage

            result = EmbeddingResult(
                embeddings=tuple(embeddings),
                dimensions=dimensions,
                normalization=self._profile.normalization,
                model_identity=ModelIdentity(
                    requested=self._profile.model,
                    returned=response.model,
                ),
                usage=Usage(
                    input_tokens=(usage.prompt_tokens if usage is not None else None),
                    total_tokens=(usage.total_tokens if usage is not None else None),
                ),
                request_id=request_id,
            )

            result.validate_for(request)
            return result

        except ValueError as cause:
            raise ModelClientError(
                ErrorKind.INVALID_RESPONSE,
                "The embedding endpoint returned an invalid response.",
                status_code=http_status,
                request_id=request_id,
            ) from cause

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResult:
        request = request.model_copy(deep=True)
        options = self._request_options(request)

        timeout = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._profile.timeout_seconds
        )

        request_id = None

        try:
            async with asyncio.timeout(timeout):
                inputs = await asyncio.to_thread(
                    self._inputs,
                    request,
                )

                response = await self._sdk.with_options(
                    timeout=timeout
                ).embeddings.with_raw_response.create(
                    model=self._profile.model,
                    # Compatible providers may extend input beyond SDK text types.
                    input=cast(list[str], inputs),
                    encoding_format="float",
                    extra_body=options,
                )

                request_id = (
                    response.headers.get("x-request-id") or ""
                ).strip() or None

                return await asyncio.to_thread(
                    self._result,
                    response.http_response.content,
                    request,
                    request_id,
                    response.status_code,
                )

        except (TimeoutError, APITimeoutError) as cause:
            raise ModelClientError(
                ErrorKind.TIMEOUT,
                "The embedding request timed out.",
                retryable=True,
                request_id=request_id,
            ) from cause

        except APIStatusError as cause:
            raise _http_error(cause) from cause

        except APIConnectionError as cause:
            raise ModelClientError(
                ErrorKind.CONNECTION,
                "Could not connect to the embedding endpoint.",
                retryable=True,
            ) from cause

        except APIError as cause:
            raise ModelClientError(
                ErrorKind.PROVIDER_ERROR,
                "The model SDK could not complete the embedding request.",
            ) from cause
