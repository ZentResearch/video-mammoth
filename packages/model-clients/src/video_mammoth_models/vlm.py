import asyncio
import json
from typing import Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaValidationError
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from referencing import Registry
from referencing.exceptions import Unresolvable

from .config import VLMProfile
from .content import messages_to_api
from .contracts import (
    GenerationRequest,
    GenerationResult,
    ModelIdentity,
    ParsedJSON,
    Usage,
)
from .errors import ErrorKind, ModelClientError


class _ProviderModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)


class _ProviderMessage(_ProviderModel):
    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[JsonValue] | None = None
    function_call: dict[str, JsonValue] | None = None


class _ProviderChoice(_ProviderModel):
    index: int = Field(ge=0)
    message: _ProviderMessage
    finish_reason: str | None = None


class _ProviderUsage(_ProviderModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class _ProviderResponse(_ProviderModel):
    choices: list[_ProviderChoice] = Field(min_length=1, max_length=1)
    model: str | None = Field(default=None, min_length=1, pattern=r"\S")
    usage: _ProviderUsage | None = None


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}.")


def _load_json(value: str | bytes) -> JsonValue:
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
        f"The model endpoint returned HTTP {status}.",
        retryable=retryable,
        status_code=status,
        request_id=(cause.request_id or "").strip() or None,
        details={"retry_after": cause.response.headers.get("retry-after")},
    )


class OpenAICompatibleVLM:
    def __init__(self, profile: VLMProfile, sdk: AsyncOpenAI) -> None:
        self._profile = profile
        self._sdk = sdk.with_options(base_url=profile.base_url, max_retries=0)

    def _request_options(self, request: GenerationRequest) -> dict[str, JsonValue]:
        options: dict[str, JsonValue] = {
            self._profile.token_limit_parameter: (
                request.max_output_tokens
                if request.max_output_tokens is not None
                else self._profile.max_output_tokens
            ),
        }

        temperature = (
            request.temperature
            if request.temperature is not None
            else self._profile.temperature
        )
        if temperature is not None:
            if not self._profile.supports_temperature:
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "The selected model profile does not support temperature.",
                )
            options["temperature"] = temperature

        if request.json_schema is not None:
            if not self._profile.supports_schema:
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "The selected model profile does not support JSON Schema.",
                )

            schema = request.json_schema
            if schema.get("type") != "object" or "anyOf" in schema:
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "This structured-output format requires an object root without anyOf.",
                )

            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "model_response",
                    "strict": True,
                    "schema": schema,
                },
            }

        return options

    def _result(
        self,
        body: bytes,
        request: GenerationRequest,
        request_id: str | None,
        http_status: int,
    ) -> GenerationResult:
        try:
            response = _ProviderResponse.model_validate(_load_json(body))
            choice = response.choices[0]
            message = choice.message

            if (
                choice.index != 0
                or message.tool_calls
                or message.function_call is not None
                or choice.finish_reason in {"tool_calls", "function_call"}
            ):
                raise ValueError("Unexpected choice index or native tool call.")

            status: Literal["completed", "truncated", "refused", "unknown"]
            if message.refusal is not None or choice.finish_reason == "content_filter":
                status = "refused"
            elif choice.finish_reason == "stop":
                status = "completed"
            elif choice.finish_reason == "length":
                status = "truncated"
            else:
                status = "unknown"

            text = message.content if message.content is not None else ""
            parsed_json = None
            if status == "completed" and request.json_schema is not None:
                value = _load_json(text)
                Draft202012Validator(request.json_schema, registry=Registry()).validate(
                    value
                )
                parsed_json = ParsedJSON(value=value)
            usage = response.usage

            return GenerationResult(
                status=status,
                model_identity=ModelIdentity(
                    requested=self._profile.model,
                    returned=response.model,
                ),
                text=text,
                parsed_json=parsed_json,
                finish_reason=choice.finish_reason,
                refusal=message.refusal,
                usage=Usage(
                    input_tokens=usage.prompt_tokens if usage is not None else None,
                    output_tokens=usage.completion_tokens
                    if usage is not None
                    else None,
                    total_tokens=usage.total_tokens if usage is not None else None,
                ),
                request_id=request_id,
            )

        except (ValueError, SchemaValidationError, Unresolvable) as cause:
            raise ModelClientError(
                ErrorKind.INVALID_RESPONSE,
                "The model endpoint returned an invalid response.",
                status_code=http_status,
                request_id=request_id,
            ) from cause

    async def generate(self, request: GenerationRequest) -> GenerationResult:
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
                messages = await asyncio.to_thread(
                    messages_to_api,
                    request.messages,
                    supports_images=self._profile.supports_images,
                )
                response = await self._sdk.with_options(
                    timeout=timeout
                ).chat.completions.with_raw_response.create(
                    model=self._profile.model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    stream=False,
                    n=1,
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
                "The model request timed out.",
                retryable=True,
                request_id=request_id,
            ) from cause
        except APIStatusError as cause:
            raise _http_error(cause) from cause
        except APIConnectionError as cause:
            raise ModelClientError(
                ErrorKind.CONNECTION,
                "Could not connect to the model endpoint.",
                retryable=True,
            ) from cause
        except APIError as cause:
            raise ModelClientError(
                ErrorKind.PROVIDER_ERROR,
                "The model SDK could not complete the request.",
            ) from cause
