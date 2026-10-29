import math
from typing import Literal, Protocol, Self
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, SchemaError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class Text(Contract):
    text: str = Field(min_length=1)


class Image(Contract):
    url: str = Field(min_length=1, repr=False)

    @field_validator("url")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if value.startswith("data:image/"):
            header, separator, payload = value.partition(",")
            if separator and header.endswith(";base64") and payload:
                return value
            raise ValueError("An inline image needs a base64 data URL.")

        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return value
        raise ValueError("Use an HTTP image URL or a base64 image data URL.")


Part = Text | Image


class Message(Contract):
    role: Literal["system", "user", "assistant"]
    parts: tuple[Part, ...] = Field(min_length=1)


class Usage(Contract):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelIdentity(Contract):
    requested: str = Field(min_length=1, pattern=r"\S")
    returned: str | None = Field(default=None, min_length=1, pattern=r"\S")
    revision: str | None = Field(default=None, min_length=1, pattern=r"\S")


class ParsedJSON(Contract):
    value: JsonValue


class GenerationRequest(Contract):
    messages: tuple[Message, ...] = Field(min_length=1)
    json_schema: dict[str, JsonValue] | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("json_schema")
    @classmethod
    def validate_schema(
        cls, value: dict[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        if value is not None:
            try:
                Draft202012Validator.check_schema(value)
            except SchemaError as error:
                raise ValueError("Invalid JSON Schema.") from error
        return value


class GenerationResult(Contract):
    status: Literal["completed", "truncated", "refused", "unknown"]
    model_identity: ModelIdentity
    text: str = ""
    parsed_json: ParsedJSON | None = None
    finish_reason: str | None = None
    refusal: str | None = None
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.parsed_json is not None and self.status != "completed":
            raise ValueError("Only completed responses may contain parsed_json.")

        if self.refusal is not None and self.status != "refused":
            raise ValueError("A refusal requires status='refused'.")

        if self.status == "completed" and not self.text and self.parsed_json is None:
            raise ValueError("A completed response must contain output.")

        return self


class EmbeddingItem(Contract):
    id: str = Field(min_length=1, pattern=r"\S")
    parts: tuple[Part, ...] = Field(min_length=1)


class EmbeddingRequest(Contract):
    items: tuple[EmbeddingItem, ...] = Field(min_length=1)
    purpose: Literal["document", "query"]
    instruction: str | None = None
    dimensions: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Embedding item IDs must be unique.")
        return self


class EmbeddedItem(Contract):
    id: str = Field(min_length=1, pattern=r"\S")
    vector: tuple[float, ...] = Field(min_length=1, repr=False)


class EmbeddingResult(Contract):
    embeddings: tuple[EmbeddedItem, ...] = Field(min_length=1)
    dimensions: int = Field(gt=0)
    normalization: Literal["l2", "none"]
    model_identity: ModelIdentity
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        ids = [item.id for item in self.embeddings]
        if len(ids) != len(set(ids)):
            raise ValueError("Returned embedding IDs must be unique.")

        for item in self.embeddings:
            if len(item.vector) != self.dimensions:
                raise ValueError(f"Wrong vector dimension for item {item.id!r}.")

            if self.normalization == "l2" and not math.isclose(
                math.hypot(*item.vector), 1.0, rel_tol=1e-4, abs_tol=1e-6
            ):
                raise ValueError(f"Vector for item {item.id!r} is not L2 normalized.")

        return self

    def validate_for(self, request: EmbeddingRequest) -> None:
        """The embedding client calls this before returning its result."""
        requested_ids = tuple(item.id for item in request.items)
        returned_ids = tuple(item.id for item in self.embeddings)

        if returned_ids != requested_ids:
            raise ValueError("Results must match request IDs and order exactly.")

        if request.dimensions is not None and self.dimensions != request.dimensions:
            raise ValueError("Result dimensions differ from the requested value.")


class VLMClient(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


class EmbeddingClient(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...
