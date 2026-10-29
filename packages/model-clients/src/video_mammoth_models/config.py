import tomllib
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from .contracts import Contract


class EndpointProfile(Contract):
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1, pattern=r"^\S+$")
    api_key_env: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    timeout_seconds: float = Field(default=60.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("base_url cannot contain whitespace.")

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP or HTTPS URL.")

        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Do not put credentials in base_url.")

        if parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain a query or fragment.")

        _ = parsed.port

        return parsed.geturl().rstrip("/")


class VLMProfile(EndpointProfile):
    supports_images: bool = False
    supports_schema: bool = False
    supports_temperature: bool = False

    max_output_tokens: int = Field(default=768, gt=0)
    temperature: float | None = Field(default=None, ge=0)
    token_limit_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    @model_validator(mode="after")
    def validate_defaults(self) -> Self:
        if self.temperature is not None and not self.supports_temperature:
            raise ValueError(
                "A default temperature requires supports_temperature=true."
            )
        return self


class EmbeddingProfile(EndpointProfile):
    supports_images: bool = False
    input_format: Literal["text", "content"] = "text"
    purpose_mode: Literal["symmetric", "input_type"] = "symmetric"

    # Only configure this when the endpoint implements this exact field.
    instruction_field: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )

    supports_dimensions: bool = False
    expected_dimensions: int | None = Field(default=None, gt=0)
    normalization: Literal["l2", "none"] = "l2"

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.supports_images and self.input_format != "content":
            raise ValueError("Image embeddings require input_format='content'.")

        if self.instruction_field in {
            "model",
            "input",
            "input_type",
            "dimensions",
            "encoding_format",
        }:
            raise ValueError(
                "instruction_field conflicts with a reserved request field."
            )

        return self


class ModelProfiles(Contract):
    vlm: VLMProfile
    embeddings: EmbeddingProfile


def load_profiles(path: str | Path) -> ModelProfiles:
    config_path = Path(path).expanduser()

    with config_path.open("rb") as source:
        data = tomllib.load(source)

    return ModelProfiles.model_validate(data)
