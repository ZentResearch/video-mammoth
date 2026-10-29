import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VLMSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    model_id: str = Field(default="Qwen/Qwen3.5-2B", min_length=1, pattern=r"\S")
    revision: str | None = Field(default=None, min_length=1, pattern=r"\S")
    device: str = Field(default="cuda", pattern=r"^cuda(?::[0-9]+)?$")
    dtype: Literal["auto", "bf16", "fp16"] = "auto"
    attn_implementation: str = Field(default="sdpa", min_length=1, pattern=r"\S")
    local_files_only: bool = True

    context_tokens: int = Field(default=8192, gt=0)
    min_image_pixels: int = Field(default=65536, gt=0)
    max_image_pixels: int = Field(default=262144, gt=0)
    max_images: int = Field(default=32, gt=0)
    max_messages: int = Field(default=64, gt=0)

    enable_thinking: bool = False
    max_thinking_tokens: int = Field(default=2048, gt=0)
    min_answer_tokens: int = Field(default=256, gt=0)
    default_max_tokens: int = Field(default=768, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.min_image_pixels > self.max_image_pixels:
            raise ValueError("min_image_pixels must not exceed max_image_pixels.")
        if self.default_max_tokens > self.max_output_tokens:
            raise ValueError("default_max_tokens must not exceed max_output_tokens.")
        if self.default_max_tokens >= self.context_tokens:
            raise ValueError("The default output budget must leave room for input.")
        if (
            self.enable_thinking
            and self.default_max_tokens < self.min_answer_tokens + 2
        ):
            raise ValueError(
                "default_max_tokens must leave room for thinking, its closing "
                "marker, and min_answer_tokens."
            )
        return self


def load_settings(path: str | Path | None = None) -> VLMSettings:
    # Editable workspace installs resolve back to the root config/servers.toml.
    default = Path(__file__).resolve().parents[4] / "config" / "servers.toml"
    configured = (
        path
        if path is not None
        else os.environ.get("VIDEO_MAMMOTH_SERVER_CONFIG", str(default))
    )
    with Path(configured).expanduser().open("rb") as source:
        data = tomllib.load(source)
    return VLMSettings.model_validate(data["vlm"])
