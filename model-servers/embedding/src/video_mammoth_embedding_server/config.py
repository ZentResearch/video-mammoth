import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    model_id: str = Field(
        default="Qwen/Qwen3-VL-Embedding-2B", min_length=1, pattern=r"\S"
    )
    revision: str | None = Field(default=None, min_length=1, pattern=r"\S")
    device: str = Field(default="cuda", pattern=r"^cuda(?::[0-9]+)?$")
    dtype: Literal["auto", "bf16", "fp16"] = "auto"
    attn_implementation: str = Field(default="sdpa", min_length=1, pattern=r"\S")
    local_files_only: bool = False

    batch_size: int = Field(default=4, gt=0)
    min_pixels: int = Field(default=4096, gt=0)
    max_pixels: int = Field(default=1048576, gt=0)
    max_input_tokens: int = Field(default=8192, gt=0, le=32768)
    max_request_items: int = Field(default=64, gt=0)
    default_instruction: str = Field(
        default="Represent the user's input.",
        min_length=1,
        max_length=8192,
        pattern=r"\S",
    )

    @model_validator(mode="after")
    def validate_pixels(self) -> Self:
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels.")
        return self


def load_settings(path: str | Path | None = None) -> EmbeddingSettings:
    # Editable workspace installs resolve back to the root config/servers.toml.
    default = Path(__file__).resolve().parents[4] / "config" / "servers.toml"
    configured = (
        path
        if path is not None
        else os.environ.get("VIDEO_MAMMOTH_SERVER_CONFIG", str(default))
    )
    with Path(configured).expanduser().open("rb") as source:
        data = tomllib.load(source)
    return EmbeddingSettings.model_validate(data["embeddings"])
