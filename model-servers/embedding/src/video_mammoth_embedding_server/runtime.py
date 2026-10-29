import base64
import binascii
import gc
from collections.abc import Sequence
from contextlib import ExitStack
from io import BytesIO
from typing import Annotated, Literal

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modality_types import MultimodalInput
from sentence_transformers.util import batch_to_device

from .config import EmbeddingSettings

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 25_000_000

IMAGE_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}


class EmbeddingRuntime:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.model_id = settings.model_id

        self._model: SentenceTransformer | None = None
        self._dimensions: int | None = None

    def load(self) -> None:
        if self._model is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("This embedding server requires a CUDA GPU.")

        if self.settings.dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[self.settings.dtype]

        model = SentenceTransformer(
            self.model_id,
            revision=self.settings.revision,
            device=self.settings.device,
            local_files_only=self.settings.local_files_only,
            trust_remote_code=False,
            model_kwargs={
                "dtype": dtype,
                "attn_implementation": self.settings.attn_implementation,
            },
        )

        dimensions = model.get_embedding_dimension()
        if not isinstance(dimensions, int) or dimensions < 1:
            raise RuntimeError("The model did not declare its embedding dimensions.")

        self._dimensions = dimensions
        self._model = model.eval()

    @torch.inference_mode()
    def embed(
        self,
        inputs: Sequence[dict[str, JsonValue]],
        *,
        instruction: str | None,
        input_type: Literal["search_query", "search_document"],
    ) -> list[list[float]]:
        model = self._model

        if model is None:
            raise RuntimeError("Call load() before embed().")

        if not 1 <= len(inputs) <= self.settings.max_request_items:
            raise ValueError(
                f"Provide between 1 and {self.settings.max_request_items} input items."
            )

        if input_type not in {"search_query", "search_document"}:
            raise ValueError("Unsupported input_type.")

        if instruction is not None and (
            not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction) > 8_192
        ):
            raise ValueError("instruction must contain 1-8,192 characters.")

        # Qwen uses the same default prompt for queries and documents.
        prompt = (
            self.settings.default_instruction if instruction is None else instruction
        )

        items = [_Input.model_validate(item) for item in inputs]

        vectors: list[list[float]] = []

        for start in range(0, len(items), self.settings.batch_size):
            batch = items[start : start + self.settings.batch_size]
            vectors.extend(self._embed_batch(model, batch, prompt))

        return vectors

    def close(self) -> None:
        self._model = None
        self._dimensions = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embed_batch(
        self,
        model: SentenceTransformer,
        items: Sequence["_Input"],
        prompt: str,
    ) -> list[list[float]]:
        with ExitStack() as stack:
            batch = [self._model_input(item, stack) for item in items]

            # Prepare model inputs once on the CPU.
            # Reject oversized inputs instead of silently truncating them.
            features = model.preprocess(
                batch,
                prompt=prompt,
                processing_kwargs={
                    "text": {
                        "truncation": False,
                    },
                    "image": {
                        "size": {
                            "shortest_edge": self.settings.min_pixels,
                            "longest_edge": self.settings.max_pixels,
                        },
                    },
                },
            )

            input_ids = features["input_ids"]

            if input_ids.shape[1] > self.settings.max_input_tokens:
                raise ValueError(
                    f"An input exceeds the "
                    f"{self.settings.max_input_tokens}-token limit; "
                    "send less text or a lower-resolution image."
                )

            # Runs Qwen and its saved pooling/normalization modules.
            output = model(batch_to_device(features, model.device))

            vectors = output["sentence_embedding"].float()

            if vectors.shape != (
                len(items),
                self._dimensions,
            ):
                raise RuntimeError("The model returned an unexpected vector shape.")

            if not torch.isfinite(vectors).all().item():
                raise RuntimeError("The model returned non-finite vectors.")

            lengths = torch.linalg.vector_norm(
                vectors,
                dim=1,
                keepdim=True,
            )

            if (lengths == 0).any().item():
                raise RuntimeError("The model returned a zero-length vector.")

            return (vectors / lengths).cpu().tolist()

    def _model_input(
        self,
        item: "_Input",
        stack: ExitStack,
    ) -> MultimodalInput:
        result: MultimodalInput = {}

        texts = [part.text for part in item.content if isinstance(part, _TextPart)]

        if texts:
            result["text"] = "\n".join(texts)

        images = [
            part.image_url.url for part in item.content if isinstance(part, _ImagePart)
        ]

        if len(images) > 1:
            raise ValueError("Provide at most one image per embedding item.")

        if images:
            image = self._decode_image(images[0])
            stack.callback(image.close)
            result["image"] = image

        return result

    @staticmethod
    def _decode_image(url: str) -> Image.Image:
        header, separator, encoded = url.partition(",")

        media_type = header.removeprefix("data:").removesuffix(";base64")

        if (
            not separator
            or header != f"data:{media_type};base64"
            or media_type not in IMAGE_TYPES.values()
        ):
            raise ValueError("Use an inline PNG, JPEG, or WebP data URL.")

        max_encoded_length = 4 * ((MAX_IMAGE_BYTES + 2) // 3)

        if len(encoded) > max_encoded_length:
            raise ValueError("The encoded image exceeds the 20 MiB limit.")

        try:
            data = base64.b64decode(
                encoded,
                validate=True,
            )
        except (binascii.Error, ValueError) as cause:
            raise ValueError("Invalid image base64 data.") from cause

        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Image data is empty or exceeds 20 MiB.")

        try:
            with Image.open(
                BytesIO(data),
                formats=list(IMAGE_TYPES),
            ) as source:
                detected_type = IMAGE_TYPES.get(source.format or "")

                if detected_type != media_type:
                    raise ValueError("Image contents do not match its media type.")

                pixels = source.width * source.height
                if pixels > MAX_SOURCE_PIXELS:
                    raise ValueError("Source images must not exceed 25 million pixels.")

                if getattr(source, "is_animated", False):
                    raise ValueError("Use a still image, not an animation.")

                return source.convert("RGB")

        except (
            OSError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as cause:
            raise ValueError("Cannot decode the supplied image.") from cause


class _WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class _TextPart(_WireModel):
    type: Literal["text"]
    text: str = Field(
        min_length=1,
        max_length=65_536,
        pattern=r"\S",
    )


class _ImageURL(_WireModel):
    url: str = Field(min_length=1)


class _ImagePart(_WireModel):
    type: Literal["image_url"]
    image_url: _ImageURL


class _Input(_WireModel):
    content: list[
        Annotated[
            _TextPart | _ImagePart,
            Field(discriminator="type"),
        ]
    ] = Field(
        min_length=1,
        max_length=8,
    )
