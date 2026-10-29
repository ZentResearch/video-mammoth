import base64
import binascii
import gc
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from io import BytesIO
from typing import Annotated, Literal

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)

from .config import VLMSettings
from .structured import JSONDecoder

logger = logging.getLogger(__name__)


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class TextPart(WireModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class ImageURL(WireModel):
    url: str = Field(min_length=1)


class ImagePart(WireModel):
    type: Literal["image_url"]
    image_url: ImageURL


type ContentPart = Annotated[
    TextPart | ImagePart,
    Field(discriminator="type"),
]


class ChatMessage(WireModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    finish_reason: Literal["stop", "length"]
    input_tokens: int
    output_tokens: int


class VLMRuntime:
    def __init__(self, settings: VLMSettings) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self.context_tokens = settings.context_tokens

        self.model = None
        self.processor = None
        self.json_decoder: JSONDecoder | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("The VLM server requires a CUDA GPU.")

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.settings.revision,
            local_files_only=self.settings.local_files_only,
        )
        if (
            self.settings.enable_thinking
            and "</think>" not in processor.tokenizer.get_vocab()
        ):
            raise ValueError(
                "The selected processor does not support thinking markers."
            )
        processor.image_processor.size = {
            "shortest_edge": self.settings.min_image_pixels,
            "longest_edge": self.settings.max_image_pixels,
        }

        if self.settings.dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[self.settings.dtype]

        model = (
            AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                revision=self.settings.revision,
                local_files_only=self.settings.local_files_only,
                dtype=dtype,
                attn_implementation=self.settings.attn_implementation,
            )
            .to(self.settings.device)
            .eval()
        )

        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or ())
        chat_eos = processor.tokenizer.eos_token_id
        if chat_eos is not None and chat_eos not in eos_ids:
            eos_ids.append(chat_eos)
        if not eos_ids:
            raise ValueError("The selected model/tokenizer has no stop token.")
        model.generation_config.eos_token_id = eos_ids

        self.processor = processor
        self.model = model

        self.context_tokens = min(
            self.context_tokens,
            model.config.text_config.max_position_embeddings,
        )

    @torch.inference_mode()
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_schema: dict[str, JsonValue] | None = None,
    ) -> Completion:
        if self.model is None or self.processor is None:
            raise RuntimeError("The VLM is not loaded.")

        max_tokens = (
            self.settings.default_max_tokens if max_tokens is None else max_tokens
        )
        temperature = (
            self.settings.default_temperature if temperature is None else temperature
        )
        if not 1 <= len(messages) <= self.settings.max_messages:
            raise ValueError(
                f"Use between 1 and {self.settings.max_messages} messages."
            )
        if not 1 <= max_tokens <= self.settings.max_output_tokens:
            raise ValueError(
                f"max_tokens must be between 1 and {self.settings.max_output_tokens}."
            )
        if (
            self.settings.enable_thinking
            and max_tokens < self.settings.min_answer_tokens + 2
        ):
            raise ValueError(
                f"With thinking enabled, max_tokens must be at least "
                f"{self.settings.min_answer_tokens + 2}: thinking and its closing "
                "marker must leave room for min_answer_tokens."
            )

        with ExitStack() as images:
            conversation = self._messages(messages, images)

            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                enable_thinking=self.settings.enable_thinking,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        input_tokens = inputs["input_ids"].shape[-1]

        if input_tokens + max_tokens > self.context_tokens:
            raise ValueError(
                f"Request needs {input_tokens} input + {max_tokens} output tokens; "
                f"the limit is {self.context_tokens}. "
                "Use fewer images or less text."
            )

        inputs = inputs.to(self.settings.device)

        options = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            options.update(
                temperature=temperature,
                top_p=self.settings.top_p,
                top_k=self.settings.top_k,
            )

        eos = self.model.generation_config.eos_token_id
        eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        processors = LogitsProcessorList()
        thinking = None
        if self.settings.enable_thinking:
            end_thinking = self.processor.tokenizer.convert_tokens_to_ids("</think>")
            thinking_limit = min(
                self.settings.max_thinking_tokens,
                max_tokens - self.settings.min_answer_tokens - 1,
            )
            thinking = _ThinkingBudget(
                input_tokens,
                thinking_limit,
                end_thinking,
                eos_ids,
                self.processor.tokenizer.get_vocab().get("<think>"),
            )
            processors.append(thinking)

        structured = None
        if json_schema is not None:
            if self.json_decoder is None:
                self.json_decoder = JSONDecoder(self.processor.tokenizer, eos_ids)
            structured = self.json_decoder.constrain(
                json_schema,
                input_tokens,
                thinking.end_token_id if thinking is not None else None,
            )
            processors.append(structured)
        if processors:
            options["logits_processor"] = processors

        output = self.model.generate(**inputs, **options)
        generated = output[0, input_tokens:]

        answer = generated
        if thinking is not None:
            end_thinking = thinking.end_token_id
            positions = (generated == end_thinking).nonzero()
            # Only text after the closing marker belongs to the final answer.
            answer = (
                generated[int(positions[0, 0]) + 1 :]
                if len(positions)
                else generated[:0]
            )

        text = self.processor.decode(
            answer,
            skip_special_tokens=True,
        ).strip()

        stopped = len(generated) > 0 and int(generated[-1]) in eos_ids
        finish_reason = "stop" if stopped else "length"
        if stopped and structured is not None:
            structured.validate(text)
        if not text or not stopped:
            logger.warning(
                "VLM generation: finish_reason=%s, generated_tokens=%d/%d, "
                "answer_chars=%d, thinking_forced=%s",
                finish_reason,
                len(generated),
                max_tokens,
                len(text),
                thinking.forced if thinking is not None else False,
            )

        return Completion(
            text=text,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=len(generated),
        )

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.json_decoder = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _messages(
        self,
        messages: Sequence[ChatMessage],
        images: ExitStack,
    ) -> list[dict]:
        conversation = []
        image_count = 0

        for message in messages:
            if isinstance(message.content, str):
                conversation.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )
                continue

            content = []

            for part in message.content:
                if isinstance(part, TextPart):
                    content.append(
                        {
                            "type": "text",
                            "text": part.text,
                        }
                    )
                else:
                    if message.role != "user":
                        raise ValueError("Images belong in user messages.")

                    image_count += 1
                    if image_count > self.settings.max_images:
                        raise ValueError(
                            f"Use at most {self.settings.max_images} images."
                        )

                    image = _open_image(part.image_url.url)
                    images.callback(image.close)

                    content.append(
                        {
                            "type": "image",
                            "image": image,
                        }
                    )

            conversation.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )

        return conversation


class _ThinkingBudget(LogitsProcessor):
    """End native thinking within its allowance for one unbatched request."""

    def __init__(
        self,
        prompt_length: int,
        token_limit: int,
        end_token_id: int,
        eos_token_ids: Sequence[int],
        begin_token_id: int | None = None,
    ) -> None:
        self.prompt_length = prompt_length
        self.token_limit = token_limit
        self.end_token_id = end_token_id
        self.eos_token_ids = tuple(eos_token_ids)
        self.control_token_ids = (
            (end_token_id,)
            if begin_token_id is None
            else (begin_token_id, end_token_id)
        )
        self.begin_token_id = begin_token_id
        self.ended = False
        self.forced = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.ended:
            scores[:, self.control_token_ids] = -torch.inf
            return scores

        generated_tokens = input_ids.shape[-1] - self.prompt_length
        if generated_tokens and int(input_ids[0, -1]) == self.end_token_id:
            self.ended = True
            scores[:, self.control_token_ids] = -torch.inf
            return scores

        if generated_tokens >= self.token_limit:
            scores.fill_(-torch.inf)
            scores[:, self.end_token_id] = 0
            self.forced = True
        else:
            # A chat must reach the final-answer phase before it can end.
            scores[:, self.eos_token_ids] = -torch.inf
            if self.begin_token_id is not None:
                scores[:, self.begin_token_id] = -torch.inf
        return scores


def _open_image(url: str) -> Image.Image:
    formats = {
        "data:image/png;base64": "PNG",
        "data:image/jpeg;base64": "JPEG",
        "data:image/webp;base64": "WEBP",
    }

    header, separator, encoded = url.partition(",")
    image_format = formats.get(header)
    max_bytes = 20 * 1024 * 1024

    if not separator or image_format is None:
        raise ValueError("Use an inline base64 PNG, JPEG, or WebP image.")

    if len(encoded) > 4 * ((max_bytes + 2) // 3):
        raise ValueError("An image exceeds the 20 MiB limit.")

    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as cause:
        raise ValueError("Invalid base64 image.") from cause

    if not data or len(data) > max_bytes:
        raise ValueError("An image is empty or exceeds the 20 MiB limit.")

    try:
        with Image.open(BytesIO(data), formats=[image_format]) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError("An image exceeds the source-pixel limit.")

            if getattr(source, "is_animated", False):
                raise ValueError("Use a still image.")

            return source.convert("RGB")

    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as cause:
        raise ValueError("Cannot decode the supplied image.") from cause
