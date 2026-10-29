import base64
from contextlib import ExitStack
from io import BytesIO

from PIL import Image as PILImage
from transformers import AutoProcessor
from video_mammoth_models.contracts import GenerationRequest, Text

from .config import VLMSettings


class LocalTokenCounter:
    """Count a local VLM request, including images, without loading model weights."""

    def __init__(self, settings: VLMSettings) -> None:
        self.processor = AutoProcessor.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            local_files_only=settings.local_files_only,
        )
        self.processor.image_processor.size = {
            "shortest_edge": settings.min_image_pixels,
            "longest_edge": settings.max_image_pixels,
        }
        self.enable_thinking = settings.enable_thinking

    def __call__(self, request: GenerationRequest) -> int:
        conversation = []

        with ExitStack() as resources:
            for message in request.messages:
                content = []
                for part in message.parts:
                    if isinstance(part, Text):
                        content.append({"type": "text", "text": part.text})
                    else:
                        if not part.url.startswith("data:image/"):
                            raise ValueError(
                                "Local counting requires inline image data."
                            )
                        data = base64.b64decode(
                            part.url.split(",", 1)[1], validate=True
                        )
                        buffer = resources.enter_context(BytesIO(data))
                        image = resources.enter_context(PILImage.open(buffer))
                        content.append({"type": "image", "image": image})
                conversation.append({"role": message.role, "content": content})

            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

        return int(inputs["input_ids"].shape[-1])
