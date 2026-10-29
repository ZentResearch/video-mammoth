import base64
import binascii
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from pydantic import JsonValue

from .contracts import Image, Message, Part, Text
from .errors import ErrorKind, ModelClientError

DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_PIXELS = 25_000_000

IMAGE_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}


def _check_limits(max_bytes: int, max_pixels: int) -> None:
    for name, value in (("max_bytes", max_bytes), ("max_pixels", max_pixels)):
        if type(value) is not int or value <= 0:
            raise ModelClientError(
                ErrorKind.INVALID_REQUEST,
                f"{name} must be a positive integer.",
            )


def _image_media_type(data: bytes, *, max_bytes: int, max_pixels: int) -> str:
    if not data or len(data) > max_bytes:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST,
            "Image data is empty or exceeds the configured byte limit.",
        )

    try:
        with PILImage.open(BytesIO(data), formats=list(IMAGE_TYPES)) as decoded:
            media_type = IMAGE_TYPES.get(decoded.format or "")
            if media_type is None:
                raise ModelClientError(
                    ErrorKind.INVALID_REQUEST,
                    "Use PNG, JPEG, WebP.",
                )

            if decoded.width * decoded.height > max_pixels:
                raise ModelClientError(
                    ErrorKind.INVALID_REQUEST,
                    "Image dimensions exceed the configured pixel limit.",
                )

            if getattr(decoded, "is_animated", False):
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "Animated images are not supported; supply a still image.",
                )

            decoded.load()
            return media_type

    except (OSError, ValueError, SyntaxError, PILImage.DecompressionBombError) as cause:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST,
            "Cannot decode a supported image from the supplied bytes.",
        ) from cause


def image_from_file(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Image:
    _check_limits(max_bytes, max_pixels)
    image_path = Path(path).expanduser()

    try:
        with image_path.open("rb") as source:
            data = source.read(max_bytes + 1)
    except OSError as cause:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST,
            "Cannot read the image file.",
            details={"path": str(image_path)},
        ) from cause

    media_type = _image_media_type(data, max_bytes=max_bytes, max_pixels=max_pixels)
    encoded = base64.b64encode(data).decode("ascii")
    return Image(url=f"data:{media_type};base64,{encoded}")


def _check_inline_image(image: Image, *, max_bytes: int, max_pixels: int) -> None:
    header, _, payload = image.url.partition(",")
    declared_type = header.removeprefix("data:").removesuffix(";base64")

    if declared_type not in IMAGE_TYPES.values():
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST,
            "Unsupported inline image media type.",
        )

    # Reject oversized encoded input before allocating decoded bytes.
    max_base64_characters = 4 * ((max_bytes + 2) // 3)
    if len(payload) > max_base64_characters:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST,
            "Inline image exceeds the configured byte limit.",
        )

    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as cause:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST,
            "Inline image contains invalid base64 data.",
        ) from cause

    detected_type = _image_media_type(data, max_bytes=max_bytes, max_pixels=max_pixels)
    if declared_type != detected_type:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST,
            "Inline image media type does not match its contents.",
        )


def parts_to_api(
    parts: Sequence[Part],
    *,
    supports_images: bool,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list[dict[str, JsonValue]]:
    _check_limits(max_bytes, max_pixels)
    if not isinstance(supports_images, bool):
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST, "supports_images must be a boolean."
        )
    if not parts:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST, "At least one content part is required."
        )

    content: list[dict[str, JsonValue]] = []
    for part in parts:
        if isinstance(part, Text):
            content.append({"type": "text", "text": part.text})

        elif isinstance(part, Image):
            if not supports_images:
                raise ModelClientError(
                    ErrorKind.UNSUPPORTED_CAPABILITY,
                    "The selected model profile does not support images.",
                )

            if part.url.startswith("data:"):
                _check_inline_image(part, max_bytes=max_bytes, max_pixels=max_pixels)

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": part.url},
                }
            )

        else:
            raise ModelClientError(
                ErrorKind.INVALID_REQUEST, "Unknown content-part type."
            )

    return content


def messages_to_api(
    messages: Sequence[Message],
    *,
    supports_images: bool,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list[dict[str, JsonValue]]:
    if not messages:
        raise ModelClientError(
            ErrorKind.INVALID_REQUEST, "At least one message is required."
        )

    result: list[dict[str, JsonValue]] = []
    for message in messages:
        if not isinstance(message, Message):
            raise ModelClientError(
                ErrorKind.INVALID_REQUEST, "Expected a Message object."
            )

        if message.role != "user" and any(
            isinstance(part, Image) for part in message.parts
        ):
            raise ModelClientError(
                ErrorKind.UNSUPPORTED_CAPABILITY,
                "This chat format supports images only in user messages.",
            )

        result.append(
            {
                "role": message.role,
                "content": parts_to_api(
                    message.parts,
                    supports_images=supports_images,
                    max_bytes=max_bytes,
                    max_pixels=max_pixels,
                ),
            }
        )

    return result
