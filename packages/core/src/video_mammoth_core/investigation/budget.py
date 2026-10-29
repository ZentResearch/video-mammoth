from collections.abc import Callable, Sequence

from video_mammoth_models.contracts import GenerationRequest

from ..contracts import RequestBudget, SubtitleCue, VideoFrame


class ContextBudgetError(ValueError):
    """Fixed request content cannot fit after optional context is removed."""


class InvestigationLimitError(RuntimeError):
    """A run has exhausted its total call or elapsed-time allowance."""


class TokenCounterCache:
    """Reuse the last exact request's count without retaining a request history."""

    def __init__(self, count_tokens: Callable[[GenerationRequest], int]) -> None:
        self._count_tokens = count_tokens
        self._request: GenerationRequest | None = None
        self._tokens = 0

    def __call__(self, request: GenerationRequest) -> int:
        if request != self._request:
            tokens = self._count_tokens(request)
            self._request = request
            self._tokens = tokens
        return self._tokens


def input_tokens_fit(budget: RequestBudget, input_tokens: int) -> bool:
    """Check assembled input size, including visual tokens; reject invalid counts."""
    if type(input_tokens) is not int or input_tokens < 0:
        raise ValueError("The token counter must return a nonnegative integer.")

    return input_tokens <= budget.max_input_tokens


def check_request_budget(
    budget: RequestBudget,
    frames: Sequence[VideoFrame],
    cues: Sequence[SubtitleCue],
    *,
    input_tokens: int,
) -> None:
    """Validate supplied raw frames/cues and tokens, without transforming images."""
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
    ):
        raise ValueError("input_tokens must be a nonnegative integer.")

    if len(frames) > budget.max_frames:
        raise ValueError("Too many images for this request.")

    if any(
        frame.width * frame.height > budget.max_pixels_per_frame for frame in frames
    ):
        raise ValueError("An image exceeds the request pixel limit.")

    if sum(len(cue.text) for cue in cues) > budget.max_subtitle_chars:
        raise ValueError("Subtitle text exceeds the request limit.")

    if input_tokens > budget.max_input_tokens:
        raise ValueError("The request exceeds its input-token budget.")
