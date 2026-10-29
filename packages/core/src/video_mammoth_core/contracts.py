from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)
from video_mammoth_models.contracts import EmbeddingResult, ModelIdentity

MAX_CLIP_SECONDS = 30.0


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class VideoInfo(Contract):
    path: Path
    stream_index: int = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    container_start_seconds: float | None = None
    video_start_seconds: float | None = None
    average_fps: float | None = Field(default=None, gt=0)
    codec: str | None = Field(default=None, min_length=1)


class VideoFrame(Contract):
    frame_id: int = Field(ge=0)
    video_path: Path
    stream_index: int = Field(ge=0)
    image_path: Path
    requested_seconds: float = Field(ge=0)
    presentation_seconds: float
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SubtitleCue(Contract):
    cue_id: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cue(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("A subtitle must end after it starts.")

        if not self.text.strip():
            raise ValueError("Subtitle text must not be blank.")

        return self

    def overlaps(self, start: float, end: float) -> bool:
        return self.start_seconds < end and start < self.end_seconds


class Segment(Contract):
    clip_id: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    cue_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_segment(self) -> Self:
        if not (
            self.start_seconds
            < self.end_seconds
            <= self.start_seconds + MAX_CLIP_SECONDS
        ):
            raise ValueError(
                f"A clip must have positive length, at most {MAX_CLIP_SECONDS} seconds."
            )

        if any(cue_id < 0 for cue_id in self.cue_ids):
            raise ValueError("Subtitle IDs must be nonnegative.")

        if len(self.cue_ids) != len(set(self.cue_ids)):
            raise ValueError("A clip must not reference a subtitle twice.")

        return self

    def contains(self, seconds: float) -> bool:
        return self.start_seconds <= seconds < self.end_seconds


class RequestBudget(Contract):
    max_frames: int = Field(gt=0)
    # Raw-frame checks use this limit; model servers own processing resolution.
    max_pixels_per_frame: int = Field(gt=0)
    max_subtitle_chars: int = Field(gt=0)

    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    safety_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        reserved_tokens = self.max_output_tokens + self.safety_tokens

        if reserved_tokens >= self.context_window_tokens:
            raise ValueError("The budget must leave room for model input.")

        return self

    @property
    def max_input_tokens(self) -> int:
        return self.context_window_tokens - self.max_output_tokens - self.safety_tokens


class SourceFile(Contract):
    path: Path
    size_bytes: int = Field(ge=0)
    modified_ns: int = Field(ge=0)


class PreparedClip(Segment):
    frame_ids: tuple[int, ...] = ()


class PreparedEvidence(Contract):
    schema_version: Literal[1] = 1
    preparation_id: UUID = Field(default_factory=uuid4)

    video: VideoInfo
    video_source: SourceFile
    subtitle_source: SourceFile | None = None

    requested_seconds: tuple[float, ...] = Field(min_length=1)
    image_format: Literal["PNG"] = "PNG"
    png_compress_level: Literal[1] = 1
    decode_batch_size: int = Field(gt=0)
    save_workers: int = Field(gt=0)
    max_pending_frames: int = Field(gt=0)

    frames: tuple[VideoFrame, ...] = Field(min_length=1)
    cues: tuple[SubtitleCue, ...] = ()
    clips: tuple[PreparedClip, ...] = Field(min_length=1)


class VisualPrompt(Contract):
    name: str = Field(
        min_length=1,
        pattern=r"\S",
    )
    instruction: str = Field(
        min_length=1,
        pattern=r"\S",
    )
    query_instruction: str = Field(
        min_length=1,
        pattern=r"\S",
    )


class EmbeddingPrompts(Contract):
    visual: tuple[VisualPrompt, ...] = Field(
        min_length=1,
    )
    subtitle_instruction: str = Field(
        min_length=1,
        pattern=r"\S",
    )
    subtitle_query_instruction: str = Field(
        min_length=1,
        pattern=r"\S",
    )

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        names = [prompt.name for prompt in self.visual]

        if len(names) != len(set(names)) or "subtitles" in names:
            raise ValueError(
                "Visual channels need unique names; 'subtitles' is reserved."
            )

        return self


class EmbeddingBatch(Contract):
    preparation_id: UUID
    channel: str = Field(min_length=1, pattern=r"\S")
    instruction: str = Field(min_length=1, pattern=r"\S")
    source_ids: tuple[int, ...] = Field(min_length=1)
    result: EmbeddingResult

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        if any(item < 0 for item in self.source_ids):
            raise ValueError("Source IDs must be nonnegative.")

        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("A batch must not repeat source IDs.")

        kind = "cue" if self.channel == "subtitles" else "frame"

        expected = tuple(
            f"{self.preparation_id}:{kind}:{item}" for item in self.source_ids
        )
        returned = tuple(item.id for item in self.result.embeddings)

        if returned != expected:
            raise ValueError("Batch vectors must match source IDs in order.")

        return self


class SearchRequest(Contract):
    query: str = Field(min_length=1, pattern=r"\S")
    style: str = Field(default="general", min_length=1, pattern=r"\S")
    channels: tuple[str, ...] | None = Field(default=None, min_length=1)
    top_k: int = Field(default=5, gt=0)
    instruction: str | None = Field(default=None, min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def validate_channels(self) -> Self:
        if self.channels is not None:
            if any(not name.strip() for name in self.channels):
                raise ValueError("Channel names must not be blank.")

            if len(set(self.channels)) != len(self.channels):
                raise ValueError("Select each channel only once.")

        return self


class SourceMatch(Contract):
    channel: str = Field(min_length=1, pattern=r"\S")
    source_kind: Literal["frame", "cue"]
    source_id: int = Field(ge=0)
    start_seconds: float
    end_seconds: float
    score: float

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("A matched source must have a positive duration.")

        return self


class ClipCandidate(PreparedClip):
    score: float
    matches: tuple[SourceMatch, ...] = Field(min_length=1)


class SearchResult(Contract):
    preparation_id: UUID
    collection_id: UUID
    request: SearchRequest
    instruction: str = Field(min_length=1, pattern=r"\S")
    channels: tuple[str, ...] = Field(min_length=1)
    model_identity: ModelIdentity
    candidates: tuple[ClipCandidate, ...] = ()


class OutlineSection(Contract):
    section_id: int = Field(ge=0)
    cue_ids: tuple[int, ...] = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def validate_section(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("An outline section must have a positive duration.")

        if any(cue_id < 0 for cue_id in self.cue_ids):
            raise ValueError("Cue IDs must be nonnegative.")

        if len(set(self.cue_ids)) != len(self.cue_ids):
            raise ValueError("Each section must reference a cue only once.")

        return self


class VideoOutline(Contract):
    schema_version: Literal[1] = 1
    preparation_id: UUID
    cache_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis: Literal["subtitles", "unavailable"]
    overview: str = Field(min_length=1, pattern=r"\S")
    sections: tuple[OutlineSection, ...] = ()
    model_identity: ModelIdentity | None = None

    @model_validator(mode="after")
    def validate_basis(self) -> Self:
        if (self.basis == "subtitles") != bool(self.sections):
            raise ValueError("Subtitle outlines need at least one section.")

        if self.sections and self.model_identity is None:
            raise ValueError("Generated outlines must record model identity.")

        return self


class SearchPlan(Contract):
    queries: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_queries(self) -> Self:
        normalized = [" ".join(query.casefold().split()) for query in self.queries]

        if any(not query for query in normalized):
            raise ValueError("Search queries must not be blank.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("Search queries must not repeat.")

        return self


class InitialSearchResult(Contract):
    plan: SearchPlan
    searches: tuple[SearchResult, ...] = Field(min_length=1)
    clip_ids: tuple[int, ...] = ()
    rank_constant: int = Field(default=60, ge=0)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        queries = tuple(search.request.query for search in self.searches)
        if queries != self.plan.queries:
            raise ValueError("Search results must match the plan in order.")

        first = self.searches[0]
        for search in self.searches:
            if (
                search.preparation_id != first.preparation_id
                or search.collection_id != first.collection_id
            ):
                raise ValueError("Initial searches must use the same collection.")

        expected = {
            candidate.clip_id
            for search in self.searches
            for candidate in search.candidates
        }
        if (
            len(self.clip_ids) != len(set(self.clip_ids))
            or set(self.clip_ids) != expected
        ):
            raise ValueError(
                "The clip queue must contain every candidate exactly once."
            )

        return self


def _nonblank_action_text(value: str) -> str:
    if not value.strip():
        raise ValueError("Action text must not be blank.")
    return value


type ActionText = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_nonblank_action_text),
]


class SourceReference(Contract):
    source_kind: Literal["frame", "cue"]
    source_id: int = Field(ge=0)


class PagedAction(Contract):
    cursor: ActionText | None = Field(
        default=None,
        description=(
            "Null starts reading. To continue, use a cursor returned by the "
            "same tool with the same target arguments."
        ),
    )


class SearchAction(SearchRequest):
    tool: Literal["search"]
    query: ActionText = Field(description="The question or keywords to search.")
    style: ActionText = Field(
        default="general",
        description="General, subtitles, or a configured visual channel's name.",
    )
    instruction: ActionText | None = Field(
        default=None,
        description="Optional embedding instruction overriding the selected style.",
    )


class OpenClipAction(PagedAction):
    tool: Literal["open_clip"]
    clip_id: int = Field(ge=0)


class OpenNeighborAction(PagedAction):
    tool: Literal["open_neighbor"]
    clip_id: int = Field(ge=0, description="The anchor clip, not the neighbor.")
    direction: Literal["previous", "next"]


class ReadSourcesAction(PagedAction):
    tool: Literal["read_sources"]
    sources: tuple[SourceReference, ...] = Field(min_length=1)


class ReadOutlineAction(PagedAction):
    tool: Literal["read_outline"]
    section_id: int | None = Field(default=None, ge=0)


class ReadNotebookAction(PagedAction):
    """Read the investigation's saved working notes.
    Null entry_id lists entries; an ID selects a particular entry.
    """

    tool: Literal["read_notebook"]
    entry_id: int | None = Field(default=None, ge=0)


class WriteNotebookAction(Contract):
    tool: Literal["write_notebook"]
    entry_id: int | None = Field(
        default=None,
        ge=0,
        description="Null creates an entry; an existing ID updates that entry.",
    )
    kind: Literal["plan", "observation", "hypothesis", "question", "contradiction"]
    text: ActionText
    sources: tuple[SourceReference, ...] = ()
    status: Literal["active", "resolved"] = "active"

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if self.kind in {"observation", "contradiction"} and not self.sources:
            raise ValueError("Observations and contradictions require sources.")
        return self


class RequestFinishAction(Contract):
    tool: Literal["request_finish"]
    answer: ActionText
    sources: tuple[SourceReference, ...] = ()
    gaps: tuple[ActionText, ...] = ()

    @model_validator(mode="after")
    def validate_support_or_gaps(self) -> Self:
        if not self.sources and not self.gaps:
            raise ValueError("Supply answer sources or explain the unresolved gaps.")
        return self


class InvestigationLimits(Contract):
    max_model_calls: int = Field(default=128, gt=0)
    max_seconds: float = Field(default=3600.0, gt=0)
    max_no_progress_steps: int = Field(default=8, gt=0)
    max_attempts: int = Field(default=2, gt=0)


type ReviewText = Annotated[ActionText, Field(max_length=400)]


class AnswerReview(Contract):
    """A bounded, cumulative assessment; page coverage is checked by Python."""

    verdict: Literal["supported", "needs_more_evidence"]
    summary: ActionText = Field(max_length=1600)
    sources: tuple[SourceReference, ...] = ()
    gaps: tuple[ReviewText, ...] = Field(default=(), max_length=8)
    contradictions: tuple[ReviewText, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_verdict(self) -> Self:
        if self.verdict == "supported" and (
            not self.sources or self.gaps or self.contradictions
        ):
            raise ValueError("Supported answers need sources and no unresolved issues.")
        if self.verdict == "needs_more_evidence" and not (
            self.gaps or self.contradictions
        ):
            raise ValueError("Explain the missing or contradictory evidence.")
        return self


type ToolAction = (
    SearchAction
    | OpenClipAction
    | OpenNeighborAction
    | ReadSourcesAction
    | ReadOutlineAction
    | ReadNotebookAction
    | WriteNotebookAction
    | RequestFinishAction
)


class ActionRequest(Contract):
    action: ToolAction


class EvidenceView(Contract):
    kind: Literal["frame", "cue", "overview", "outline"]
    source_id: int = Field(ge=0)
    start: int = Field(default=0, ge=0)
    end: int = Field(default=0, ge=0)
    # Dimensions of the saved source image; the reader sends its original bytes.
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        if self.kind == "frame":
            if self.start or self.end or not self.width or not self.height:
                raise ValueError(
                    "Frame views need source dimensions and no text offsets."
                )
        elif self.end <= self.start or self.width or self.height:
            raise ValueError(
                "Text views need a positive character range and no dimensions."
            )
        return self


class EvidencePage(Contract):
    preparation_id: UUID
    tool: Literal["open_clip", "open_neighbor", "read_sources", "read_outline"]
    views: tuple[EvidenceView, ...] = ()
    visual_window: tuple[float, float] | None = None
    notice: str = ""
    next_cursor: str | None = None


class NotebookEntry(Contract):
    entry_id: int = Field(ge=0)
    kind: Literal[
        "plan",
        "observation",
        "hypothesis",
        "question",
        "contradiction",
    ]
    text: ActionText
    sources: tuple[SourceReference, ...] = ()
    status: Literal["active", "resolved"] = "active"
    updated_step: int = Field(gt=0)


class NotebookState(Contract):
    schema_version: Literal[1] = 1
    run_id: UUID = Field(default_factory=uuid4)

    question: str = Field(min_length=1, pattern=r"\S")
    preparation_id: UUID
    collection_id: UUID

    status: Literal["running", "completed", "incomplete"] = "running"
    step_count: int = Field(default=0, ge=0)
    entries: tuple[NotebookEntry, ...] = ()
    presented: tuple[EvidenceView, ...] = ()

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        if tuple(entry.entry_id for entry in self.entries) != tuple(
            range(len(self.entries))
        ):
            raise ValueError("Notebook entry IDs must be consecutive.")

        if any(entry.updated_step > self.step_count for entry in self.entries):
            raise ValueError("A note cannot come from a future step.")

        return self


class NotebookEvent(Contract):
    run_id: UUID
    step_count: int = Field(gt=0)
    action: ToolAction
    result: dict[str, JsonValue]
    presented: tuple[EvidenceView, ...] = ()
