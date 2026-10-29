import math
from collections.abc import Mapping, Sequence
from typing import Literal

from ..contracts import MAX_CLIP_SECONDS, PreparedEvidence, Segment, SubtitleCue


def create_segments(
    cues: Sequence[SubtitleCue],
    duration_seconds: float,
) -> tuple[Segment, ...]:
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be positive and finite.")

    ordered = sorted(
        cues,
        key=lambda cue: (
            cue.start_seconds,
            cue.end_seconds,
            cue.cue_id,
        ),
    )

    if len({cue.cue_id for cue in ordered}) != len(ordered):
        raise ValueError("Subtitle IDs must be unique.")

    segments: list[Segment] = []
    active: list[SubtitleCue] = []
    cursor = 0
    start = 0.0

    while start < duration_seconds:
        end = min(
            start + MAX_CLIP_SECONDS,
            float(duration_seconds),
        )

        # Remove cues that ended before this window.
        active = [cue for cue in active if cue.end_seconds > start]

        # Add cues that begin before this window ends.
        while cursor < len(ordered) and ordered[cursor].start_seconds < end:
            cue = ordered[cursor]

            if cue.end_seconds > start:
                active.append(cue)

            cursor += 1

        segments.append(
            Segment(
                clip_id=len(segments),
                start_seconds=start,
                end_seconds=end,
                cue_ids=tuple(cue.cue_id for cue in active),
            )
        )

        start = end

    return tuple(segments)


def get_segment_cues(
    segment: Segment,
    catalog: Mapping[int, SubtitleCue],
) -> tuple[SubtitleCue, ...]:
    selected: list[SubtitleCue] = []

    for cue_id in segment.cue_ids:
        try:
            cue = catalog[cue_id]
        except KeyError as cause:
            raise ValueError(f"Unknown subtitle ID: {cue_id}.") from cause

        if cue.cue_id != cue_id or not cue.overlaps(
            segment.start_seconds,
            segment.end_seconds,
        ):
            raise ValueError(f"Subtitle {cue_id} does not match this clip.")

        selected.append(cue)

    return tuple(selected)


def get_referenced_cues(
    evidence: PreparedEvidence,
) -> tuple[SubtitleCue, ...]:
    catalog = {cue.cue_id: cue for cue in evidence.cues}

    if len(catalog) != len(evidence.cues):
        raise ValueError("Subtitle cue IDs must be unique.")

    selected = {
        cue.cue_id for clip in evidence.clips for cue in get_segment_cues(clip, catalog)
    }

    return tuple(cue for cue in evidence.cues if cue.cue_id in selected)


def get_neighbor(
    catalog: Mapping[int, Segment],
    clip_id: int,
    direction: Literal["previous", "next"],
) -> Segment | None:
    if isinstance(clip_id, bool) or not isinstance(clip_id, int):
        raise TypeError("clip_id must be an integer.")

    if clip_id not in catalog:
        raise ValueError("Unknown clip ID.")

    if direction not in ("previous", "next"):
        raise ValueError("Use direction='previous' or 'next'.")

    step = -1 if direction == "previous" else 1

    return catalog.get(clip_id + step)
