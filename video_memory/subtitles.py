"""Read timestamped subtitles and choose boundaries without splitting cues."""

import html
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubtitleCue:
    cue_id: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class Segment:
    clip_id: str
    start_seconds: float
    end_seconds: float
    cues: tuple[SubtitleCue, ...]

    @property
    def transcript(self) -> str:
        return "\n".join(cue.text for cue in self.cues)


def format_timestamp(seconds: float, *, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    separator = "," if srt else "."
    clock = f"{minutes:02}:{whole_seconds:02}{separator}{milliseconds:03}"
    return f"{hours:02}:{clock}" if hours or srt else clock


def read_srt(path: Path, *, offset_seconds: float = 0.0) -> list[SubtitleCue]:
    content = (
        path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    )
    timing = re.compile(
        r"(?P<start>\d+:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
        r"(?P<end>\d+:\d{2}:\d{2}[,.]\d{3})"
    )
    cues = []
    # Caption rows may be blank; split only before another cue header.
    for block in re.split(
        r"\n[ \t]*\n(?=(?:\d+[ \t]*\n)?[^\n]*-->)", content.strip()
    ):
        lines = block.strip().splitlines()
        timing_line = next(
            (index for index, line in enumerate(lines[:2]) if "-->" in line), None
        )
        if timing_line is None:
            raise ValueError(
                "Expected an SRT file with HH:MM:SS,mmm --> HH:MM:SS,mmm cues."
            )
        match = timing.search(lines[timing_line])
        if match is None:
            raise ValueError(f"Invalid SRT timestamp: {lines[timing_line]}")
        timestamps = []
        for key in ("start", "end"):
            hours, minutes, seconds = match[key].replace(",", ".").split(":")
            timestamps.append(
                int(hours) * 3600 + int(minutes) * 60 + float(seconds) + offset_seconds
            )
        start, end = timestamps
        text = "\n".join(lines[timing_line + 1 :]).strip()
        if start < 0 or end <= start:
            raise ValueError(f"Invalid subtitle interval: {match[0]}")
        if text:
            cues.append(SubtitleCue(len(cues), start, end, text))
    if not cues:
        raise ValueError("The SRT contains no nonempty subtitle cues.")
    return sorted(
        cues, key=lambda cue: (cue.start_seconds, cue.end_seconds, cue.cue_id)
    )


def create_segments(
    cues: list[SubtitleCue],
    duration_seconds: float,
    *,
    target_seconds: float = 30.0,
    boundary_slack: float = 5.0,
) -> list[Segment]:
    """Prefer nearby sentence endings; overlapping cues remain in one segment."""
    if (
        not all(
            math.isfinite(value)
            for value in (duration_seconds, target_seconds, boundary_slack)
        )
        or duration_seconds <= 0
        or target_seconds <= 0
        or not 0 <= boundary_slack < target_seconds
    ):
        raise ValueError(
            "Use a positive clip length and a boundary slack smaller than that length."
        )
    if max(cue.start_seconds for cue in cues) >= duration_seconds:
        raise ValueError(
            "The SRT has cues starting after this video ends. Check the file or --subtitle-offset."
        )

    groups: list[dict] = []
    for cue in cues:
        if cue.start_seconds >= duration_seconds:
            continue
        if groups and cue.start_seconds < groups[-1]["end"] - 1e-6:
            if cue.end_seconds >= groups[-1]["end"]:
                groups[-1]["text"] = cue.text
            groups[-1]["end"] = max(
                groups[-1]["end"], min(cue.end_seconds, duration_seconds)
            )
        else:
            groups.append(
                {
                    "start": cue.start_seconds,
                    "end": min(cue.end_seconds, duration_seconds),
                    "text": cue.text,
                }
            )

    ends = [group["end"] for group in groups]
    sentence_ends = [
        group["end"]
        for group in groups
        if re.search(
            r"[.!?。！？][\"'’”\)\]]*$", re.sub(r"<[^>]*>", "", group["text"]).strip()
        )
    ]
    segments = []
    start = 0.0
    cue_index = 0
    while start < duration_seconds - 1e-6:
        target = start + target_seconds
        if duration_seconds - start <= target_seconds + boundary_slack:
            end = duration_seconds
        else:
            left = bisect_left(sentence_ends, target - boundary_slack)
            right = bisect_right(sentence_ends, target + boundary_slack)
            nearby = sentence_ends[left:right]
            if not nearby:
                left = bisect_left(ends, target - boundary_slack)
                right = bisect_right(ends, target + boundary_slack)
                nearby = ends[left:right]
            if nearby:
                end = min(nearby, key=lambda value: (abs(value - target), value))
            else:
                position = bisect_left(ends, target)
                group = groups[position] if position < len(groups) else None
                if group is not None and group["start"] < target < group["end"]:
                    choices = [
                        value
                        for value in (group["start"], group["end"])
                        if value > start + 1e-6
                    ]
                    end = min(choices, key=lambda value: (abs(value - target), value))
                else:
                    end = target
        if duration_seconds - end < max(1.0, target_seconds * 0.2):
            end = duration_seconds
        segment_cues = []
        while cue_index < len(cues) and cues[cue_index].start_seconds < end - 1e-6:
            segment_cues.append(cues[cue_index])
            cue_index += 1
        segments.append(Segment(f"{len(segments):06}", start, end, tuple(segment_cues)))
        start = end
    return segments


def render_srt(
    cues: tuple[SubtitleCue, ...] | list[SubtitleCue], *, offset_seconds: float = 0.0
) -> str:
    return "\n\n".join(
        f"{index}\n{format_timestamp(cue.start_seconds - offset_seconds, srt=True)} --> "
        f"{format_timestamp(cue.end_seconds - offset_seconds, srt=True)}\n{cue.text}"
        for index, cue in enumerate(cues, 1)
    ) + ("\n" if cues else "")


def transcript_for_embedding(cues: tuple[SubtitleCue, ...]) -> str:
    return " ".join(
        html.unescape(re.sub(r"<[^>]*>", "", cue.text)).strip() for cue in cues
    )
