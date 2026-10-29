import re
from pathlib import Path

from ..contracts import SubtitleCue

_TIMESTAMP = r"[0-9]{2,}:[0-5][0-9]:[0-5][0-9][,.][0-9]{3}"

_TIMING = re.compile(rf"(?P<start>{_TIMESTAMP})[ \t]*-->[ \t]*(?P<end>{_TIMESTAMP})")


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")

    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_srt(path: str | Path) -> tuple[SubtitleCue, ...]:
    subtitle_path = Path(path).expanduser().resolve(strict=True)

    if not subtitle_path.is_file():
        raise ValueError("The subtitle path must point to a regular file.")

    content = subtitle_path.read_text(encoding="utf-8-sig")
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not content:
        return ()

    cues: list[SubtitleCue] = []

    blocks = re.split(r"\n[ \t]*\n(?:[ \t]*\n)*", content)

    for cue_id, block in enumerate(blocks):
        lines = block.splitlines()

        # SRT counters are optional here; IDs come from file order.
        if lines[0].strip().isdigit():
            lines = lines[1:]

        timing = _TIMING.fullmatch(lines[0].strip()) if lines else None

        if timing is None:
            raise ValueError(f"Invalid SRT timing in subtitle {cue_id + 1}.")

        try:
            cue = SubtitleCue(
                cue_id=cue_id,
                start_seconds=_timestamp_seconds(timing["start"]),
                end_seconds=_timestamp_seconds(timing["end"]),
                text="\n".join(lines[1:]).strip(),
            )
        except (ValueError, OverflowError) as cause:
            raise ValueError(f"Invalid subtitle {cue_id + 1}.") from cause

        cues.append(cue)

    return tuple(
        sorted(
            cues,
            key=lambda cue: (
                cue.start_seconds,
                cue.end_seconds,
                cue.cue_id,
            ),
        )
    )
