import os
from itertools import pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

from ..contracts import MAX_CLIP_SECONDS, PreparedEvidence, SourceFile
from ..indexing.segments import create_segments


def identify_source(path: str | Path) -> SourceFile:
    source = Path(path).expanduser().resolve(strict=True)

    if not source.is_file():
        raise ValueError("The source must be a regular file.")

    stat = source.stat()

    return SourceFile(
        path=source,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
    )


def validate_prepared(
    evidence: PreparedEvidence,
    directory: str | Path,
    *,
    verify_images: bool = False,
) -> None:
    directory = Path(directory).expanduser().resolve(strict=True)

    if evidence.video_source.path != evidence.video.path:
        raise ValueError("Video identity and video metadata disagree.")

    for source in (evidence.video_source, evidence.subtitle_source):
        if source is not None and identify_source(source.path) != source:
            raise ValueError(f"Source changed: {source.path}")

    if evidence.subtitle_source is None and evidence.cues:
        raise ValueError("Subtitles must identify their source file.")

    times = evidence.requested_seconds

    if any(not 0 <= time < evidence.video.duration_seconds for time in times):
        raise ValueError("Sampling times must be inside the video.")

    if any(left > right for left, right in pairwise(times)):
        raise ValueError("Sampling times must be in increasing order.")

    if len(evidence.frames) != len(times):
        raise ValueError("Some requested frames are missing.")

    segments = create_segments(
        evidence.cues,
        evidence.video.duration_seconds,
    )
    frame_ids: list[list[int]] = [[] for _ in segments]

    for index, frame in enumerate(evidence.frames):
        if (
            frame.frame_id != index
            or frame.requested_seconds != times[index]
            or frame.video_path != evidence.video.path
            or frame.stream_index != evidence.video.stream_index
        ):
            raise ValueError(f"Frame {index} does not match its sampling request.")

        expected = directory / "frames" / f"frame_{index:09d}.png"

        if not frame.image_path.is_absolute() or frame.image_path.resolve() != expected:
            raise ValueError(f"Unexpected image path for frame {index}.")

        if not expected.is_file() or expected.stat().st_size == 0:
            raise ValueError(f"Missing or empty image: {expected}")

        with Image.open(expected) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (frame.width, frame.height)
            ):
                raise ValueError(f"Image metadata does not match frame {index}.")

            if verify_images:
                image.verify()

        clip_id = int(frame.requested_seconds // MAX_CLIP_SECONDS)
        frame_ids[clip_id].append(frame.frame_id)

    if len(evidence.clips) != len(segments):
        raise ValueError("The clip plan is incomplete.")

    for clip, segment, ids in zip(
        evidence.clips,
        segments,
        frame_ids,
        strict=True,
    ):
        if clip.model_dump(
            exclude={"frame_ids"}
        ) != segment.model_dump() or clip.frame_ids != tuple(ids):
            raise ValueError(f"Invalid references in clip {segment.clip_id}.")


def save_prepared(
    evidence: PreparedEvidence,
    directory: str | Path,
) -> Path:
    directory = Path(directory).expanduser().resolve(strict=True)
    destination = directory / "manifest.json"

    if destination.exists():
        raise FileExistsError("A completed manifest already exists here.")

    validate_prepared(evidence, directory)

    # Store image paths relative to this folder.
    stored = evidence.model_copy(
        update={
            "frames": tuple(
                frame.model_copy(
                    update={"image_path": frame.image_path.relative_to(directory)}
                )
                for frame in evidence.frames
            )
        }
    )

    temporary: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(stored.model_dump_json(indent=2))
            output.flush()
            os.fsync(output.fileno())

        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return destination


def load_prepared(
    directory: str | Path,
    *,
    verify_images: bool = False,
) -> PreparedEvidence:
    directory = Path(directory).expanduser().resolve(strict=True)

    stored = PreparedEvidence.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )

    if any(frame.image_path.is_absolute() for frame in stored.frames):
        raise ValueError("Stored image paths must be relative.")

    evidence = stored.model_copy(
        update={
            "frames": tuple(
                frame.model_copy(
                    update={"image_path": (directory / frame.image_path).resolve()}
                )
                for frame in stored.frames
            )
        }
    )

    validate_prepared(
        evidence,
        directory,
        verify_images=verify_images,
    )

    return evidence
