from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from ..contracts import (
    MAX_CLIP_SECONDS,
    PreparedClip,
    PreparedEvidence,
    VideoInfo,
)
from ..media.subtitles import read_srt
from ..media.video import sample_frames
from ..storage.prepared import identify_source, save_prepared
from .segments import create_segments


def prepare_video(
    video: VideoInfo,
    seconds: Sequence[float],
    output_directory: str | Path,
    *,
    subtitle_path: str | Path | None = None,
    decode_batch_size: int = 4,
    save_workers: int = 4,
    max_pending_frames: int = 16,
) -> PreparedEvidence:
    video_source = identify_source(video.path)
    video = video.model_copy(update={"path": video_source.path})

    subtitle_source = (
        identify_source(subtitle_path) if subtitle_path is not None else None
    )

    cues = read_srt(subtitle_source.path) if subtitle_source is not None else ()

    segments = create_segments(
        cues,
        video.duration_seconds,
    )
    times = tuple(seconds)

    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)

    # The sampler creates frames/ itself.
    # Each yielded record points to a finished image.
    with closing(
        sample_frames(
            video,
            times,
            directory / "frames",
            decode_batch_size=decode_batch_size,
            save_workers=save_workers,
            max_pending_frames=max_pending_frames,
        )
    ) as sampled:
        frames = tuple(sampled)

    frame_ids: list[list[int]] = [[] for _ in segments]

    for frame in frames:
        clip_id = int(frame.requested_seconds // MAX_CLIP_SECONDS)
        frame_ids[clip_id].append(frame.frame_id)

    clips = tuple(
        PreparedClip(
            **segment.model_dump(),
            frame_ids=tuple(ids),
        )
        for segment, ids in zip(
            segments,
            frame_ids,
            strict=True,
        )
    )

    evidence = PreparedEvidence(
        video=video,
        video_source=video_source,
        subtitle_source=subtitle_source,
        requested_seconds=times,
        decode_batch_size=decode_batch_size,
        save_workers=save_workers,
        max_pending_frames=max_pending_frames,
        frames=frames,
        cues=cues,
        clips=clips,
    )

    save_prepared(evidence, directory)
    return evidence
