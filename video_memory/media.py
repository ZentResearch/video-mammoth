"""Timestamped frame sampling and optional on-demand clip exports."""

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .storage import get_clip
from .subtitles import Segment, SubtitleCue, format_timestamp, render_srt


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_seconds: float
    timeline_origin: float
    video_start_seconds: float
    video_end_seconds: float


def probe_video(path: Path) -> VideoInfo:
    from torchcodec.decoders import VideoDecoder

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=start_time,duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        reason = error.stderr.strip() or str(error)
        raise ValueError(f"Could not read video: {reason}") from error
    container = json.loads(result.stdout)["format"]
    metadata = VideoDecoder(path, device="cpu", seek_mode="exact").metadata
    begin, end = metadata.begin_stream_seconds, metadata.end_stream_seconds
    if begin is None or end is None or end <= begin:
        raise ValueError("Could not determine the video presentation timeline.")
    origin = float(container.get("start_time", begin))
    return VideoInfo(path, end - origin, origin, begin, end)


def sample_segment_frames(
    decoder,
    video: VideoInfo,
    segment: Segment,
    clip_directory: Path,
    *,
    fps: float,
    save_workers: int = 4,
) -> list[dict]:
    """Use a global sample grid and retain both requested time and actual frame PTS."""
    directory = clip_directory / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    first = math.ceil(segment.start_seconds * fps - 1e-8)
    last = math.ceil(segment.end_seconds * fps - 1e-8)
    frame_ids = [
        frame_id
        for frame_id in range(first, last)
        if video.video_start_seconds
        <= video.timeline_origin + frame_id / fps
        < video.video_end_seconds
    ]
    records = []
    with ThreadPoolExecutor(max_workers=save_workers) as pool:
        for offset in range(0, len(frame_ids), 8):
            batch_ids = frame_ids[offset : offset + 8]
            timestamps = [
                video.timeline_origin + frame_id / fps for frame_id in batch_ids
            ]
            batch = decoder.get_frames_played_at(timestamps)
            if getattr(decoder, "cpu_fallback", None):
                raise RuntimeError(
                    "CUDA video decoding fell back to CPU. Use --decode-device cpu for this codec."
                )
            pixels = batch.data.cpu().permute(0, 2, 3, 1).contiguous().numpy()
            saves = []
            for frame_id, array, pts in zip(
                batch_ids, pixels, batch.pts_seconds.tolist(), strict=True
            ):
                filename = f"frame_{frame_id:09}.png"
                image = Image.fromarray(array)
                saves.append(pool.submit(image.save, directory / filename))
                records.append(
                    {
                        "frame_id": frame_id,
                        "sample_seconds": frame_id / fps,
                        "presentation_seconds": pts - video.timeline_origin,
                        "timestamp": format_timestamp(pts - video.timeline_origin),
                        "path": f"frames/{filename}",
                    }
                )
            for save in saves:
                save.result()
    return records


def export_clip(directory: Path, clip_id: str | int, output: Path) -> dict:
    clip = get_clip(directory, clip_id)
    output = output.expanduser().resolve()
    if (
        output.suffix.lower() != ".mp4"
        or output.exists()
        or output.with_suffix(".srt").exists()
    ):
        raise ValueError(
            "Choose a new .mp4 output path; existing MP4/SRT files are not overwritten."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-ss",
            str(clip["start_seconds"]),
            "-i",
            clip["video_path"],
            "-t",
            str(clip["end_seconds"] - clip["start_seconds"]),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            str(output),
        ],
        check=True,
    )
    cues = [SubtitleCue(**cue) for cue in clip["cues"]]
    output.with_suffix(".srt").write_text(
        render_srt(cues, offset_seconds=clip["start_seconds"]),
        encoding="utf-8",
    )
    return {
        "video": str(output),
        "srt": str(output.with_suffix(".srt")),
        "start": clip["start"],
        "end": clip["end"],
    }
