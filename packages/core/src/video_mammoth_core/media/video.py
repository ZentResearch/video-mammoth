import json
import math
import subprocess
from collections import deque
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import torch
from PIL import Image as PILImage
from torchcodec.decoders import VideoDecoder, set_cuda_backend

from ..contracts import VideoFrame, VideoInfo


def _seconds(value: object) -> float | None:
    if value is None or value == "N/A":
        return None

    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("Expected a numeric timestamp.")

    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError("Timestamps must be finite.")

    return seconds


def _frame_rate(value: object) -> float | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError("Expected a frame-rate string.")

    if value in {"N/A", "0/0", "0/1"}:
        return None

    rate = float(Fraction(value))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Frame rate must be positive and finite.")

    return rate


def _sampling_times(
    seconds: Sequence[float],
    video: VideoInfo,
) -> list[float]:
    times: list[float] = []

    for value in seconds:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value < video.duration_seconds
        ):
            raise ValueError("Every requested time must be inside the video duration.")

        value = float(value)
        if times and value < times[-1]:
            raise ValueError("Requested times must be in increasing order.")
        times.append(value)

    if not times:
        raise ValueError("Provide at least one timestamp.")

    return times


def _save_png(frame: VideoFrame, image: PILImage.Image) -> VideoFrame:
    created = False
    try:
        with frame.image_path.open("xb") as destination:
            created = True
            image.save(destination, format="PNG", compress_level=1)
    except BaseException:
        if created:
            frame.image_path.unlink(missing_ok=True)
        raise
    finally:
        image.close()

    return frame


def sample_frames(
    video: VideoInfo,
    seconds: Sequence[float],
    output_directory: str | Path,
    *,
    decode_batch_size: int = 8,
    save_workers: int = 4,
    max_pending_frames: int = 16,
) -> Generator[VideoFrame]:
    for name, value in {
        "decode_batch_size": decode_batch_size,
        "save_workers": save_workers,
        "max_pending_frames": max_pending_frames,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    times = _sampling_times(seconds, video)

    video_path = video.path.expanduser().resolve(strict=True)
    if not video_path.is_file():
        raise ValueError("The video path must point to a regular file.")

    origin = video.container_start_seconds
    if origin is None:
        origin = video.video_start_seconds
    if origin is None:
        raise ValueError("The video's timeline origin is unknown.")

    source_times = [origin + value for value in times]
    if not all(math.isfinite(value) for value in source_times):
        raise ValueError("Source timestamps must be finite.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")

    with set_cuda_backend("nvdec"):
        decoder = VideoDecoder(
            video_path,
            stream_index=video.stream_index,
            device="cuda",
            dimension_order="NHWC",
            seek_mode="exact",
        )

    try:
        begin = decoder.metadata.begin_stream_seconds
        end = decoder.metadata.end_stream_seconds

        if (
            begin is None
            or end is None
            or not math.isfinite(begin)
            or not math.isfinite(end)
            or end <= begin
        ):
            raise ValueError("The decoder did not report valid video bounds.")

        if any(not begin <= value < end for value in source_times):
            raise ValueError(
                "A requested time has no video frame. "
                f"Playable interval: [{begin - origin:g}, "
                f"{end - origin:g}) seconds."
            )

        directory = Path(output_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=False)

        pending: deque[Future[VideoFrame]] = deque()

        with ThreadPoolExecutor(max_workers=save_workers) as pool:
            try:
                for offset in range(0, len(times), decode_batch_size):
                    batch_times = times[offset : offset + decode_batch_size]

                    batch = decoder.get_frames_played_at(
                        source_times[offset : offset + decode_batch_size]
                    )

                    fallback = decoder.cpu_fallback
                    if (
                        not fallback.status_known
                        or fallback
                        or batch.data.device.type != "cuda"
                    ):
                        raise RuntimeError(
                            f"GPU decoding was not confirmed: {fallback}"
                        )

                    if (
                        batch.data.dtype != torch.uint8
                        or batch.data.ndim != 4
                        or batch.data.shape[0] != len(batch_times)
                        or batch.data.shape[-1] != 3
                    ):
                        raise ValueError("Expected a batch of RGB uint8 frames.")

                    pixels = batch.data.cpu().contiguous().numpy()
                    presentations = batch.pts_seconds.tolist()
                    durations = batch.duration_seconds.tolist()
                    del batch

                    for position, (requested, pts, duration) in enumerate(
                        zip(
                            batch_times,
                            presentations,
                            durations,
                            strict=True,
                        )
                    ):
                        while len(pending) >= max_pending_frames:
                            yield pending.popleft().result()

                        frame_id = offset + position
                        array = pixels[position]

                        frame = VideoFrame(
                            frame_id=frame_id,
                            video_path=video_path,
                            stream_index=video.stream_index,
                            image_path=(directory / f"frame_{frame_id:09d}.png"),
                            requested_seconds=requested,
                            presentation_seconds=float(pts) - origin,
                            duration_seconds=float(duration),
                            width=int(array.shape[1]),
                            height=int(array.shape[0]),
                        )

                        image = PILImage.fromarray(array)
                        try:
                            save_img_future = pool.submit(_save_png, frame, image)
                            pending.append(save_img_future)
                        except BaseException:
                            image.close()
                            raise

                    del pixels

                while pending:
                    yield pending.popleft().result()
            finally:
                for future in pending:
                    future.cancel()
    finally:
        del decoder


def probe_video(
    path: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> VideoInfo:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite.")

    video_path = Path(path).expanduser().resolve(strict=True)
    if not video_path.is_file():
        raise ValueError("The video path must point to a regular file.")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "format=start_time,duration:"
            "stream=index,codec_name,width,height,avg_frame_rate,start_time"
        ),
        "-of",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as cause:
        raise RuntimeError("ffprobe is not installed or is not on PATH.") from cause
    except subprocess.TimeoutExpired as cause:
        raise TimeoutError("Video inspection timed out.") from cause
    except subprocess.CalledProcessError as cause:
        raise ValueError("ffprobe could not inspect this video file.") from cause
    except OSError as cause:
        raise RuntimeError("Could not start ffprobe.") from cause

    try:
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise TypeError("Expected an object from ffprobe.")

        streams = data.get("streams")
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError("Expected one readable video stream.")

        stream = streams[0]
        container = data.get("format")
        if not isinstance(stream, dict) or not isinstance(container, dict):
            raise TypeError("Missing stream or container information.")

        duration = _seconds(container.get("duration"))
        if duration is None:
            raise ValueError("The file duration is unknown.")

        return VideoInfo(
            path=video_path,
            stream_index=stream["index"],
            duration_seconds=duration,
            width=stream["width"],
            height=stream["height"],
            container_start_seconds=_seconds(container.get("start_time")),
            video_start_seconds=_seconds(stream.get("start_time")),
            average_fps=_frame_rate(stream.get("avg_frame_rate")),
            codec=stream.get("codec_name"),
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as cause:
        raise ValueError(
            "The file has no readable video stream or valid video metadata."
        ) from cause
