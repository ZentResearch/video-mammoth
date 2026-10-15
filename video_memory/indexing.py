"""Build or resume an ordered video memory from a video and an SRT file."""

import hashlib
import json
import math
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from .encoder import MODEL_ALIASES, EmbeddingModel
from .media import VideoInfo, probe_video, sample_segment_frames
from .storage import SCHEMA_VERSION, load_index, save_vectors, write_json
from .subtitles import (
    Segment,
    create_segments,
    format_timestamp,
    read_srt,
    render_srt,
    transcript_for_embedding,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "prompts.json"


@dataclass
class PreparedIndex:
    video: VideoInfo
    segments: list[Segment]
    directory: Path
    manifest: dict


def load_prompts(path: Path = DEFAULT_PROMPTS) -> dict:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    visual = prompts["visual"]
    names = [prompt["name"] for prompt in visual]
    if not visual or len(set(names)) != len(names) or "transcript" in names:
        raise ValueError("Visual prompts need unique names; 'transcript' is reserved.")
    for prompt in visual:
        if not all(
            isinstance(prompt[key], str) and prompt[key].strip()
            for key in ("name", "instruction", "query_instruction")
        ):
            raise ValueError(
                "Each visual prompt needs a name, instruction, and query_instruction."
            )
    for key in ("transcript_instruction", "transcript_query_instruction"):
        if not isinstance(prompts[key], str) or not prompts[key].strip():
            raise ValueError(f"The prompt configuration needs {key}.")
    return prompts


def prepare_index(
    video_path: Path,
    srt_path: Path,
    recipe: dict,
    *,
    memory_directory: Path,
    output: Path | None,
) -> PreparedIndex:
    video_path, srt_path = (
        video_path.expanduser().resolve(),
        srt_path.expanduser().resolve(),
    )
    video = probe_video(video_path)
    cues = read_srt(srt_path, offset_seconds=recipe["subtitle_offset"])
    segments = create_segments(
        cues,
        video.duration_seconds,
        target_seconds=recipe["clip_seconds"],
        boundary_slack=recipe["boundary_slack"],
    )
    with video_path.open("rb") as source:
        video_hash = hashlib.file_digest(source, "sha256").hexdigest()
    recipe = {
        **recipe,
        "video_sha256": video_hash,
        "srt_sha256": hashlib.sha256(srt_path.read_bytes()).hexdigest(),
    }
    recipe_id = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", video_path.stem).strip("-") or "video"
    directory = (
        (
            output
            if output is not None
            else memory_directory / f"{slug}-{recipe_id[:12]}"
        )
        .expanduser()
        .resolve()
    )
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = load_index(directory, require_complete=False)
        if manifest["recipe_id"] != recipe_id:
            raise ValueError(
                "This output contains a different indexing recipe. Use another --output directory."
            )
        preserve_source_files(directory, video_path, srt_path, manifest)
        return PreparedIndex(video, segments, directory, manifest)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            "The output directory is nonempty and is not a video-memory index."
        )
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_hash[:16],
        "recipe_id": recipe_id,
        "source_video": str(video_path),
        "source_srt": str(srt_path),
        "duration_seconds": video.duration_seconds,
        "duration": format_timestamp(video.duration_seconds),
        "timeline_origin": video.timeline_origin,
        "status": "building",
        "completed_clips": 0,
        "recipe": recipe,
        "model": {"model_id": recipe["model_id"], "revision": recipe["model_revision"]},
        "clips": [
            {
                "clip_id": segment.clip_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "start": format_timestamp(segment.start_seconds),
                "end": format_timestamp(segment.end_seconds),
                "directory": f"clips/{segment.clip_id}",
            }
            for segment in segments
        ],
    }
    if recipe["copy_video"]:
        manifest["video_copy"] = f"source{video_path.suffix.lower()}"
    write_json(manifest_path, manifest)
    preserve_source_files(directory, video_path, srt_path, manifest)
    return PreparedIndex(video, segments, directory, manifest)


def preserve_source_files(
    directory: Path, video_path: Path, srt_path: Path, manifest: dict
) -> None:
    """Finish source copies atomically, including after an interrupted --copy-video run."""
    sources = [(srt_path, directory / "source.srt")]
    if manifest.get("video_copy"):
        sources.append((video_path, directory / manifest["video_copy"]))
    for source, destination in sources:
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
    manifest["source_video"] = str(video_path)
    manifest["source_srt"] = str(srt_path)
    write_json(directory / "manifest.json", manifest)


def prepare_clip_evidence(index: PreparedIndex, segment: Segment, decoder) -> dict:
    directory = index.directory / "clips" / segment.clip_id
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = sample_segment_frames(
        decoder,
        index.video,
        segment,
        directory,
        fps=index.manifest["recipe"]["fps"],
    )
    metadata = {
        **asdict(segment),
        "start": format_timestamp(segment.start_seconds),
        "end": format_timestamp(segment.end_seconds),
        "transcript": segment.transcript,
        "frames": frames,
    }
    (directory / "transcript.txt").write_text(
        segment.transcript + "\n", encoding="utf-8"
    )
    (directory / "subtitles.srt").write_text(render_srt(segment.cues), encoding="utf-8")
    write_json(metadata_path, metadata)
    return metadata


def embed_segments(
    index: PreparedIndex, model: EmbeddingModel, *, decode_device: str
) -> None:
    from torchcodec.decoders import VideoDecoder

    manifest = index.manifest
    prompts = manifest["recipe"]["prompts"]
    decoder = VideoDecoder(index.video.path, device=decode_device, seek_mode="exact")
    for completed, segment in enumerate(index.segments, 1):
        clip_directory = index.directory / "clips" / segment.clip_id
        vector_path = clip_directory / "embeddings.pt"
        if not vector_path.exists():
            metadata = prepare_clip_evidence(index, segment, decoder)
            paths = [
                str(clip_directory / frame["path"]) for frame in metadata["frames"]
            ]
            transcript = transcript_for_embedding(segment.cues)
            if (
                len(model.model.tokenizer.encode(transcript))
                > model.model.max_seq_length - 256
            ):
                raise ValueError(
                    f"Clip {segment.clip_id} has too much subtitle text for the model. Split unusually long/overlapping cues upstream."
                )
            frame_vectors = model.embed_frames(paths, prompts["visual"])
            transcript_vector = (
                model.encode([transcript], prompts["transcript_instruction"])[0].cpu()
                if transcript.strip()
                else None
            )
            save_vectors(vector_path, frame_vectors, transcript_vector)
            print(
                f"Clip {segment.clip_id}: {metadata['start']}–{metadata['end']}; {len(paths)} frames; transcript={transcript_vector is not None}",
                file=sys.stderr,
                flush=True,
            )
        manifest["completed_clips"] = completed
        write_json(index.directory / "manifest.json", manifest)


def index_video(
    video_path: Path,
    srt_path: Path,
    *,
    memory_directory: Path = Path("memory"),
    output: Path | None = None,
    model_name: str = "qwen",
    prompts: dict | None = None,
    clip_seconds: float = 30.0,
    boundary_slack: float = 5.0,
    fps: float = 1.0,
    subtitle_offset: float = 0.0,
    batch_size: int = 1,
    max_pixels: int = 1_048_576,
    device: str = "cuda",
    decode_device: str = "cuda",
    copy_video: bool = False,
    trust_remote_code: bool = False,
    model_revision: str | None = None,
) -> Path:
    started = time.perf_counter()
    if (
        not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(subtitle_offset)
        or batch_size < 1
        or max_pixels < 65_536
    ):
        raise ValueError(
            "Use positive FPS and batch size, a finite subtitle offset, and at least 65536 --max-pixels."
        )
    recipe = {
        "model_id": MODEL_ALIASES.get(model_name, model_name),
        "model_revision": model_revision,
        "prompts": prompts if prompts is not None else load_prompts(),
        "clip_seconds": clip_seconds,
        "boundary_slack": boundary_slack,
        "fps": fps,
        "subtitle_offset": subtitle_offset,
        "max_pixels": max_pixels,
        "copy_video": copy_video,
        "schema_version": SCHEMA_VERSION,
    }
    index = prepare_index(
        video_path, srt_path, recipe, memory_directory=memory_directory, output=output
    )
    manifest = index.manifest
    if manifest["status"] == "complete":
        print(f"Already indexed: {index.directory}", file=sys.stderr)
        return index.directory
    print(
        f"Indexing {manifest['duration']}: {len(index.segments)} non-overlapping segments, {fps:g} FPS, {len(recipe['prompts']['visual'])} visual prompts.",
        file=sys.stderr,
        flush=True,
    )
    try:
        model = EmbeddingModel(
            model_name,
            device=device,
            batch_size=batch_size,
            max_pixels=max_pixels,
            revision=manifest["model"].get("revision"),
            trust_remote_code=trust_remote_code,
        )
        manifest["model"] = {
            "model_id": model.model_id,
            "revision": model.revision,
            "dimension": model.dimension,
            "trust_remote_code": model.trust_remote_code,
        }
        manifest["status"] = "building"
        manifest.pop("error", None)
        manifest["dependencies"] = {
            name: version(name)
            for name in ("torch", "torchcodec", "transformers", "sentence-transformers")
        }
        write_json(index.directory / "manifest.json", manifest)
        embed_segments(index, model, decode_device=decode_device)
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        write_json(index.directory / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json(index.directory / "manifest.json", manifest)
    return index.directory
