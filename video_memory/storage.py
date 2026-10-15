"""Readable metadata and safe PyTorch tensors; no model is needed to inspect evidence."""

import json
from pathlib import Path

import torch

SCHEMA_VERSION = 1


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_index(directory: Path, *, require_complete: bool = True) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("This index uses an unsupported schema version.")
    if require_complete and manifest["status"] != "complete":
        raise ValueError(
            "This index is incomplete. Run its original index command again to resume."
        )
    return manifest


def get_clip(directory: Path, clip_id: str | int) -> dict:
    directory = directory.expanduser().resolve()
    manifest = load_index(directory)
    clip_id = f"{int(clip_id):06}"
    reference = next(
        (clip for clip in manifest["clips"] if clip["clip_id"] == clip_id), None
    )
    if reference is None:
        raise ValueError(f"Clip {clip_id} does not exist in this index.")
    clip_directory = directory / reference["directory"]
    clip = json.loads((clip_directory / "metadata.json").read_text(encoding="utf-8"))
    clip["video_id"] = manifest["video_id"]
    clip["video_path"] = (
        str((directory / manifest["video_copy"]).resolve())
        if manifest.get("video_copy")
        else manifest["source_video"]
    )
    clip["original_srt"] = str((directory / "source.srt").resolve())
    clip["transcript_path"] = str((clip_directory / "transcript.txt").resolve())
    clip["subtitle_path"] = str((clip_directory / "subtitles.srt").resolve())
    for frame in clip["frames"]:
        frame["path"] = str((clip_directory / frame["path"]).resolve())
    return clip


def save_vectors(
    path: Path, frame_vectors: torch.Tensor, transcript_vector: torch.Tensor | None
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "frames": frame_vectors.cpu(),
            "transcript": transcript_vector.cpu()
            if transcript_vector is not None
            else None,
        },
        temporary,
    )
    temporary.replace(path)


def list_indexes(memory_directory: Path) -> list[dict]:
    indexes = []
    for path in sorted(memory_directory.glob("*/manifest.json")):
        manifest = load_index(path.parent, require_complete=False)
        indexes.append(
            {
                "video_id": manifest["video_id"],
                "index": str(path.parent.resolve()),
                "source_video": manifest["source_video"],
                "status": manifest["status"],
                "clips": len(manifest["clips"]),
                "completed_clips": manifest["completed_clips"],
                "duration": manifest["duration"],
                "model_id": manifest["model"]["model_id"],
            }
        )
    return indexes
