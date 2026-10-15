"""Rank unique segments using transcript and per-prompt frame similarities."""

import json
from pathlib import Path

import torch

from .encoder import EmbeddingModel
from .storage import load_index


def encode_search_query(
    model: EmbeddingModel,
    query: str,
    prompts: dict,
    channels: set[str],
    instruction_override: str | None = None,
) -> dict[str, torch.Tensor]:
    instructions = {"transcript": prompts["transcript_query_instruction"]}
    instructions.update(
        (prompt["name"], prompt["query_instruction"]) for prompt in prompts["visual"]
    )
    return {
        name: model.encode(
            [query],
            instruction_override if instruction_override is not None else instruction,
            is_query=True,
        )[0]
        for name, instruction in instructions.items()
        if name in channels
    }


def score_clip(
    directory: Path,
    reference: dict,
    query_vectors: dict[str, torch.Tensor],
    visual_prompts: list[dict],
    *,
    frame_top_n: int,
    device: torch.device,
) -> dict | None:
    clip_directory = directory / reference["directory"]
    vectors = torch.load(
        clip_directory / "embeddings.pt", map_location="cpu", weights_only=True
    )
    metadata = json.loads(
        (clip_directory / "metadata.json").read_text(encoding="utf-8")
    )
    scores, evidence = {}, []
    if "transcript" in query_vectors and vectors["transcript"] is not None:
        scores["transcript"] = float(
            vectors["transcript"].to(device) @ query_vectors["transcript"]
        )
    if len(vectors["frames"]):
        frame_vectors = vectors["frames"].to(device)
        for channel_index, prompt in enumerate(visual_prompts):
            name = prompt["name"]
            if name not in query_vectors:
                continue
            similarities = frame_vectors[:, channel_index] @ query_vectors[name]
            values, indices = similarities.topk(min(frame_top_n, len(similarities)))
            scores[name] = float(values.mean())
            evidence.append(
                {
                    "channel": name,
                    "similarity": float(values[0]),
                    **metadata["frames"][int(indices[0])],
                }
            )
    if not scores:
        return None
    for frame in evidence:
        frame["path"] = str((clip_directory / frame["path"]).resolve())
    return {
        "clip_id": reference["clip_id"],
        "start_seconds": reference["start_seconds"],
        "end_seconds": reference["end_seconds"],
        "start": reference["start"],
        "end": reference["end"],
        "score": max(scores.values()),
        "channel_scores": scores,
        "best_channel": max(scores, key=scores.get),
        "frame_matches": sorted(
            evidence, key=lambda item: item["similarity"], reverse=True
        ),
        "transcript": metadata["transcript"],
        "metadata_path": str((clip_directory / "metadata.json").resolve()),
    }


@torch.inference_mode()
def search(
    directory: Path,
    query: str,
    *,
    top_k: int = 5,
    channels: list[str] | None = None,
    fusion: str = "max",
    frame_top_n: int = 1,
    device: str = "cuda",
    query_instruction: str | None = None,
    model: EmbeddingModel | None = None,
) -> dict:
    """Accept a loaded model for repeated agent calls without reloading GPU weights."""
    directory = directory.expanduser().resolve()
    manifest = load_index(directory)
    if (
        not query.strip()
        or top_k < 1
        or frame_top_n < 1
        or fusion not in {"max", "rrf"}
    ):
        raise ValueError(
            "Provide a nonempty query, positive top-k/frame-top-n, and max or rrf fusion."
        )
    prompts = manifest["recipe"]["prompts"]
    available = {"transcript", *(prompt["name"] for prompt in prompts["visual"])}
    selected = set(channels) if channels is not None else available
    if not selected or selected - available:
        raise ValueError(f"Available channels: {', '.join(sorted(available))}")
    model_config = manifest["model"]
    model = model or EmbeddingModel(
        model_config["model_id"],
        device=device,
        revision=model_config["revision"],
        max_pixels=manifest["recipe"]["max_pixels"],
        trust_remote_code=model_config["trust_remote_code"],
    )
    if (
        model.model_id != model_config["model_id"]
        or model.dimension != model_config["dimension"]
    ):
        raise ValueError(
            "Search must use the model and embedding dimensions saved in this index."
        )
    if model_config["revision"] and model.revision != model_config["revision"]:
        raise ValueError(
            "The supplied search model has a different checkpoint revision from this index."
        )

    query_vectors = encode_search_query(
        model, query, prompts, selected, query_instruction
    )
    candidates = []
    for reference in manifest["clips"]:
        candidate = score_clip(
            directory,
            reference,
            query_vectors,
            prompts["visual"],
            frame_top_n=frame_top_n,
            device=model.device,
        )
        if candidate is not None:
            candidates.append(candidate)
    if fusion == "rrf":
        for item in candidates:
            item["score"] = 0.0
        for name in sorted(selected):
            ranking = sorted(
                (item for item in candidates if name in item["channel_scores"]),
                key=lambda item: (-item["channel_scores"][name], item["clip_id"]),
            )
            for rank, item in enumerate(ranking, 1):
                item["score"] += 1 / (60 + rank)
    candidates.sort(key=lambda item: (-item["score"], item["clip_id"]))
    return {
        "video_id": manifest["video_id"],
        "index": str(directory),
        "source_video": manifest["source_video"],
        "query": query,
        "fusion": fusion,
        "channels": sorted(selected),
        "frame_top_n": frame_top_n,
        "total_clips": len(manifest["clips"]),
        "candidates": candidates[:top_k],
    }
