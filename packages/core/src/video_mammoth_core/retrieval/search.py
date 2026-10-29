from collections.abc import Mapping
from typing import Literal

import torch
from video_mammoth_models.contracts import (
    EmbeddingClient,
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResult,
    Text,
)

from ..contracts import (
    ClipCandidate,
    EmbeddingPrompts,
    PreparedClip,
    PreparedEvidence,
    SearchRequest,
    SearchResult,
    SourceMatch,
    SubtitleCue,
    VideoFrame,
)
from ..storage.embeddings import EmbeddingCollection, EmbeddingStore

type ChannelScores = dict[str, dict[int, float]]


async def search_index(
    request: SearchRequest,
    *,
    evidence: PreparedEvidence,
    store: EmbeddingStore,
    client: EmbeddingClient,
    general_instruction: str,
) -> SearchResult:
    metadata = store.metadata

    if metadata.preparation_id != evidence.preparation_id:
        raise ValueError("Evidence and embeddings belong to different preparations.")

    if metadata.normalization != "l2":
        raise ValueError("Search currently requires L2-normalized embeddings.")

    channels = _select_channels(request.channels, metadata.prompts)
    instruction = _resolve_instruction(request, metadata.prompts, general_instruction)

    response = await _embed_query(client, request.query, instruction, metadata)
    query_vector = torch.tensor(
        response.embeddings[0].vector,
        dtype=torch.float32,
        device="cpu",
    )

    scores = _score_chunks(store, query_vector, channels)
    candidates = _rank_clips(evidence, scores, request.top_k)

    return SearchResult(
        preparation_id=evidence.preparation_id,
        collection_id=metadata.collection_id,
        request=request,
        instruction=instruction,
        channels=channels,
        model_identity=response.model_identity,
        candidates=candidates,
    )


def _select_channels(
    requested: tuple[str, ...] | None,
    prompts: EmbeddingPrompts,
) -> tuple[str, ...]:
    available = tuple(prompt.name for prompt in prompts.visual) + ("subtitles",)

    if requested is None:
        return available

    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown search channels: {sorted(unknown)}")

    return tuple(name for name in available if name in requested)


def _resolve_instruction(
    request: SearchRequest,
    prompts: EmbeddingPrompts,
    general_instruction: str,
) -> str:
    if request.instruction is not None:
        return request.instruction

    if request.style == "general":
        if not isinstance(general_instruction, str) or not general_instruction.strip():
            raise ValueError("general_instruction must be nonblank text.")

        return general_instruction

    if request.style == "subtitles":
        return prompts.subtitle_query_instruction

    for prompt in prompts.visual:
        if prompt.name == request.style:
            return prompt.query_instruction

    raise ValueError(f"Unknown search style: {request.style!r}")


async def _embed_query(
    client: EmbeddingClient,
    query: str,
    instruction: str,
    metadata: EmbeddingCollection,
) -> EmbeddingResult:
    request = EmbeddingRequest(
        items=(EmbeddingItem(id="query", parts=(Text(text=query),)),),
        purpose="query",
        instruction=instruction,
    )
    result = await client.embed(request)
    result.validate_for(request)

    if result.dimensions != metadata.dimensions:
        raise ValueError("Query and saved vectors have different dimensions.")

    if result.normalization != metadata.normalization:
        raise ValueError("Query and saved vectors use different normalization.")

    saved = metadata.model_identity
    returned = result.model_identity

    if returned.requested != saved.requested:
        raise ValueError("Query and saved vectors use different model IDs.")

    for name in ("returned", "revision"):
        old = getattr(saved, name)
        new = getattr(returned, name)
        if old is not None and new is not None and old != new:
            raise ValueError(f"Query model {name} differs from the saved collection.")

    return result


@torch.inference_mode()
def _score_chunks(
    store: EmbeddingStore,
    query_vector: torch.Tensor,
    channels: tuple[str, ...],
) -> ChannelScores:
    scores: ChannelScores = {channel: {} for channel in channels}

    for chunk in store.iter_chunks(channels):
        values = (chunk.vectors @ query_vector).tolist()

        for source_id, score in zip(chunk.ids.tolist(), values, strict=True):
            scores[chunk.channel][source_id] = score

    return scores


def _rank_clips(
    evidence: PreparedEvidence,
    scores: ChannelScores,
    top_k: int,
) -> tuple[ClipCandidate, ...]:
    ranked: list[tuple[float, PreparedClip]] = []

    for clip in evidence.clips:
        score = _clip_score(clip, scores)
        if score is not None:
            ranked.append((score, clip))

    ranked.sort(key=lambda item: (-item[0], item[1].clip_id))

    frames = {frame.frame_id: frame for frame in evidence.frames}
    cues = {cue.cue_id: cue for cue in evidence.cues}

    return tuple(
        ClipCandidate(
            **clip.model_dump(),
            score=score,
            matches=_clip_matches(clip, scores, frames, cues),
        )
        for score, clip in ranked[:top_k]
    )


def _clip_score(
    clip: PreparedClip,
    scores: ChannelScores,
) -> float | None:
    best = None

    for channel, values in scores.items():
        source_ids = clip.cue_ids if channel == "subtitles" else clip.frame_ids

        for source_id in source_ids:
            score = values.get(source_id)
            if score is not None and (best is None or score > best):
                best = score

    return best


def _clip_matches(
    clip: PreparedClip,
    scores: ChannelScores,
    frames: Mapping[int, VideoFrame],
    cues: Mapping[int, SubtitleCue],
) -> tuple[SourceMatch, ...]:
    matches: list[SourceMatch] = []

    for channel, values in scores.items():
        source_ids = clip.cue_ids if channel == "subtitles" else clip.frame_ids

        for source_id in source_ids:
            score = values.get(source_id)
            if score is None:
                continue

            kind: Literal["frame", "cue"]
            if channel == "subtitles":
                cue = cues[source_id]
                kind = "cue"
                start = cue.start_seconds
                end = cue.end_seconds
            else:
                frame = frames[source_id]
                kind = "frame"
                start = frame.presentation_seconds
                end = start + frame.duration_seconds

            matches.append(
                SourceMatch(
                    channel=channel,
                    source_kind=kind,
                    source_id=source_id,
                    start_seconds=start,
                    end_seconds=end,
                    score=score,
                )
            )

    matches.sort(key=lambda match: (-match.score, match.channel, match.source_id))
    return tuple(matches)
