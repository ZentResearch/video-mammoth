import asyncio
import tomllib
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from itertools import batched
from pathlib import Path
from uuid import UUID

from video_mammoth_models.content import image_from_file
from video_mammoth_models.contracts import (
    EmbeddingClient,
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResult,
    Text,
)

from ..contracts import (
    EmbeddingBatch,
    EmbeddingPrompts,
    PreparedEvidence,
    SubtitleCue,
    VideoFrame,
)
from ..storage.embeddings import EmbeddingCollection, embedding_writer
from .segments import get_referenced_cues


async def embed_prepared(
    evidence: PreparedEvidence,
    client: EmbeddingClient,
    *,
    prompts: EmbeddingPrompts,
    output_directory: str | Path,
    request_batch_size: int = 16,
    chunk_rows: int = 4096,
) -> EmbeddingCollection:
    with embedding_writer(
        output_directory,
        evidence=evidence,
        prompts=prompts,
        chunk_rows=chunk_rows,
    ) as writer:
        async with aclosing(
            iter_embedding_batches(
                evidence,
                client,
                prompts=prompts,
                request_batch_size=request_batch_size,
            )
        ) as batches:
            async for batch in batches:
                writer.append(batch)

        return writer.finish()


async def iter_embedding_batches(
    evidence: PreparedEvidence,
    client: EmbeddingClient,
    *,
    prompts: EmbeddingPrompts,
    request_batch_size: int = 16,
) -> AsyncGenerator[EmbeddingBatch]:
    if type(request_batch_size) is not int or request_batch_size < 1:
        raise ValueError("request_batch_size must be a positive integer.")

    cues = get_referenced_cues(evidence)

    for frames in batched(evidence.frames, request_batch_size):
        items = await asyncio.to_thread(
            _frame_items,
            evidence.preparation_id,
            frames,
        )

        for prompt in prompts.visual:
            response = await _embed_items(
                client,
                items,
                prompt.instruction,
            )

            yield EmbeddingBatch(
                preparation_id=evidence.preparation_id,
                channel=prompt.name,
                instruction=prompt.instruction,
                source_ids=tuple(frame.frame_id for frame in frames),
                result=response,
            )

    for cue_batch in batched(cues, request_batch_size):
        response = await _embed_items(
            client,
            _cue_items(evidence.preparation_id, cue_batch),
            prompts.subtitle_instruction,
        )

        yield EmbeddingBatch(
            preparation_id=evidence.preparation_id,
            channel="subtitles",
            instruction=prompts.subtitle_instruction,
            source_ids=tuple(cue.cue_id for cue in cue_batch),
            result=response,
        )


def load_embedding_prompts(path: str | Path) -> EmbeddingPrompts:
    with Path(path).expanduser().open("rb") as source:
        data = tomllib.load(source)

    visual = data.get("visual")
    if not isinstance(visual, list):
        raise TypeError("The prompt file needs [[visual]] entries.")

    data["visual"] = tuple(visual)
    return EmbeddingPrompts.model_validate(data)


def _frame_items(
    preparation_id: UUID,
    frames: Sequence[VideoFrame],
) -> tuple[EmbeddingItem, ...]:
    return tuple(
        EmbeddingItem(
            id=f"{preparation_id}:frame:{frame.frame_id}",
            parts=(image_from_file(frame.image_path),),
        )
        for frame in frames
    )


def _cue_items(
    preparation_id: UUID,
    cues: Sequence[SubtitleCue],
) -> tuple[EmbeddingItem, ...]:
    return tuple(
        EmbeddingItem(
            id=f"{preparation_id}:cue:{cue.cue_id}",
            parts=(Text(text=cue.text),),
        )
        for cue in cues
    )


async def _embed_items(
    client: EmbeddingClient,
    items: tuple[EmbeddingItem, ...],
    instruction: str,
) -> EmbeddingResult:
    request = EmbeddingRequest(
        items=items,
        purpose="document",
        instruction=instruction,
    )

    result = await client.embed(request)
    return result
