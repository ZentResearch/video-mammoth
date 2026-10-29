from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, NamedTuple
from uuid import UUID, uuid4

import torch
from pydantic import Field
from video_mammoth_models.contracts import ModelIdentity

from ..contracts import (
    Contract,
    EmbeddingBatch,
    EmbeddingPrompts,
    PreparedEvidence,
)
from ..indexing.segments import get_referenced_cues


class ModelSpec(Contract):
    model_identity: ModelIdentity
    dimensions: int = Field(gt=0)
    normalization: Literal["l2", "none"]


class ChunkInfo(Contract):
    channel: str = Field(min_length=1)
    path: str = Field(min_length=1)
    rows: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EmbeddingCollection(ModelSpec):
    schema_version: Literal[2] = 2
    collection_id: UUID
    configuration_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_id: UUID
    prompts: EmbeddingPrompts
    dtype: Literal["float32"] = "float32"
    chunk_rows: int = Field(gt=0)
    chunks: tuple[ChunkInfo, ...] = Field(min_length=1)


class StoredChunk(NamedTuple):
    channel: str
    ids: torch.Tensor
    vectors: torch.Tensor


@dataclass(slots=True)
class _Buffer:
    ids: torch.Tensor
    vectors: torch.Tensor
    used: int = 0


@contextmanager
def embedding_writer(
    directory: str | Path,
    *,
    evidence: PreparedEvidence,
    prompts: EmbeddingPrompts,
    chunk_rows: int = 4096,
) -> Generator[EmbeddingWriter]:
    _integer(chunk_rows, minimum=1)

    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    expected = _source_ids(evidence, prompts)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}-",
    ) as temporary:
        writer = EmbeddingWriter(
            Path(temporary),
            destination,
            evidence.preparation_id,
            prompts,
            expected,
            chunk_rows,
        )
        try:
            yield writer
        finally:
            writer.closed = True


def open_embeddings(
    directory: str | Path,
    *,
    evidence: PreparedEvidence,
    cache_bytes: int = 256 * 1024 * 1024,
) -> EmbeddingStore:
    return EmbeddingStore(
        directory,
        evidence=evidence,
        cache_bytes=cache_bytes,
    )


class EmbeddingWriter:
    def __init__(
        self,
        staging: Path,
        destination: Path,
        preparation_id: UUID,
        prompts: EmbeddingPrompts,
        expected: dict[str, tuple[int, ...]],
        chunk_rows: int,
    ) -> None:
        self.staging = staging
        self.destination = destination
        self.preparation_id = preparation_id
        self.prompts = prompts
        self.expected = expected
        self.chunk_rows = chunk_rows
        self.closed = False

        self._spec: ModelSpec | None = None
        self._counts = dict.fromkeys(expected, 0)
        self._numbers = dict.fromkeys(expected, 0)
        self._buffers: dict[str, _Buffer] = {}
        self._chunks: list[ChunkInfo] = []
        self._folders = _folders(prompts)
        self._instructions = {
            **{p.name: p.instruction for p in prompts.visual},
            "subtitles": prompts.subtitle_instruction,
        }

    def append(self, batch: EmbeddingBatch) -> None:
        if self.closed:
            raise RuntimeError("The writer is closed.")

        if batch.preparation_id != self.preparation_id:
            raise ValueError("Batch belongs to a different preparation.")

        channel = batch.channel
        if channel not in self.expected:
            raise ValueError(f"Unknown channel: {channel}")

        if batch.instruction != self._instructions[channel]:
            raise ValueError("Batch instruction does not match its channel.")

        start = self._counts[channel]
        end = start + len(batch.source_ids)

        if batch.source_ids != self.expected[channel][start:end]:
            raise ValueError("Batch IDs are missing, repeated, or out of order.")

        spec = ModelSpec(
            model_identity=batch.result.model_identity,
            dimensions=batch.result.dimensions,
            normalization=batch.result.normalization,
        )

        if self._spec is not None and spec != self._spec:
            raise ValueError("Model metadata changed during embedding.")
        self._spec = spec

        vectors = torch.tensor(
            [item.vector for item in batch.result.embeddings],
            dtype=torch.float32,
            device="cpu",
        )
        _check_vectors(vectors, len(batch.source_ids), spec)

        if channel not in self._buffers:
            self._buffers[channel] = _Buffer(
                ids=torch.empty(
                    self.chunk_rows,
                    dtype=torch.int64,
                    device="cpu",
                ),
                vectors=torch.empty(
                    (self.chunk_rows, spec.dimensions),
                    dtype=torch.float32,
                    device="cpu",
                ),
            )

        buffer = self._buffers[channel]
        ids = torch.tensor(
            batch.source_ids,
            dtype=torch.int64,
            device="cpu",
        )

        cursor = 0
        while cursor < len(ids):
            take = min(
                len(ids) - cursor,
                self.chunk_rows - buffer.used,
            )
            stop = buffer.used + take

            buffer.ids[buffer.used : stop].copy_(ids[cursor : cursor + take])
            buffer.vectors[buffer.used : stop].copy_(vectors[cursor : cursor + take])

            buffer.used = stop
            cursor += take

            if buffer.used == self.chunk_rows:
                self._flush(channel)

        self._counts[channel] = end

    def finish(self) -> EmbeddingCollection:
        if self.closed:
            raise RuntimeError("The writer is closed.")

        if self._spec is None or any(
            self._counts[name] != len(ids) for name, ids in self.expected.items()
        ):
            raise ValueError("The collection does not cover all expected evidence.")

        for channel in self._buffers:
            self._flush(channel)

        collection = EmbeddingCollection(
            **self._spec.model_dump(),
            collection_id=uuid4(),
            configuration_id=_configuration_id(
                self.prompts,
                self._spec,
            ),
            preparation_id=self.preparation_id,
            prompts=self.prompts,
            chunk_rows=self.chunk_rows,
            chunks=tuple(self._chunks),
        )

        with (self.staging / "metadata.json").open(
            "x",
            encoding="utf-8",
        ) as output:
            output.write(collection.model_dump_json(indent=2))
            output.flush()
            os.fsync(output.fileno())

        if self.destination.exists():
            raise FileExistsError(f"Output already exists: {self.destination}")

        self.staging.rename(self.destination)
        self.closed = True
        return collection

    def _flush(self, channel: str) -> None:
        buffer = self._buffers[channel]
        if not buffer.used:
            return

        relative = f"{self._folders[channel]}/{self._numbers[channel]:06d}.pt"
        path = self.staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)

        id_key = "cue_ids" if channel == "subtitles" else "frame_ids"

        with path.open("xb") as output:
            # Clone slices to exclude unused buffer storage.
            torch.save(
                {
                    id_key: buffer.ids[: buffer.used].clone(),
                    "vectors": buffer.vectors[: buffer.used].clone(),
                },
                output,
            )
            output.flush()
            os.fsync(output.fileno())

        with path.open("rb") as source:
            checksum = file_digest(
                source,
                "sha256",
            ).hexdigest()

        self._chunks.append(
            ChunkInfo(
                channel=channel,
                path=relative,
                rows=buffer.used,
                sha256=checksum,
            )
        )

        self._numbers[channel] += 1
        buffer.used = 0


class EmbeddingStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        evidence: PreparedEvidence,
        cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        _integer(cache_bytes, minimum=0)

        self.directory = Path(directory).expanduser().resolve(strict=True)
        self.metadata = EmbeddingCollection.model_validate_json(
            (self.directory / "metadata.json").read_text(encoding="utf-8")
        )

        metadata = self.metadata
        if metadata.preparation_id != evidence.preparation_id:
            raise ValueError("Collection belongs to a different preparation.")

        if metadata.configuration_id != _configuration_id(
            metadata.prompts,
            metadata,
        ):
            raise ValueError("Configuration identity does not match metadata.")

        self._expected = _source_ids(
            evidence,
            metadata.prompts,
        )
        folders = _folders(metadata.prompts)
        counts = dict.fromkeys(self._expected, 0)
        numbers = dict.fromkeys(self._expected, 0)

        self._entries: list[tuple[ChunkInfo, int]] = []

        for info in metadata.chunks:
            if info.channel not in counts or info.rows > metadata.chunk_rows:
                raise ValueError("Unknown channel or invalid chunk size.")

            path = f"{folders[info.channel]}/{numbers[info.channel]:06d}.pt"
            if info.path != path:
                raise ValueError("Unexpected chunk path or order.")

            self._entries.append((info, counts[info.channel]))
            counts[info.channel] += info.rows
            numbers[info.channel] += 1

        if any(counts[name] != len(ids) for name, ids in self._expected.items()):
            raise ValueError("Chunk metadata has incomplete or excess coverage.")

        self.cache_bytes = cache_bytes
        self.cached_bytes = 0
        self._cache: OrderedDict[str, StoredChunk] = OrderedDict()

    def iter_chunks(
        self,
        channels: Sequence[str] | None = None,
    ) -> Iterator[StoredChunk]:
        if isinstance(channels, str):
            raise TypeError("Provide a list of channel names.")

        selected = set(self._expected if channels is None else channels)
        if not selected or selected - self._expected.keys():
            raise ValueError("Select one or more configured channels.")

        for info, start in self._entries:
            if info.channel not in selected:
                continue

            if info.path in self._cache:
                self._cache.move_to_end(info.path)
                yield self._cache[info.path]
                continue

            chunk = self._read(info, start)
            size = chunk.ids.numel() * 8 + chunk.vectors.numel() * 4

            while self._cache and self.cached_bytes + size > self.cache_bytes:
                _, old = self._cache.popitem(last=False)
                self.cached_bytes -= old.ids.numel() * 8 + old.vectors.numel() * 4

            if size <= self.cache_bytes:
                self._cache[info.path] = chunk
                self.cached_bytes += size

            yield chunk

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cached_bytes = 0

    def _read(
        self,
        info: ChunkInfo,
        start: int,
    ) -> StoredChunk:
        path = (self.directory / info.path).resolve(strict=True)

        if not path.is_relative_to(self.directory):
            raise ValueError("Chunk path escapes the collection directory.")

        with path.open("rb") as source:
            if (
                file_digest(
                    source,
                    "sha256",
                ).hexdigest()
                != info.sha256
            ):
                raise ValueError(f"Chunk checksum mismatch: {info.path}")

            source.seek(0)
            data = torch.load(
                source,
                map_location="cpu",
                weights_only=True,
            )

        id_key = "cue_ids" if info.channel == "subtitles" else "frame_ids"

        if not isinstance(data, dict) or set(data) != {id_key, "vectors"}:
            raise ValueError("Unexpected chunk contents.")

        ids = data[id_key]
        vectors = data["vectors"]

        expected = torch.tensor(
            self._expected[info.channel][start : start + info.rows],
            dtype=torch.int64,
            device="cpu",
        )

        if (
            not isinstance(ids, torch.Tensor)
            or ids.device.type != "cpu"
            or ids.layout != torch.strided
            or ids.dtype != torch.int64
            or ids.shape != (info.rows,)
            or ids.storage_offset() != 0
            or ids.untyped_storage().nbytes() != ids.numel() * 8
            or not torch.equal(ids, expected)
        ):
            raise ValueError("Chunk IDs do not match the prepared evidence.")

        _check_vectors(
            vectors,
            info.rows,
            self.metadata,
        )

        return StoredChunk(info.channel, ids, vectors)


def _source_ids(
    evidence: PreparedEvidence,
    prompts: EmbeddingPrompts,
) -> dict[str, tuple[int, ...]]:
    frames = tuple(frame.frame_id for frame in evidence.frames)

    if len(set(frames)) != len(frames):
        raise ValueError("Source frame IDs must be unique.")

    cues = tuple(cue.cue_id for cue in get_referenced_cues(evidence))

    return {
        **{prompt.name: frames for prompt in prompts.visual},
        "subtitles": cues,
    }


def _folders(
    prompts: EmbeddingPrompts,
) -> dict[str, str]:
    # Arbitrary channel names never become filesystem paths.
    return {
        **{prompt.name: f"visual/{i:03d}" for i, prompt in enumerate(prompts.visual)},
        "subtitles": "subtitles",
    }


def _configuration_id(
    prompts: EmbeddingPrompts,
    spec: ModelSpec,
) -> str:
    value = {
        "prompts": prompts.model_dump(mode="json"),
        "model_identity": spec.model_identity.model_dump(mode="json"),
        "dimensions": spec.dimensions,
        "normalization": spec.normalization,
        "dtype": "float32",
    }

    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _check_vectors(
    value: object,
    rows: int,
    spec: ModelSpec,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.dtype != torch.float32
        or value.shape != (rows, spec.dimensions)
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.untyped_storage().nbytes() != value.numel() * 4
    ):
        raise ValueError("Expected a compact CPU float32 vector matrix.")

    if not torch.isfinite(value).all().item():
        raise ValueError("Vectors contain non-finite values.")

    if (
        spec.normalization == "l2"
        and not torch.isclose(
            torch.linalg.vector_norm(value, dim=1),
            torch.ones(
                rows,
                dtype=torch.float32,
                device="cpu",
            ),
            rtol=1e-4,
            atol=1e-6,
        )
        .all()
        .item()
    ):
        raise ValueError("Vectors are not L2 normalized.")


def _integer(value: int, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Expected an integer >= {minimum}.")
