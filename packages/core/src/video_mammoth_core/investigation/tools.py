from collections.abc import Callable

from pydantic import JsonValue
from video_mammoth_models.contracts import EmbeddingClient

from ..contracts import (
    EvidencePage,
    NotebookState,
    OpenClipAction,
    OpenNeighborAction,
    PreparedEvidence,
    ReadNotebookAction,
    ReadOutlineAction,
    ReadSourcesAction,
    SearchAction,
    SearchRequest,
    SearchResult,
    SourceReference,
    ToolAction,
    WriteNotebookAction,
)
from ..retrieval.search import search_index
from ..storage.embeddings import EmbeddingStore
from .evidence import EvidenceReader, page_result
from .notebook import read_notebook


async def execute_search(
    action: SearchAction,
    notebook: NotebookState,
    evidence: PreparedEvidence,
    *,
    store: EmbeddingStore,
    client: EmbeddingClient,
    general_instruction: str,
) -> SearchResult:
    """Run one model-requested search against this investigation's collection."""
    if (
        notebook.preparation_id != evidence.preparation_id
        or notebook.collection_id != store.metadata.collection_id
    ):
        raise ValueError(
            "Notebook, evidence, and embeddings must share the same scope."
        )

    request = SearchRequest(**action.model_dump(exclude={"tool"}))
    return await search_index(
        request,
        evidence=evidence,
        store=store,
        client=client,
        general_instruction=general_instruction,
    )


def execute_tool(
    action: ToolAction,
    notebook: NotebookState,
    evidence: PreparedEvidence,
    *,
    presented_sources: tuple[SourceReference, ...] = (),
    max_page_chars: int = 2000,
    reader: EvidenceReader | None = None,
    can_show: Callable[[EvidencePage], bool] | None = None,
) -> dict[str, JsonValue]:
    """Prepare a notebook operation; record_step commits the write."""
    if notebook.preparation_id != evidence.preparation_id:
        raise ValueError("Notebook and evidence belong to different preparations.")

    try:
        if isinstance(
            action,
            (OpenClipAction, OpenNeighborAction, ReadSourcesAction, ReadOutlineAction),
        ):
            if reader is None or can_show is None:
                raise ValueError(
                    "Evidence reads require a reader and request-budget checker."
                )
            return page_result(reader.read(action, can_show))

        if isinstance(action, WriteNotebookAction):
            _check_sources(action, evidence, presented_sources)

            entry_id = (
                len(notebook.entries) if action.entry_id is None else action.entry_id
            )

            if action.entry_id is not None and entry_id >= len(notebook.entries):
                raise ValueError(f"Unknown notebook entry: {entry_id}.")

            return {
                "tool": action.tool,
                "status": "ok",
                "entry_id": entry_id,
                "operation": ("created" if action.entry_id is None else "updated"),
            }

        if isinstance(action, ReadNotebookAction):
            return read_notebook(
                notebook,
                action,
                max_chars=max_page_chars,
            )

        return {
            "tool": action.tool,
            "status": "unavailable",
            "message": "This action is not connected to an executor yet.",
        }

    except (OSError, ValueError) as error:
        return {
            "tool": action.tool,
            "status": "error",
            "message": str(error),
        }


def _check_sources(
    action: WriteNotebookAction,
    evidence: PreparedEvidence,
    presented_sources: tuple[SourceReference, ...],
) -> None:
    available = {
        "frame": {frame.frame_id for frame in evidence.frames},
        "cue": {cue.cue_id for cue in evidence.cues},
    }

    if any(
        source.source_id not in available[source.source_kind]
        for source in action.sources
    ):
        raise ValueError("A notebook source does not exist in the prepared evidence.")

    if action.kind in {"observation", "contradiction"} and (
        set(action.sources) - set(presented_sources)
    ):
        raise ValueError(
            "These sources have not been presented as original evidence. "
            "Inspect them first; use a plan or hypothesis for unverified ideas."
        )
