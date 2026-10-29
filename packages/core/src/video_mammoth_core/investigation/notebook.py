import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from pydantic import JsonValue

from ..contracts import (
    EvidenceView,
    NotebookEntry,
    NotebookEvent,
    NotebookState,
    ReadNotebookAction,
    ToolAction,
    WriteNotebookAction,
)


def open_notebook(
    directory: str | Path,
    *,
    question: str,
    preparation_id: UUID,
    collection_id: UUID,
) -> NotebookState:
    requested = NotebookState(
        question=question,
        preparation_id=preparation_id,
        collection_id=collection_id,
    )

    directory = Path(directory).expanduser().resolve()
    path = directory / "state.json"

    if path.exists():
        saved = _read_state(path)
        _check_identity(saved, requested)
        return _recover_notebook(directory, saved)

    directory.mkdir(parents=True, exist_ok=True)
    _write_state(path, requested)
    return requested


def save_notebook(
    state: NotebookState,
    directory: str | Path,
) -> Path:
    # Revalidate changes before writing, including model_copy updates.
    state = NotebookState.model_validate(state.model_dump())

    path = Path(directory).expanduser().resolve(strict=True) / "state.json"
    saved = _read_state(path)
    _check_identity(saved, state)

    if saved.run_id != state.run_id:
        raise ValueError("Notebook run_id does not match the saved run.")

    if state.step_count < saved.step_count:
        raise ValueError("Notebook step_count must not move backwards.")

    _write_state(path, state)
    return path


def _read_state(path: Path) -> NotebookState:
    return NotebookState.model_validate_json(path.read_text(encoding="utf-8"))


def _check_identity(
    saved: NotebookState,
    requested: NotebookState,
) -> None:
    for name in ("question", "preparation_id", "collection_id"):
        if getattr(saved, name) != getattr(requested, name):
            raise ValueError(f"Notebook {name} does not match this investigation.")


def _write_state(path: Path, state: NotebookState) -> None:
    temporary: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".state-",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(state.model_dump_json(indent=2))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_step(
    notebook: NotebookState,
    action: ToolAction,
    result: dict[str, JsonValue],
    directory: str | Path,
    *,
    presented: tuple[EvidenceView, ...] = (),
) -> NotebookState:
    """Commit one action/result, then save the updated notebook."""
    directory = Path(directory).expanduser().resolve(strict=True)
    if _read_state(directory / "state.json") != notebook:
        raise ValueError("Notebook changed; reload it before recording another step.")

    event = NotebookEvent(
        run_id=notebook.run_id,
        step_count=notebook.step_count + 1,
        action=action,
        result=result,
        presented=presented,
    )
    updated = _apply_event(notebook, event)

    with (directory / "events.jsonl").open("ab") as output:
        output.write((event.model_dump_json() + "\n").encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())

    save_notebook(updated, directory)
    return updated


def read_notebook(
    notebook: NotebookState,
    action: ReadNotebookAction,
    *,
    max_chars: int = 2000,
) -> dict[str, JsonValue]:
    """List note previews, or page through one complete saved entry."""
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer.")

    if action.entry_id is None:
        content = [
            {
                "entry_id": entry.entry_id,
                "kind": entry.kind,
                "status": entry.status,
                "preview": entry.text[:160],
                "text_chars": len(entry.text),
            }
            for entry in notebook.entries
        ]
    else:
        if action.entry_id >= len(notebook.entries):
            raise ValueError(f"Unknown notebook entry: {action.entry_id}.")
        content = notebook.entries[action.entry_id].model_dump(mode="json")

    text = json.dumps(content, ensure_ascii=False, indent=2)

    # Reads don't change this revision. Writing a note does.
    revision = max(
        (entry.updated_step for entry in notebook.entries),
        default=0,
    )
    target = "all" if action.entry_id is None else str(action.entry_id)
    binding = f"{notebook.run_id}:{revision}:{target}"

    start = 0
    if action.cursor is not None:
        prefix, separator, offset = action.cursor.rpartition(":")

        if not separator or prefix != binding or not offset.isdecimal():
            raise ValueError("Cursor belongs to another notebook, target, or revision.")

        start = int(offset)
        if not 0 <= start < len(text):
            raise ValueError("Cursor offset is outside this notebook response.")

    end = min(start + max_chars, len(text))

    return {
        "tool": "read_notebook",
        "status": "ok",
        "entry_id": action.entry_id,
        "text": text[start:end],
        "offset": start,
        "total_chars": len(text),
        "next_cursor": f"{binding}:{end}" if end < len(text) else None,
    }


def _apply_event(
    notebook: NotebookState,
    event: NotebookEvent,
) -> NotebookState:
    if event.run_id != notebook.run_id or event.step_count != notebook.step_count + 1:
        raise ValueError(
            "Notebook events must belong to this run and follow step order."
        )

    if event.result.get("tool") != event.action.tool or event.result.get(
        "status"
    ) not in {"ok", "error", "unavailable"}:
        raise ValueError("Tool result does not match the recorded action.")

    entries = list(notebook.entries)
    action = event.action

    if isinstance(action, WriteNotebookAction) and event.result["status"] == "ok":
        entry_id = len(entries) if action.entry_id is None else action.entry_id

        if action.entry_id is not None and entry_id >= len(entries):
            raise ValueError(f"Unknown notebook entry: {entry_id}.")

        if event.result.get("entry_id") != entry_id:
            raise ValueError("The write result refers to a different notebook entry.")

        entry = NotebookEntry(
            entry_id=entry_id,
            kind=action.kind,
            text=action.text,
            sources=tuple(dict.fromkeys(action.sources)),
            status=action.status,
            updated_step=event.step_count,
        )

        if action.entry_id is None:
            entries.append(entry)
        else:
            entries[entry_id] = entry

    data = notebook.model_dump()
    data.update(
        step_count=event.step_count,
        entries=tuple(entry.model_dump() for entry in entries),
        presented=tuple(
            view.model_dump()
            for view in dict.fromkeys(notebook.presented + event.presented)
        ),
    )
    return NotebookState.model_validate(data)


def _recover_notebook(
    directory: Path,
    notebook: NotebookState,
) -> NotebookState:
    """Replay committed events missing from the state snapshot."""
    path = directory / "events.jsonl"
    if not path.exists():
        return notebook

    recovered = notebook
    previous_step: int | None = None

    with path.open("r+b") as source:
        while True:
            position = source.tell()
            line = source.readline()

            if not line:
                break

            # A partial final line was never fully committed.
            if not line.endswith(b"\n"):
                source.truncate(position)
                source.flush()
                os.fsync(source.fileno())
                break

            event = NotebookEvent.model_validate_json(line)

            if event.run_id != notebook.run_id:
                raise ValueError("Event history belongs to another investigation.")

            if previous_step is not None and event.step_count != previous_step + 1:
                raise ValueError("Event history has missing or repeated steps.")

            previous_step = event.step_count

            if event.step_count > recovered.step_count:
                recovered = _apply_event(recovered, event)

    if previous_step is not None and previous_step != recovered.step_count:
        raise ValueError("Notebook state is ahead of its event history.")

    if recovered != notebook:
        save_notebook(recovered, directory)

    return recovered
