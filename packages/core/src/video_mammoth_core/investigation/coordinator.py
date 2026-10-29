import asyncio
import json
import math
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from uuid import uuid4

from pydantic import JsonValue, ValidationError
from video_mammoth_models.contracts import (
    EmbeddingClient,
    GenerationRequest,
    GenerationResult,
    Message,
    Text,
    VLMClient,
)
from video_mammoth_models.errors import ErrorKind, ModelClientError
from video_mammoth_models.factory import ModelClients

from ..contracts import (
    ActionRequest,
    AnswerReview,
    EvidencePage,
    EvidenceView,
    InitialSearchResult,
    InvestigationLimits,
    NotebookState,
    OpenClipAction,
    OpenNeighborAction,
    PreparedEvidence,
    ReadOutlineAction,
    ReadSourcesAction,
    RequestBudget,
    RequestFinishAction,
    SearchAction,
    SearchPlan,
    SearchRequest,
    SearchResult,
    ToolAction,
    VideoOutline,
    WriteNotebookAction,
)
from ..retrieval.search import search_index
from ..storage.embeddings import EmbeddingStore
from .budget import (
    ContextBudgetError,
    InvestigationLimitError,
    TokenCounterCache,
    input_tokens_fit,
)
from .context import build_decision_request
from .evidence import EvidenceReader, page_result
from .notebook import open_notebook, record_step, save_notebook
from .outline import TokenCounter, get_outline
from .planning import plan_searches
from .review import review_answer
from .tools import execute_search, execute_tool

# JSON tool results and investigation outcomes.
type Record = dict[str, JsonValue]


async def investigate(
    question: str,
    *,
    evidence: PreparedEvidence,
    store: EmbeddingStore,
    clients: ModelClients,
    run_directory: Path,
    budget: RequestBudget,
    max_steps: int,
    outline_directory: Path,
    outline_cache_identity: dict[str, JsonValue],
    count_tokens: TokenCounter,
    general_instruction: str,
    max_searches: int = 3,
    top_k_per_query: int = 5,
    rank_constant: int = 60,
    limits: InvestigationLimits | None = None,
) -> Record:
    if not question.strip():
        raise ValueError("The question must not be blank.")

    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    limits = limits if limits is not None else InvestigationLimits()

    notebook = _open_notebook(question, evidence, store, run_directory)
    run_directory = Path(run_directory).expanduser().resolve(strict=True)
    if notebook.status != "running":
        saved = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
        if (
            any(
                saved.get(name) != str(getattr(notebook, name))
                for name in ("run_id", "preparation_id", "collection_id")
            )
            or saved.get("question") != question
            or saved.get("status") != notebook.status
        ):
            raise ValueError("Saved outcome does not match the investigation.")
        return saved

    started = monotonic()
    if notebook.step_count >= max_steps:
        return _finish(
            notebook,
            {"status": "incomplete", "reason": "Step limit reached."},
            run_directory,
            evidence=evidence,
        )
    if not isinstance(count_tokens, TokenCounterCache):
        count_tokens = TokenCounterCache(count_tokens)
    vlm = _RunVLM(clients.vlm, limits.max_model_calls, started + limits.max_seconds)
    latest_result: Record | None = None
    latest_search: SearchResult | None = None
    last_review: Record = {}
    pending_action: ToolAction | None = None
    shown: tuple[EvidenceView, ...] = ()
    reviewed_views: list[EvidenceView] = []
    review_id = None
    outcome: Record = {"status": "incomplete", "reason": "Step limit reached."}
    seen: set[str] = set()
    no_progress = 0

    try:
        async with asyncio.timeout(limits.max_seconds):
            # 1. Get the overview, plan searches, and find initial candidates.
            outline = await get_outline(
                evidence,
                vlm,
                directory=outline_directory,
                cache_identity=outline_cache_identity,
                budget=budget,
                count_tokens=count_tokens,
            )
            plan = await plan_searches(
                question,
                outline,
                vlm,
                budget=budget,
                count_tokens=count_tokens,
                max_searches=max_searches,
                max_attempts=limits.max_attempts,
            )
            initial = await run_initial_searches(
                plan,
                evidence=evidence,
                store=store,
                client=clients.embeddings,
                general_instruction=general_instruction,
                top_k_per_query=top_k_per_query,
                rank_constant=rank_constant,
            )
            reader = EvidenceReader(evidence, outline, budget)
            for search in initial.searches:
                seen.update(_progress_keys(None, {}, (), search))
            for entry in notebook.entries:
                seen.add(
                    "note:"
                    + json.dumps(
                        entry.model_dump(
                            mode="json", include={"kind", "text", "sources", "status"}
                        ),
                        sort_keys=True,
                    )
                )

            for _ in range(max_steps - notebook.step_count):
                vlm.check_time()
                request, shown = _build_context(
                    notebook,
                    outline,
                    initial,
                    budget,
                    count_tokens=count_tokens,
                    reader=reader,
                    latest_result=latest_result,
                    latest_search=latest_search,
                )
                action = await _choose_action(
                    request,
                    vlm,
                    budget=budget,
                    count_tokens=count_tokens,
                    max_attempts=limits.max_attempts,
                )
                pending_action = action
                presented_sources = ()
                if isinstance(action, RequestFinishAction) or (
                    isinstance(action, WriteNotebookAction)
                    and action.kind in {"observation", "contradiction"}
                ):
                    presented_sources = reader.fully_presented(
                        notebook.presented + shown
                    )
                search = None
                review_id = None

                # 2. Execute one action. Completion re-reads original evidence.
                if isinstance(action, RequestFinishAction):
                    review_id = uuid4()
                    reviewed_views = []
                    last_review = {
                        "answer": action.answer,
                        "proposed_gaps": list(action.gaps),
                        "review_id": str(review_id),
                    }
                    result = await review_answer(
                        action,
                        notebook,
                        vlm,
                        reader=reader,
                        budget=budget,
                        count_tokens=count_tokens,
                        presented_sources=presented_sources,
                        directory=run_directory,
                        review_id=review_id,
                        max_attempts=limits.max_attempts,
                        reviewed_views=reviewed_views,
                    )
                    last_review.update(result)
                elif isinstance(action, SearchAction):
                    try:
                        search = await execute_search(
                            action,
                            notebook,
                            evidence,
                            store=store,
                            client=clients.embeddings,
                            general_instruction=general_instruction,
                        )
                    except (ModelClientError, OSError, ValueError) as error:
                        result = {
                            "tool": "search",
                            "status": "error",
                            "message": str(error),
                        }
                    else:
                        latest_search = search
                        result = {
                            "tool": "search",
                            "status": "ok",
                            "search": search.model_dump(mode="json"),
                        }
                else:
                    can_show = None
                    if isinstance(
                        action,
                        (
                            OpenClipAction,
                            OpenNeighborAction,
                            ReadSourcesAction,
                            ReadOutlineAction,
                        ),
                    ):
                        can_show = partial(
                            _page_fits,
                            notebook=notebook,
                            outline=outline,
                            initial=initial,
                            reader=reader,
                            budget=budget,
                            count_tokens=count_tokens,
                            latest_search=latest_search,
                        )
                    result = execute_tool(
                        action,
                        notebook,
                        evidence,
                        reader=reader,
                        can_show=can_show,
                        presented_sources=presented_sources,
                    )

                # 3. Persist the full result, then build a small next-step brief.
                notebook = record_step(
                    notebook, action, result, run_directory, presented=shown
                )
                pending_action = None
                keys = _progress_keys(action, result, shown, search)
                no_progress = 0 if keys - seen else no_progress + 1
                seen.update(keys)
                if (
                    isinstance(action, RequestFinishAction)
                    and result.get("status") == "ok"
                    and result["review"]["verdict"] == "supported"
                ):
                    outcome = result
                    break
                latest_result = _result_brief(result)
                if no_progress >= limits.max_no_progress_steps:
                    outcome = {
                        **last_review,
                        "status": "incomplete",
                        "reason": "No-progress limit reached.",
                    }
                    break
            else:
                outcome = {**last_review, **outcome}

    except (
        TimeoutError,
        ModelClientError,
        ContextBudgetError,
        RuntimeError,
    ) as error:
        reason = (
            "Time limit reached." if isinstance(error, TimeoutError) else str(error)
        )
        if pending_action is not None:
            failure: Record = {
                "tool": pending_action.tool,
                "status": "error",
                "message": reason,
            }
            if review_id is not None:
                failure["review_id"] = str(review_id)
                complete = set(reader.fully_presented(tuple(reviewed_views)))
                last_review.update(
                    reviewed_sources=[
                        source.model_dump(mode="json")
                        for source in pending_action.sources
                        if source in complete
                    ],
                    pending_sources=[
                        source.model_dump(mode="json")
                        for source in pending_action.sources
                        if source not in complete
                    ],
                )
                failure.update(
                    {
                        key: last_review[key]
                        for key in ("reviewed_sources", "pending_sources")
                    }
                )
            notebook = record_step(
                notebook, pending_action, failure, run_directory, presented=shown
            )
        outcome = {**last_review, "status": "incomplete", "reason": reason}

    outcome["model_calls"] = vlm.calls
    outcome["elapsed_seconds"] = monotonic() - started
    return _finish(
        notebook,
        outcome,
        run_directory,
        evidence=evidence,
    )


async def run_initial_searches(
    plan: SearchPlan,
    *,
    evidence: PreparedEvidence,
    store: EmbeddingStore,
    client: EmbeddingClient,
    general_instruction: str,
    top_k_per_query: int = 5,
    rank_constant: int = 60,
) -> InitialSearchResult:
    """Execute a plan and return its searches plus an ordered, unique clip queue."""
    if type(rank_constant) is not int or rank_constant < 0:
        raise ValueError("rank_constant must be a nonnegative integer.")

    searches: list[SearchResult] = []

    for query in plan.queries:
        request = SearchRequest(
            query=query,
            style="general",
            channels=None,
            instruction=None,
            top_k=top_k_per_query,
        )
        result = await search_index(
            request,
            evidence=evidence,
            store=store,
            client=client,
            general_instruction=general_instruction,
        )
        searches.append(result)

    results = tuple(searches)

    return InitialSearchResult(
        plan=plan,
        searches=results,
        clip_ids=_merge_clip_ids(results, rank_constant),
        rank_constant=rank_constant,
    )


def _merge_clip_ids(
    searches: tuple[SearchResult, ...],
    rank_constant: int,
) -> tuple[int, ...]:
    """Combine query rankings using reciprocal rank fusion."""
    scores: dict[int, float] = {}

    for search in searches:
        for rank, candidate in enumerate(search.candidates, start=1):
            clip_id = candidate.clip_id
            scores[clip_id] = scores.get(clip_id, 0.0) + 1.0 / (rank_constant + rank)

    return tuple(sorted(scores, key=lambda clip_id: (-scores[clip_id], clip_id)))


# Helpers in execution order.


def _open_notebook(
    question: str,
    evidence: PreparedEvidence,
    store: EmbeddingStore,
    run_directory: Path,
) -> NotebookState:
    if store.metadata.preparation_id != evidence.preparation_id:
        raise ValueError("Evidence and embeddings belong to different preparations.")

    return open_notebook(
        run_directory,
        question=question,
        preparation_id=evidence.preparation_id,
        collection_id=store.metadata.collection_id,
    )


def _build_context(
    notebook: NotebookState,
    outline: VideoOutline,
    initial: InitialSearchResult,
    budget: RequestBudget,
    *,
    count_tokens: TokenCounter,
    reader: EvidenceReader,
    latest_result: Record | None = None,
    latest_search: SearchResult | None = None,
    max_candidates: int = 8,
    max_source_matches: int = 2,
) -> tuple[GenerationRequest, tuple[EvidenceView, ...]]:
    """Build a decision request and return the evidence views attached to it."""
    page = _latest_page(latest_result)

    request = build_decision_request(
        notebook,
        outline,
        initial,
        budget=budget,
        count_tokens=count_tokens,
        max_candidates=max_candidates,
        max_source_matches=max_source_matches,
        latest_result=latest_result,
        latest_search=latest_search,
        page=page,
        evidence_parts=reader.parts(page) if page is not None else (),
    )
    return request, page.views if page is not None else ()


async def _choose_action(
    request: GenerationRequest,
    vlm: VLMClient,
    *,
    budget: RequestBudget,
    count_tokens: TokenCounter,
    max_attempts: int = 2,
    retry_delay_seconds: float = 1.0,
) -> ToolAction:
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")

    if (
        type(retry_delay_seconds) not in (int, float)
        or not math.isfinite(retry_delay_seconds)
        or retry_delay_seconds < 0
    ):
        raise ValueError("retry_delay_seconds must be finite and nonnegative.")

    feedback = ""
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        current = _action_attempt(request, feedback, budget, count_tokens)

        try:
            result = await vlm.generate(current)
        except ModelClientError as error:
            if error.kind in {
                ErrorKind.AUTHENTICATION,
                ErrorKind.INVALID_REQUEST,
                ErrorKind.UNSUPPORTED_CAPABILITY,
            } or (not error.retryable and error.kind != ErrorKind.INVALID_RESPONSE):
                raise

            feedback = f"Previous model request failed: {error.message}"[:600]
            last_error = error

            if attempt + 1 < max_attempts and error.kind != ErrorKind.INVALID_RESPONSE:
                await asyncio.sleep(retry_delay_seconds)
            continue

        last_error = None

        for name, used, limit in (
            ("input", result.usage.input_tokens, budget.max_input_tokens),
            ("output", result.usage.output_tokens, budget.max_output_tokens),
        ):
            if used is not None and used > limit:
                raise RuntimeError(f"The VLM exceeded the {name}-token allowance.")

        if result.status == "refused":
            raise RuntimeError("The VLM refused action selection.")

        if result.status != "completed":
            feedback = (
                f"Previous generation was {result.status}; "
                f"finish_reason={result.finish_reason}, "
                f"generated_tokens={result.usage.output_tokens}. "
                "Finish thinking and return one short, complete action."
            )
            continue

        try:
            return ActionRequest.model_validate_json(result.text).action
        except ValidationError as error:
            feedback = f"Invalid action arguments: {error}"[:600]
            last_error = error

    raise RuntimeError(
        f"Action selection failed after {max_attempts} attempts: {feedback}"
    ) from last_error


def _action_attempt(
    request: GenerationRequest,
    feedback: str,
    budget: RequestBudget,
    count_tokens: TokenCounter,
) -> GenerationRequest:
    """Fit correction feedback without removing original request content."""
    while True:
        messages = request.messages

        if feedback:
            messages += (
                Message(
                    role="user",
                    parts=(Text(text="Correct the previous action. " + feedback),),
                ),
            )

        current = request.model_copy(update={"messages": messages})
        if input_tokens_fit(budget, count_tokens(current)):
            return current

        if not feedback:
            raise ContextBudgetError(
                "The action request exceeds its input-token budget."
            )

        feedback = feedback[: len(feedback) // 2]


def _finish(
    notebook: NotebookState,
    result: Record,
    run_directory: Path,
    *,
    evidence: PreparedEvidence,
) -> Record:
    review = (
        AnswerReview.model_validate_json(json.dumps(result["review"]))
        if "review" in result
        else None
    )
    supported = (
        result.get("status") == "ok"
        and review is not None
        and review.verdict == "supported"
    )
    status = "completed" if supported else "incomplete"
    citations = []
    if review is not None:
        frames = {frame.frame_id: frame for frame in evidence.frames}
        cues = {cue.cue_id: cue for cue in evidence.cues}
        for source in dict.fromkeys(review.sources):
            if source.source_kind == "frame":
                frame = frames[source.source_id]
                start, end = (
                    frame.presentation_seconds,
                    frame.presentation_seconds + frame.duration_seconds,
                )
            else:
                cue = cues[source.source_id]
                start, end = cue.start_seconds, cue.end_seconds
            citations.append(
                {
                    **source.model_dump(mode="json"),
                    "start_seconds": start,
                    "end_seconds": end,
                }
            )
    gaps = list(review.gaps + review.contradictions) if review else []
    gaps.extend(result.get("proposed_gaps", []))
    gaps.extend(
        entry.text
        for entry in notebook.entries
        if entry.status == "active" and entry.kind in {"question", "contradiction"}
    )
    outcome: Record = {
        "run_id": str(notebook.run_id),
        "preparation_id": str(notebook.preparation_id),
        "collection_id": str(notebook.collection_id),
        "question": notebook.question,
        "status": status,
        "answer": result.get("answer") if supported else None,
        "proposed_answer": None if supported else result.get("answer"),
        "sources": citations,
        "gaps": list(dict.fromkeys(gaps)) if not supported else [],
        "reason": "Answer supported by reviewed evidence."
        if supported
        else result.get("reason", "Evidence review is incomplete."),
        "review": review.model_dump(mode="json") if review else None,
        "review_id": result.get("review_id"),
        "pending_sources": result.get("pending_sources"),
        "step_count": notebook.step_count,
        "model_calls": result.get("model_calls", 0),
        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
    }

    # Publish the result before marking the notebook finished.
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=run_directory,
            prefix=".result-",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(outcome, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, run_directory / "result.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    save_notebook(notebook.model_copy(update={"status": status}), run_directory)
    return outcome


@dataclass
class _RunVLM:
    client: VLMClient
    max_calls: int
    deadline: float
    calls: int = 0

    def check_time(self) -> None:
        if monotonic() >= self.deadline:
            raise InvestigationLimitError("Time limit reached.")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.check_time()
        if self.calls >= self.max_calls:
            raise InvestigationLimitError("Model-call limit reached.")
        remaining = self.deadline - monotonic()
        timeout = min(request.timeout_seconds or remaining, remaining)
        self.calls += 1
        result = await self.client.generate(
            request.model_copy(update={"timeout_seconds": timeout})
        )
        self.check_time()
        return result


def _result_brief(result: Record) -> Record:
    if result.get("status") != "ok":
        return result
    if result.get("tool") == "search":
        return {"tool": "search", "status": "ok"}
    if result.get("tool") == "request_finish":
        return {
            key: result[key]
            for key in ("tool", "status", "review", "proposed_gaps")
            if key in result
        }
    return result


def _progress_keys(
    action: ToolAction | None,
    result: Record,
    shown: tuple[EvidenceView, ...],
    search: SearchResult | None = None,
) -> set[str]:
    """Recognize new content, ignoring score changes and note update timestamps."""
    keys = {"view:" + view.model_dump_json() for view in shown}
    if search is not None:
        for candidate in search.candidates:
            keys.add(f"clip:{candidate.clip_id}")
            for match in candidate.matches:
                keys.add(
                    f"match:{candidate.clip_id}:{match.channel}:{match.source_kind}:{match.source_id}"
                )
    if result.get("status") == "ok":
        page = _latest_page(result)
        if page is not None:
            keys.update("view:" + view.model_dump_json() for view in page.views)
        elif isinstance(action, WriteNotebookAction):
            keys.add(
                "note:"
                + json.dumps(
                    action.model_dump(
                        mode="json", include={"kind", "text", "sources", "status"}
                    ),
                    sort_keys=True,
                )
            )
        elif result.get("tool") == "read_notebook":
            keys.add(
                "notebook:"
                + json.dumps(
                    {key: result.get(key) for key in ("entry_id", "offset", "text")},
                    sort_keys=True,
                )
            )
    return keys


def _latest_page(result: Record | None) -> EvidencePage | None:
    if result is None or result.get("status") != "ok" or "page" not in result:
        return None

    return EvidencePage.model_validate_json(json.dumps(result["page"], allow_nan=False))


def _page_fits(
    page: EvidencePage,
    *,
    notebook: NotebookState,
    outline: VideoOutline,
    initial: InitialSearchResult,
    reader: EvidenceReader,
    budget: RequestBudget,
    count_tokens: TokenCounter,
    latest_search: SearchResult | None = None,
) -> bool:
    # An evidence-reading action advances the step, without changing notes.
    upcoming = notebook.model_copy(update={"step_count": notebook.step_count + 1})

    try:
        build_decision_request(
            upcoming,
            outline,
            initial,
            budget=budget,
            count_tokens=count_tokens,
            latest_result=page_result(page),
            latest_search=latest_search,
            page=page,
            evidence_parts=reader.parts(page),
        )
    except ContextBudgetError:
        return False

    return True
