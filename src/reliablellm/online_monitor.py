"""Online, reference-free scoring: run the same evaluators used offline in
run_eval.py against every live call, and attach the score to that call's
trace as feedback right after it happens — so drift shows up on the next
call, not next quarter's benchmark run.

Only groundedness and format_compliance run here. Both evaluators in
after/evaluators.py are reference-free — they judge the answer against the
source document and its own shape, not against a reference_answer — which is
exactly what production traffic has available. task_success needs a
reference_answer, so it stays offline-only, scored in run_eval.py against
the curated dataset.

Works with whichever tracing backend the caller is using:
- LangSmith (run_after.py): pass the run_id of the live trace and the score
  is posted as Feedback on that run via the LangSmith API.
- OpenTelemetry (run_otel.py): call this while the request's span is still
  current (i.e. inside its `with` block) and the score is recorded as span
  attributes — no run_id needed.
Either, both, or neither can be active; each path is a no-op if its
prerequisite (a run_id, a recording span) isn't there.

Note this module talks to both backends directly, unlike the after/otel
agents themselves — that's expected. It *is* the observability layer, not
the thing being observed.
"""

from __future__ import annotations

from typing import Any

from langsmith import Client
from opentelemetry import trace as otel_trace

from reliablellm.after.evaluators import format_compliance, groundedness

ONLINE_EVALUATORS = (groundedness, format_compliance)

_langsmith_client: Client | None = None
_session_id_cache: dict[str, str] = {}


def _client() -> Client:
    global _langsmith_client
    if _langsmith_client is None:
        _langsmith_client = Client()
    return _langsmith_client


def _session_id_for(project_name: str) -> str:
    """Resolve a LangSmith project name to the session_id create_feedback
    wants. Without it, feedback ingestion can't be backgrounded and the SDK
    warns that the run_id-only path is deprecated."""
    if project_name not in _session_id_cache:
        _session_id_cache[project_name] = str(
            _client().read_project(project_name=project_name).id
        )
    return _session_id_cache[project_name]


def score_live_call(
    question: str,
    result: Any,
    *,
    langsmith_run_id: str | None = None,
    langsmith_project: str | None = None,
) -> dict[str, dict]:
    """Score one production answer with the reference-free evaluators and
    attach the result to whichever trace is active. Returns the raw
    evaluator outputs keyed by evaluator name, for callers that want to do
    their own aggregation (e.g. a rolling drift check across many calls)."""
    inputs = {"question": question}
    outputs = result.model_dump() if hasattr(result, "model_dump") else {"answer": result}

    session_id = _session_id_for(langsmith_project) if langsmith_project else None

    scores: dict[str, dict] = {}
    for evaluator in ONLINE_EVALUATORS:
        feedback = evaluator(inputs, outputs, {})
        scores[feedback["key"]] = feedback
        _emit(feedback, langsmith_run_id=langsmith_run_id, langsmith_session_id=session_id)
    return scores


def _emit(feedback: dict, *, langsmith_run_id: str | None, langsmith_session_id: str | None) -> None:
    key, score, comment = feedback["key"], feedback["score"], feedback.get("comment")
    flag = "" if score else "  <-- FLAGGED"
    comment_suffix = f"  ({comment})" if comment else ""
    print(f"[online-monitor] {key}={score}{flag}{comment_suffix}")

    if langsmith_run_id is not None:
        _client().create_feedback(
            run_id=langsmith_run_id,
            key=key,
            score=score,
            comment=comment,
            session_id=langsmith_session_id,
        )

    span = otel_trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute(f"online_monitor.{key}.score", bool(score))
        if comment:
            span.set_attribute(f"online_monitor.{key}.comment", comment)
