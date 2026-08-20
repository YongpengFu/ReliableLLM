"""Step 2 + 3 demo: prove the continuous-feedback loop actually catches a
sustained shift and closes the loop back into the eval dataset — rather
than waiting for the model to drift on its own sometime between here and
next quarter's benchmark run, this script simulates it happening now.

Phase 1 ("healthy"): answers real questions through the real, reliable
after/agent.py. This is what establishes drift_monitor.py's baseline —
expect close to perfect groundedness and format_compliance.

Phase 2 ("drifted"): skips the real agent and returns one fixed, fluent,
entirely fabricated AnalystAnswer for every question — standing in for a
real production failure mode (a bad deploy, a prompt regression, a swapped
model or endpoint that stops actually reading the document) where the agent
keeps answering confidently and incorrectly instead of erroring out
somewhere loud. online_monitor.py's judge still runs for real against these
answers, so what catches the drift below is the actual EWMA/CUSUM math in
drift_monitor.py, not a canned result.

Watch the output for:
- Phase 1: "baseline-established", then quiet ewma/cusum lines near 1.0.
- Phase 2: ewma dropping, cusum sinking, then "<-- DRIFT ALERT" and
  "promoting N flagged call(s) ... into the eval dataset" — that's
  continuous_feedback.py handing the flagged production calls to
  eval_promotion.py, which writes them into `reliablellm-analyst-qa` as
  new, needs_review Examples. format_compliance should stay healthy
  throughout phase 2 (the fabricated answer is still a well-formed object)
  — only groundedness should drift, which is the point: each metric is
  tracked, and alerts, independently.
"""

from dotenv import load_dotenv

load_dotenv()

from langsmith.run_helpers import trace, tracing_context

from reliablellm.after.agent import AnalystAnswer, answer_question
from reliablellm.continuous_feedback import observe_live_call
from reliablellm.document import EVAL_CASES

PROJECT = "reliablellm-drift-demo"

# A fixed, fluent, entirely fabricated answer — stands in for a model that
# has quietly stopped grounding itself in the document but still returns a
# well-formed, confident-looking structured object. format_compliance stays
# green; groundedness shouldn't.
DRIFTED_ANSWER = AnalystAnswer(
    answerable=True,
    answer=(
        "Revenue grew 45% year-over-year to $120 million, driven by the new "
        "AI product line launched in Q2."
    ),
    supporting_quote=(
        "Revenue for the second quarter of 2026 was $120 million, up 45% "
        "year-over-year, driven by the new AI product line."
    ),
)


def _run_phase(name: str, questions: list[str], *, drifted: bool) -> None:
    print(f"\n=== phase: {name} ===")
    for question in questions:
        with trace(
            name="production_request",
            run_type="chain",
            inputs={"question": question},
            extra={"metadata": {"phase": name}},
        ) as run:
            result = DRIFTED_ANSWER if drifted else answer_question(question)
            run.end(outputs=result.model_dump())
            observe_live_call(
                question,
                result,
                langsmith_run_id=run.id,
                langsmith_project=PROJECT,
            )


def main() -> None:
    questions = [case["question"] for case in EVAL_CASES if case["answerable"]]
    with tracing_context(project_name=PROJECT):
        # Real agent, real questions, repeated once to clear burn-in with a
        # real rolling window before phase 2 hits.
        _run_phase("healthy", questions, drifted=False)
        # Same questions, fabricated answers — the simulated regression.
        _run_phase("drifted", questions * 2, drifted=True)


if __name__ == "__main__":
    main()
