"""Run the reliable agent over the eval questions and print the structured,
grounded answers. Traces to the `reliablellm-after` LangSmith project so it
can be visually compared against `reliablellm-before`.

Each question is wrapped in its own `production_request` trace, and scored
online by online_monitor.py the moment it finishes — see that module for
why only groundedness and format_compliance run here (task_success needs a
reference_answer this script doesn't have). The score is attached to the
trace as LangSmith Feedback, so you can filter runs by it in the UI without
waiting for a batch eval. continuous_feedback.py takes it a step further:
it feeds every score into a per-metric EWMA/CUSUM drift tracker, and
auto-promotes flagged calls into the eval dataset if a metric's rolling
distribution actually drifts (see continuous_feedback.py and
drift_monitor.py) — 8 questions isn't enough history to trip it here; see
run_drift_demo.py for that in action.
"""

from dotenv import load_dotenv

load_dotenv()

from langsmith.run_helpers import trace, tracing_context

from reliablellm.after.agent import answer_question
from reliablellm.continuous_feedback import observe_live_call
from reliablellm.document import EVAL_CASES


def main() -> None:
    with tracing_context(project_name="reliablellm-after"):
        for case in EVAL_CASES:
            print(f"Q: {case['question']}")
            with trace(
                name="production_request",
                run_type="chain",
                inputs={"question": case["question"]},
            ) as run:
                result = answer_question(case["question"])
                run.end(outputs=result.model_dump())
                observe_live_call(
                    case["question"],
                    result,
                    langsmith_run_id=run.id,
                    langsmith_project="reliablellm-after",
                )
            print(f"answerable: {result.answerable}")
            print(f"answer: {result.answer}")
            print(f"supporting_quote: {result.supporting_quote}")
            print(f"(answerable per source doc: {case['answerable']})")
            print("-" * 60)


if __name__ == "__main__":
    main()
