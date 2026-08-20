"""Step 3 of the continuous-feedback-loop design: close the loop back into
the eval harness. When drift_monitor.py flags a sustained shift in a
metric, promote the production calls behind it into the same LangSmith
dataset run_eval.py scores against (`reliablellm-analyst-qa`) as new
Examples — so the benchmark grows from what production is actually seeing,
instead of staying frozen at whatever EVAL_CASES looked like when someone
last hand-wrote it.

Promoted examples carry no reference_answer: production has no ground
truth, and task_success (the only evaluator in after/evaluators.py that
needs one) would just be scoring against a fabricated target. They're
tagged `needs_review: true` in metadata, along with which metric flagged
them and the run they came from, so they're immediately available for
groundedness/format_compliance scoring in the next run_eval.py pass and
ready for a human to backfill a reference_answer and clear the flag
whenever someone gets to it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langsmith import Client

from reliablellm.run_eval import DATASET_NAME, ensure_dataset

if TYPE_CHECKING:
    from reliablellm.continuous_feedback import Observation


def promote_flagged_calls(client: Client, flagged: list["Observation"], metric: str) -> list[str]:
    """Add each flagged production call to the eval dataset as a new,
    needs-review Example. Returns the created example ids. No-ops (and
    creates nothing) if `flagged` is empty."""
    if not flagged:
        return []

    ensure_dataset(client)
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    flagged_at = datetime.now(timezone.utc).isoformat()

    examples = [
        {
            "inputs": {"question": obs.question},
            "outputs": {
                "reference_answer": None,
                "answerable": getattr(obs.result, "answerable", None),
            },
            "metadata": {
                "source": "production-drift",
                "flagged_metric": metric,
                "flagged_at": flagged_at,
                "needs_review": True,
                "original_run_id": str(obs.langsmith_run_id) if obs.langsmith_run_id else None,
                "flagged_feedback": {
                    key: feedback.get("comment") for key, feedback in obs.feedback.items()
                },
            },
        }
        for obs in flagged
    ]

    response = client.create_examples(dataset_id=dataset.id, examples=examples)
    for obs in flagged:
        obs.promoted = True
    return list(response.get("example_ids", []))
