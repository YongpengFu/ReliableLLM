"""Before-vs-after comparison: runs both agents through LangSmith's
evaluate() against the same dataset and evaluators, then prints a scored
comparison table.
"""

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate

from reliablellm.after.agent import answer_question as after_answer_question
from reliablellm.after.evaluators import format_compliance, groundedness, task_success
from reliablellm.before.agent import answer_question as before_answer_question
from reliablellm.document import EVAL_CASES

DATASET_NAME = "reliablellm-analyst-qa"
EVALUATORS = [groundedness, task_success, format_compliance]


def ensure_dataset(client: Client) -> str:
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Grounded analyst Q&A over the Northwind Q2 2026 earnings summary.",
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": c["question"]},
                    "outputs": {
                        "reference_answer": c["reference_answer"],
                        "answerable": c["answerable"],
                    },
                }
                for c in EVAL_CASES
            ],
        )
    return DATASET_NAME


def target_before(inputs: dict) -> dict:
    answer = before_answer_question(inputs["question"])
    return {"answer": answer, "answerable": None, "supporting_quote": None}


def target_after(inputs: dict) -> dict:
    result = after_answer_question(inputs["question"])
    return result.model_dump()


def mean_scores(results) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in results:
        for evaluation in row["evaluation_results"]["results"]:
            totals.setdefault(evaluation.key, []).append(float(evaluation.score))
    return {key: sum(scores) / len(scores) for key, scores in totals.items()}


def main() -> None:
    client = Client()
    ensure_dataset(client)

    print("Running 'before' experiment...")
    before_results = evaluate(
        target_before,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix="before",
        client=client,
    )
    before_scores = mean_scores(before_results)

    print("Running 'after' experiment...")
    after_results = evaluate(
        target_after,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix="after",
        client=client,
    )
    after_scores = mean_scores(after_results)

    print()
    print(f"{'metric':<20}{'before':>10}{'after':>10}")
    for key in ("groundedness", "task_success", "format_compliance"):
        print(f"{key:<20}{before_scores.get(key, 0):>10.2f}{after_scores.get(key, 0):>10.2f}")

    print()
    print(f"before experiment: {before_results.experiment_name}  ({before_results.url})")
    print(f"after experiment:  {after_results.experiment_name}  ({after_results.url})")


if __name__ == "__main__":
    main()
