"""Run the naive baseline over the eval questions and print the raw answers.

No tracing, no schema, no eval — just stdout. Watch the out-of-scope
questions to see it hallucinate.
"""

from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable

from reliablellm.before.agent import answer_question
from reliablellm.document import EVAL_CASES

# The agent itself has no tracing at all; this wraps just the call site so
# the run still shows up in LangSmith for a visual before/after comparison
# — as a single flat, opaque span with no structure inside it.
_traced_answer_question = traceable(
    name="before_agent_naive", project_name="reliablellm-before"
)(answer_question)


def main() -> None:
    for case in EVAL_CASES:
        print(f"Q: {case['question']}")
        answer = _traced_answer_question(case["question"])
        print(f"A: {answer}")
        print(f"(answerable per source doc: {case['answerable']})")
        print("-" * 60)


if __name__ == "__main__":
    main()
