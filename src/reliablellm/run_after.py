"""Run the reliable agent over the eval questions and print the structured,
grounded answers. Traces to the `reliablellm-after` LangSmith project so it
can be visually compared against `reliablellm-before`.
"""

from dotenv import load_dotenv

load_dotenv()

from langsmith.run_helpers import tracing_context

from reliablellm.after.agent import answer_question
from reliablellm.document import EVAL_CASES


def main() -> None:
    with tracing_context(project_name="reliablellm-after"):
        for case in EVAL_CASES:
            print(f"Q: {case['question']}")
            result = answer_question(case["question"])
            print(f"answerable: {result.answerable}")
            print(f"answer: {result.answer}")
            print(f"supporting_quote: {result.supporting_quote}")
            print(f"(answerable per source doc: {case['answerable']})")
            print("-" * 60)


if __name__ == "__main__":
    main()
