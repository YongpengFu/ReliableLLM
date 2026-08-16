"""LangSmith evaluators mirroring the deck's "What to Capture" slide:
groundedness, task success, and format compliance.

Each evaluator has the signature `(inputs, outputs, reference_outputs) -> dict`
supported natively by `langsmith.evaluate()`.
"""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from reliablellm.document import SOURCE_DOCUMENT

JUDGE_MODEL = "gpt-4o-mini"


class JudgeVerdict(BaseModel):
    score: bool = Field(description="True if the criterion is met, False otherwise.")
    reasoning: str = Field(description="One-sentence justification for the score.")


_judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(JudgeVerdict)


def _run_judge(system_prompt: str, user_prompt: str) -> JudgeVerdict:
    return _judge.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Is the answer actually supported by the source document (or does it
    correctly abstain when the document doesn't contain the answer)?"""
    answer = outputs.get("answer", "")
    verdict = _run_judge(
        system_prompt=(
            "You judge whether an AI analyst's answer is grounded in a source "
            "document: every factual claim must be directly supported by the "
            "document, OR the answer must correctly state the document doesn't "
            "contain the information. An answer that states facts not present "
            "in the document (a hallucination) is NOT grounded."
        ),
        user_prompt=(
            f"Document:\n\n{SOURCE_DOCUMENT}\n\n"
            f"Question: {inputs.get('question')}\n\n"
            f"AI answer: {answer}\n\n"
            "Is this answer grounded in the document?"
        ),
    )
    return {"key": "groundedness", "score": verdict.score, "comment": verdict.reasoning}


def task_success(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Does the answer accomplish what the analyst actually asked, matching
    the substance of the reference answer?"""
    answer = outputs.get("answer", "")
    reference = reference_outputs.get("reference_answer", "")
    verdict = _run_judge(
        system_prompt=(
            "You judge whether an AI answer successfully accomplishes the "
            "task, compared to a reference answer. The wording can differ, "
            "but the substance (facts, numbers, conclusions) must match."
        ),
        user_prompt=(
            f"Question: {inputs.get('question')}\n\n"
            f"Reference answer: {reference}\n\n"
            f"AI answer: {answer}\n\n"
            "Does the AI answer successfully match the reference answer?"
        ),
    )
    return {"key": "task_success", "score": verdict.score, "comment": verdict.reasoning}


def format_compliance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the run produce a parseable structured object with an explicit
    answerable flag and non-empty answer, suitable for downstream systems?"""
    is_compliant = (
        isinstance(outputs.get("answerable"), bool)
        and isinstance(outputs.get("answer"), str)
        and bool(outputs.get("answer"))
    )
    return {"key": "format_compliance", "score": is_compliant}
