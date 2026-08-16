"""Reliable baseline: traced, structured, grounded-by-construction.

Every call is logged to LangSmith (via @traceable), the model is forced into
a schema that requires a supporting quote, and the model is explicitly told
to say so instead of guessing when the document doesn't contain the answer.
"""

from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from reliablellm.document import SOURCE_DOCUMENT

MODEL = "gpt-4o-mini"


class AnalystAnswer(BaseModel):
    answerable: bool = Field(
        description="Whether the document actually contains the answer to the question."
    )
    answer: str = Field(
        description="The answer, or a brief explanation of what's missing if not answerable."
    )
    supporting_quote: str | None = Field(
        default=None,
        description="A direct quote from the document that supports the answer. "
        "Must be None when answerable is False.",
    )


SYSTEM_PROMPT = """\
You are an analyst assistant. You answer questions strictly using the provided
document. If the document does not contain the answer, set answerable to
false and do not guess or use outside knowledge. Every answer you give for an
answerable question must be backed by a direct supporting_quote from the
document."""

_llm = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(AnalystAnswer)


@traceable(name="after_agent_reliable", run_type="chain")
def answer_question(question: str) -> AnalystAnswer:
    result = _llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document:\n\n{SOURCE_DOCUMENT}\n\nQuestion: {question}",
            },
        ]
    )
    return result
