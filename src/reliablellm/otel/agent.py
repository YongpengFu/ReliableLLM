"""Reliable baseline, vendor-neutral variant: same structured, grounded-by-
construction agent as reliablellm.after.agent, but with zero tracing code.

Observability comes entirely from OpenTelemetry zero-code auto-instrumentation
(https://opentelemetry.io/docs/zero-code/python/): run this module via the
`opentelemetry-instrument` launcher instead of `python`, and the openai client
calls LangChain makes under the hood are captured automatically by the
`opentelemetry-instrumentation-openai-v2` instrumentor — no @traceable, no
vendor SDK import, nothing to change here if you swap where the spans go.
See run_otel.py for how it's launched.
"""

from langchain_openai import ChatOpenAI
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

# method="function_calling" is required here: the default "json_schema" method
# routes through the openai SDK's `.chat.completions.with_raw_response.parse()`,
# which opentelemetry-instrumentation-openai-v2 doesn't patch (it only wraps
# `Completions.create`/`AsyncCompletions.create`). function_calling uses
# `.create()` with a tool call, so it's the path auto-instrumentation actually
# sees.
_llm = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(
    AnalystAnswer, method="function_calling"
)


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
