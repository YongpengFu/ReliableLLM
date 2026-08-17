"""Reliable baseline: traced, structured, grounded-by-construction.

Every call is logged to LangSmith (via @traceable), the model is forced into
a schema that requires a supporting quote, and the model is explicitly told
to say so instead of guessing when the document doesn't contain the answer.

The agent doesn't get the source document handed to it up front. Instead it
runs a multi-turn tool-calling loop: it must call `search_document` to
retrieve grounding excerpts (possibly more than once, for multi-part
questions) and then call `submit_answer` to finish. That means each question
produces several LLM calls, all nested under one trace — a real agent loop,
not a single chat completion.
"""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from reliablellm.document import search_document_sections

MODEL = "gpt-4o-mini"
MAX_TURNS = 8


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


@tool
def search_document(query: str) -> str:
    """Search the earnings document for paragraphs relevant to a query and
    return the matching excerpts verbatim. Call this to gather evidence
    before answering; call it again with a different query if the question
    has multiple parts."""
    return search_document_sections(query)


@tool
def submit_answer(answerable: bool, answer: str, supporting_quote: str | None = None) -> str:
    """Submit your final structured answer. Call this exactly once, after
    you've gathered enough evidence with search_document. Set answerable to
    false and omit supporting_quote if the document doesn't contain the
    answer."""
    return "Answer submitted."


SYSTEM_PROMPT = """\
You are an analyst assistant answering questions about a company earnings
document. You are not given the document text directly — use the
search_document tool to retrieve the paragraphs relevant to the question.
You may call it more than once if the question has multiple parts.

Answer strictly using the retrieved excerpts, never outside knowledge. If two
searches with differently worded queries don't turn up the answer, stop
searching — the document doesn't contain it. Set answerable to false rather
than guessing or searching indefinitely. If a question has multiple parts,
search for each part separately; if any part isn't covered by the document,
set answerable to false for the whole question rather than answering just
the part you found. Every answerable question must be backed by a direct
supporting_quote taken from a retrieved excerpt.

When you're ready, call submit_answer exactly once with your final
structured answer. Do not answer in plain text."""

_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools([search_document, submit_answer])


@traceable(name="after_agent_reliable", run_type="chain")
def answer_question(question: str) -> AnalystAnswer:
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]

    for _ in range(MAX_TURNS):
        ai_message: AIMessage = _llm.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": "Call submit_answer with your final structured answer.",
                }
            )
            continue

        final_answer: AnalystAnswer | None = None
        for call in ai_message.tool_calls:
            if call["name"] == "submit_answer":
                final_answer = AnalystAnswer(**call["args"])
                tool_output = "Answer submitted."
            else:
                tool_output = search_document.invoke(call["args"])
            messages.append(
                {"role": "tool", "content": tool_output, "tool_call_id": call["id"]}
            )

        if final_answer is not None:
            return final_answer

    return AnalystAnswer(
        answerable=False,
        answer=f"Agent did not reach a grounded answer within {MAX_TURNS} turns.",
    )
