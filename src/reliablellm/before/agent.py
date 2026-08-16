"""Naive baseline: a raw LLM call with no tracing, no schema, no eval.

This is the "ship it and hope" version — a plain prompt over the source
document, no instruction to abstain when the answer isn't there, and no way
to see what the model actually did after the fact.
"""

from openai import OpenAI

from reliablellm.document import SOURCE_DOCUMENT

MODEL = "gpt-4o-mini"

_client = OpenAI()

PROMPT_TEMPLATE = """\
Here is a company earnings document:

{document}

Answer the following question about the document as helpfully as you can.

Question: {question}
"""


def answer_question(question: str) -> str:
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    document=SOURCE_DOCUMENT, question=question
                ),
            }
        ],
    )
    return response.choices[0].message.content or ""
