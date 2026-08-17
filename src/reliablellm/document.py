"""Shared source document and evaluation dataset for the before/after demo."""

SOURCE_DOCUMENT = """\
Northwind Analytics — Q2 2026 Earnings Summary

Revenue for the second quarter of 2026 was $84.2 million, up 18% year-over-year
from $71.4 million in Q2 2025. Subscription revenue accounted for $61.5 million
of the total, while professional services contributed $22.7 million.

Gross margin improved to 71%, compared to 68% in the prior-year quarter, driven
by continued migration of customers to the managed cloud tier.

Headcount at quarter end was 412 employees, an increase of 34 from the end of
Q1 2026. The increase was concentrated in the engineering and customer success
organizations.

The company added 47 new enterprise customers during the quarter, bringing the
total enterprise customer count to 318. Net revenue retention was 112%.

Operating expenses were $54.1 million, resulting in operating income of $4.7
million and an operating margin of 5.6%.

Management called out two risks for the second half of 2026: continued
foreign-exchange headwinds in the EMEA region, and elongated sales cycles for
deals above $250,000 in annual contract value.

The company did not repurchase any shares during the quarter and did not
declare a dividend.
"""

DOCUMENT_SECTIONS: list[str] = [
    section.strip() for section in SOURCE_DOCUMENT.strip().split("\n\n") if section.strip()
]


_SECTION_WORD_SETS = [
    {w.strip(".,?!'\"()") for w in section.lower().split()} for section in DOCUMENT_SECTIONS
]
_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "was", "were", "is", "are", "did",
    "how", "what", "in", "of", "to", "on", "at", "by", "with", "during",
    "much", "company's", "company", "did", "about",
}


def search_document_sections(query: str) -> str:
    """Return the document paragraphs most relevant to a query, scored by
    overlap on discriminative words only (query terms that appear in most
    sections, like "2026" or "quarter", are dropped since they can't tell
    sections apart). Shared by the after/ and otel/ agents' search_document
    tool, so both agents have to retrieve grounding evidence via tool calls
    instead of getting the whole document handed to them upfront."""
    query_words = {
        w for w in (t.strip(".,?!'\"()") for t in query.lower().split())
        if len(w) > 1 and w not in _STOPWORDS
    }
    if not query_words:
        return "No matching sections found in the document."

    n_sections = len(DOCUMENT_SECTIONS)
    max_matches = max(1, n_sections // 4)
    discriminative = {
        w
        for w in query_words
        if sum(1 for words in _SECTION_WORD_SETS if w in words) <= max_matches
    }
    search_words = discriminative or query_words

    scored = [
        (len(search_words & words), section)
        for section, words in zip(DOCUMENT_SECTIONS, _SECTION_WORD_SETS)
    ]
    scored = [pair for pair in scored if pair[0] > 0]
    if not scored:
        return "No matching sections found in the document."
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return "\n\n".join(section for _, section in scored[:3])


EVAL_CASES = [
    {
        "question": "What was total revenue in Q2 2026, and how much did it grow year-over-year?",
        "reference_answer": "Revenue was $84.2 million, up 18% year-over-year from $71.4 million.",
        "answerable": True,
    },
    {
        "question": "How much of Q2 2026 revenue came from subscriptions versus professional services?",
        "reference_answer": "$61.5 million from subscriptions and $22.7 million from professional services.",
        "answerable": True,
    },
    {
        "question": "What was the gross margin in Q2 2026, and how does it compare to the prior year?",
        "reference_answer": "Gross margin was 71%, up from 68% in Q2 2025.",
        "answerable": True,
    },
    {
        "question": "How many employees did the company have at the end of Q2 2026?",
        "reference_answer": "412 employees, up 34 from the end of Q1 2026.",
        "answerable": True,
    },
    {
        "question": "What was net revenue retention in Q2 2026?",
        "reference_answer": "Net revenue retention was 112%.",
        "answerable": True,
    },
    {
        "question": "What risks did management flag for the second half of 2026?",
        "reference_answer": "FX headwinds in EMEA and elongated sales cycles for deals above $250,000 ACV.",
        "answerable": True,
    },
    {
        "question": "What was the company's revenue guidance for Q3 2026?",
        "reference_answer": "The document does not provide Q3 2026 guidance.",
        "answerable": False,
    },
    {
        "question": "How many shares did the company repurchase, and what was the CEO's total compensation?",
        "reference_answer": "The document states no shares were repurchased; it does not mention CEO compensation at all.",
        "answerable": False,
    },
]
