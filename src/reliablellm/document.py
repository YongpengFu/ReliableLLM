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
