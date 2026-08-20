"""Run the vendor-neutral agent over the eval questions and print the
structured, grounded answers.

Unlike run_after.py, this module contains no tracing code at all — it must be
launched with the `opentelemetry-instrument` wrapper for spans to be captured
and exported. Plain `python -m reliablellm.run_otel` will still produce
correct answers, just with no tracing.

    uv run opentelemetry-instrument \\
        --service_name reliablellm-otel \\
        --traces_exporter console \\
        python -m reliablellm.run_otel

See README.md for how to point the exporter at a real OTel-compatible
backend (Langfuse, Jaeger, Honeycomb, ...) instead of the console.

Each question is wrapped in its own `production_request` span and scored
online by online_monitor.py the moment it finishes — see that module for why
only groundedness and format_compliance run here. This span creation lives
in the runner, not in otel/agent.py, so the agent itself stays exactly as
"zero tracing code" as its docstring claims; the online-monitoring wrapper is
observability infrastructure, not the thing being observed. continuous_feedback.py
takes it further: every score feeds a per-metric EWMA/CUSUM drift tracker,
auto-promoting flagged calls into the eval dataset when a metric's rolling
distribution actually drifts — 8 questions isn't enough history to trip it
here; see run_drift_demo.py for that in action.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# .env also carries LANGSMITH_TRACING=true for the after/ module. LangChain
# checks that env var itself on every call (independent of anything this repo
# does) and will silently start sending traces to LangSmith otherwise — which
# would defeat the point of the otel/ module. Force it off here so this run
# only produces the OpenTelemetry spans it's meant to.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from opentelemetry import trace as otel_trace

from reliablellm.continuous_feedback import observe_live_call
from reliablellm.document import EVAL_CASES
from reliablellm.otel.agent import answer_question

_tracer = otel_trace.get_tracer("reliablellm.run_otel")


def main() -> None:
    for case in EVAL_CASES:
        print(f"Q: {case['question']}")
        with _tracer.start_as_current_span("production_request") as span:
            span.set_attribute("question", case["question"])
            result = answer_question(case["question"])
            observe_live_call(case["question"], result)
        print(f"answerable: {result.answerable}")
        print(f"answer: {result.answer}")
        print(f"supporting_quote: {result.supporting_quote}")
        print(f"(answerable per source doc: {case['answerable']})")
        print("-" * 60)


if __name__ == "__main__":
    main()
