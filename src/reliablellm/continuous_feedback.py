"""Wires the three pieces of the continuous-feedback loop into the one call
site the runners use:

    1. online_monitor.py   — score every live call with reference-free evaluators.
    2. drift_monitor.py    — track each metric's score distribution over time
                              (EWMA + CUSUM), not any single call.
    3. eval_promotion.py   — when a metric drifts, promote the production
                              calls that triggered it into the eval dataset.

`observe_live_call()` is what run_after.py / run_otel.py call per question,
in place of calling online_monitor.score_live_call() directly.

State (the rolling call buffer, the per-metric trackers) lives at module
level, scoped to this process — a real deployment would back this with
whatever store the production service already uses for request state
(Redis, a queue consumer's in-memory window, ...); the shape of the loop is
the same either way.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langsmith import Client

from reliablellm.drift_monitor import DriftSignal, DriftTracker
from reliablellm.eval_promotion import promote_flagged_calls
from reliablellm.online_monitor import score_live_call

WINDOW_SIZE = 12  # how many recent calls stay eligible to be promoted when a metric drifts


@dataclass
class Observation:
    question: str
    result: Any
    feedback: dict[str, dict]
    langsmith_run_id: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promoted: bool = False


_recent: deque[Observation] = deque(maxlen=WINDOW_SIZE)
_trackers: dict[str, DriftTracker] = {}
_langsmith_client: Client | None = None


def _client() -> Client:
    global _langsmith_client
    if _langsmith_client is None:
        _langsmith_client = Client()
    return _langsmith_client


def observe_live_call(
    question: str,
    result: Any,
    *,
    langsmith_run_id: str | None = None,
    langsmith_project: str | None = None,
) -> list[DriftSignal]:
    """Score one live call, feed the scores into the drift trackers, and
    auto-promote flagged calls into the eval dataset if any metric just
    drifted. Returns one DriftSignal per metric so callers can log/inspect
    it further if they want."""
    feedback = score_live_call(
        question,
        result,
        langsmith_run_id=langsmith_run_id,
        langsmith_project=langsmith_project,
    )
    obs = Observation(
        question=question,
        result=result,
        feedback=feedback,
        langsmith_run_id=langsmith_run_id,
    )
    _recent.append(obs)

    signals = []
    for metric, fb in feedback.items():
        tracker = _trackers.setdefault(metric, DriftTracker(metric))
        signal = tracker.update(float(bool(fb["score"])))
        signals.append(signal)
        _report(signal)
        if signal.alert:
            _handle_drift(metric)
    return signals


def _report(signal: DriftSignal) -> None:
    if signal.reason in ("burn-in", "baseline-established"):
        print(f"[drift-monitor] {signal.metric}: {signal.reason} ({signal.n_samples} samples)")
        return
    if signal.new_alert:
        marker = "  <-- DRIFT ALERT (new)"
    elif signal.alert:
        marker = "  <-- still drifted"
    else:
        marker = ""
    print(
        f"[drift-monitor] {signal.metric}: value={signal.value:.0f} "
        f"ewma={signal.ewma:.2f} cusum={signal.cusum:.2f} "
        f"baseline={signal.baseline_mean:.2f}{marker}"
        + (f" ({signal.reason})" if signal.alert else "")
    )


def _handle_drift(metric: str) -> None:
    flagged = [obs for obs in _recent if not obs.promoted and not obs.feedback[metric]["score"]]
    if not flagged:
        print(f"[drift-monitor] {metric} drifted, but no unpromoted flagged calls in the window")
        return

    print(
        f"[drift-monitor] promoting {len(flagged)} flagged call(s) for "
        f"'{metric}' into the eval dataset..."
    )
    example_ids = promote_flagged_calls(_client(), flagged, metric=metric)
    print(f"[drift-monitor] promoted {len(example_ids)} example(s): {example_ids}")
