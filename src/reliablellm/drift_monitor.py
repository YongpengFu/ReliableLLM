"""Step 2 of the continuous-feedback-loop design: track the *distribution*
of an online score over time instead of alerting on any single call.

LLM-judge scores (like groundedness from online_monitor.py) are noisy per
call — one False doesn't mean the model drifted, it might just be a hard
question. A sustained shift in the rolling mean does. This module runs two
classic statistical-process-control detectors over each metric's score
stream and alerts only when one of them trips:

- EWMA (exponentially weighted moving average): smooths the noisy 0/1
  stream and alerts once the smoothed value falls a fixed margin below the
  healthy baseline. Tuned for a gradual decline.
- CUSUM (cumulative sum): accumulates each sample's deviation below
  baseline (net of a small allowed slack) and alerts once the accumulated
  total passes a decision threshold. Tuned for a smaller, steadier shift
  than EWMA reacts to, and — because a passing sample pulls the sum back
  toward zero — naturally shrugs off one-off bad samples mixed into an
  otherwise healthy stream.

Deviations are measured directly in probability space (the scores are 0/1,
or any float in [0, 1]), not normalized by an estimated variance: real
production groundedness is often a genuine 1.0 during burn-in (the agent
usually is right), which would make a variance-normalized control band
collapse to zero width and alert on the very next blip. Fixed slack/margin
constants avoid that degenerate case and are easier to reason about anyway
("tolerate 25% average badness per sample before it accumulates") than a
sigma-scaled band.

Either detector tripping counts as drift. CUSUM resets after firing so its
own accumulator doesn't just stay pinned; EWMA has no such reset and stays
below the control limit for as long as the shift persists, so `alert` stays
True call after call through a sustained incident — that's `new_alert` that
debounces it to a single transition (call it once, keep collecting evidence
quietly, don't page on every subsequent call): True only the first time
`alert` flips from False to True, and again if it recovers and drifts a
second time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BURN_IN = 4  # samples used to establish the healthy baseline before drift can fire
EWMA_LAMBDA = 0.3  # weight on the newest observation
EWMA_MARGIN = 0.35  # how far the smoothed score may fall below baseline before EWMA alerts
CUSUM_SLACK_K = 0.25  # per-sample tolerance before a deviation starts accumulating
CUSUM_THRESHOLD_H = 1.5  # accumulated deviation before CUSUM alerts


@dataclass
class DriftSignal:
    metric: str
    n_samples: int
    value: float
    baseline_mean: float | None
    ewma: float | None
    cusum: float
    alert: bool
    new_alert: bool = False
    reason: str | None = None


@dataclass
class DriftTracker:
    """Per-metric EWMA + CUSUM state over a stream of 0/1 (or any 0..1)
    scores. Call update() once per new score, in call order."""

    metric: str
    _burn_in_scores: list[float] = field(default_factory=list)
    _baseline_mean: float | None = None
    _ewma: float | None = None
    _cusum: float = 0.0
    _n: int = 0
    _alerting: bool = False

    def update(self, score: float) -> DriftSignal:
        self._n += 1

        if self._baseline_mean is None:
            self._burn_in_scores.append(score)
            if len(self._burn_in_scores) < BURN_IN:
                return DriftSignal(
                    metric=self.metric,
                    n_samples=self._n,
                    value=score,
                    baseline_mean=None,
                    ewma=None,
                    cusum=0.0,
                    alert=False,
                    reason="burn-in",
                )
            self._baseline_mean = sum(self._burn_in_scores) / len(self._burn_in_scores)
            self._ewma = self._baseline_mean
            return DriftSignal(
                metric=self.metric,
                n_samples=self._n,
                value=score,
                baseline_mean=self._baseline_mean,
                ewma=self._ewma,
                cusum=0.0,
                alert=False,
                reason="baseline-established",
            )

        self._ewma = EWMA_LAMBDA * score + (1 - EWMA_LAMBDA) * self._ewma
        lower_control_limit = self._baseline_mean - EWMA_MARGIN
        ewma_alert = self._ewma < lower_control_limit

        self._cusum = min(0.0, self._cusum + (score - self._baseline_mean) + CUSUM_SLACK_K)
        cusum_alert = self._cusum < -CUSUM_THRESHOLD_H

        alert = ewma_alert or cusum_alert
        new_alert = alert and not self._alerting
        reason = None
        if ewma_alert and cusum_alert:
            reason = "ewma+cusum"
        elif ewma_alert:
            reason = "ewma"
        elif cusum_alert:
            reason = "cusum"

        signal = DriftSignal(
            metric=self.metric,
            n_samples=self._n,
            value=score,
            baseline_mean=self._baseline_mean,
            ewma=self._ewma,
            cusum=self._cusum,
            alert=alert,
            new_alert=new_alert,
            reason=reason,
        )

        self._alerting = alert
        if alert:
            self._cusum = 0.0

        return signal
