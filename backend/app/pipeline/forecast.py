"""TIME-TO-DRY FORECAST - how many minutes until this road is safe?

Why a straight line is the wrong model
-------------------------------------
Step 4 fits a straight line through the last ten readings, which is exactly
right for deciding a DIRECTION: it is robust, it needs no assumptions, and a
sign is all it has to produce.

It is the wrong model for a FORECAST. A drying road does not lose water at a
constant rate. Evaporation from a thin film is roughly proportional to how much
water is still there, so the curve decays towards a dry baseline and flattens as
it approaches it:

    w(t) = w_dry + (w_0 - w_dry) * exp(-t / tau)

Extrapolating a straight line through that curve always crosses the dry
threshold too early, and the error grows the further ahead you look - which is
precisely where a forecast has to be trusted. So the direction stays linear and
the forecast is fitted properly.

`tau` is the time constant: the time to lose about 63% of the water still
present. It is the single number that describes how fast this surface, in this
weather, sheds water - and it is exactly the sort of thing a road authority
would want to track per site and per season.

Real time, not frame numbers
----------------------------
"Slicks viable in about 6 frames" is not a unit anybody can act on. This module
works in seconds, taken from each frame's real timestamp, and reports minutes.

It therefore only runs when the timestamps are real: a session has to span
MIN_SPAN_SECONDS of genuine wall-clock time before a forecast is offered. The
bundled demo numbers its frames 0..15 and so is correctly refused - it has no
real timeline to extrapolate.

Honesty gates
-------------
A forecast is only returned when all of these hold:

  * at least MIN_POINTS readings
  * spanning at least MIN_SPAN_SECONDS
  * actually drying (a rising curve gets no dry-time estimate)
  * the fit explains at least MIN_R_SQUARED of the variance
  * the answer lands inside MAX_HORIZON_MINUTES

Otherwise it returns None and the UI shows nothing. A forecast that is not
supported by the data is worse than no forecast.
"""

from __future__ import annotations

import math

import numpy as np

from .smoothing import DRY_MAX, DAMP_MAX

MIN_POINTS = 5
MIN_SPAN_SECONDS = 120.0      # below this the timestamps are not a real timeline
MIN_R_SQUARED = 0.80          # the curve has to actually fit
MAX_HORIZON_MINUTES = 240.0   # do not project more than four hours ahead
MIN_DECAY_RANGE = 0.06        # the surface has to have measurably changed


def _fit_exponential(t: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float] | None:
    """Fit w(t) = baseline + amplitude * exp(-t / tau).

    The baseline is not zero - dry asphalt still reads around 0.10 - and it is
    not known in advance, so it is found by a small search. For each candidate
    baseline the problem becomes linear in log space:

        log(w - baseline) = log(amplitude) - t / tau

    which is an ordinary least-squares line. The candidate with the best R2
    wins. No solver needed, and it cannot fail to converge.

    Returns (baseline, amplitude, tau_seconds, r_squared) or None.
    """
    best = None
    lowest = float(w.min())

    # Baseline must sit below every reading or the log is undefined.
    for baseline in np.linspace(0.0, max(0.0, lowest - 0.005), 24):
        residual = w - baseline
        if np.any(residual <= 1e-4):
            continue
        y = np.log(residual)

        slope, intercept = np.polyfit(t, y, 1)
        if slope >= 0:                      # not decaying
            continue
        tau = -1.0 / slope
        if not math.isfinite(tau) or tau <= 0:
            continue

        predicted = baseline + math.exp(intercept) * np.exp(-t / tau)
        ss_res = float(np.sum((w - predicted) ** 2))
        ss_tot = float(np.sum((w - w.mean()) ** 2))
        if ss_tot <= 1e-12:
            continue
        r2 = 1.0 - ss_res / ss_tot

        if best is None or r2 > best[3]:
            best = (float(baseline), float(math.exp(intercept)), float(tau), float(r2))

    return best


def _cross_time(baseline: float, amplitude: float, tau: float,
                target: float) -> float | None:
    """Solve w(t) = target for t. None if the curve never gets there."""
    residual = target - baseline
    if residual <= 1e-6 or amplitude <= 1e-9:
        return None                          # curve flattens out above the target
    ratio = residual / amplitude
    if ratio <= 0:
        return None
    t = -tau * math.log(ratio)
    return t if math.isfinite(t) else None


def forecast_drying(timestamps_s: list[float], smoothed: list[float]) -> dict | None:
    """Predict when this surface reaches the damp and dry thresholds.

    `timestamps_s` must be real wall-clock seconds. Returns None whenever the
    data does not support an honest answer - see the gates at the top.
    """
    if len(timestamps_s) < MIN_POINTS or len(timestamps_s) != len(smoothed):
        return None

    t = np.asarray(timestamps_s, dtype=np.float64)
    w = np.asarray(smoothed, dtype=np.float64)

    t = t - t[0]
    span = float(t[-1])
    if span < MIN_SPAN_SECONDS:
        return None                          # frame indices, not a real timeline

    if float(w.max() - w.min()) < MIN_DECAY_RANGE:
        return None                          # nothing has actually changed
    if w[-1] >= w[0]:
        return None                          # not drying; no dry-time to give

    fit = _fit_exponential(t, w)
    if fit is None:
        return None
    baseline, amplitude, tau, r2 = fit
    if r2 < MIN_R_SQUARED:
        return None

    now = span
    out: dict = {
        "tau_minutes": round(tau / 60.0, 2),
        "r_squared": round(r2, 4),
        "baseline": round(baseline, 4),
        "observed_minutes": round(span / 60.0, 2),
        "points": len(w),
    }

    for label, target in (("damp", DAMP_MAX), ("dry", DRY_MAX)):
        crossing = _cross_time(baseline, amplitude, tau, target)
        if crossing is None:
            out[f"{label}_at_minutes"] = None
            out[f"minutes_to_{label}"] = None
            continue
        remaining = (crossing - now) / 60.0
        if remaining < -1.0 or remaining > MAX_HORIZON_MINUTES:
            # Already long past, or too far out to claim with a straight face.
            out[f"{label}_at_minutes"] = round(crossing / 60.0, 1)
            out[f"minutes_to_{label}"] = None
            continue
        out[f"{label}_at_minutes"] = round(crossing / 60.0, 1)
        out[f"minutes_to_{label}"] = round(max(0.0, remaining), 1)

    minutes_to_dry = out.get("minutes_to_dry")
    if minutes_to_dry is None and out.get("minutes_to_damp") is None:
        return None                          # nothing useful to say

    if minutes_to_dry is not None:
        out["sentence"] = (
            "Dry in about a minute." if minutes_to_dry < 1.5
            else f"Dry in about {minutes_to_dry:.0f} minutes."
        )
    else:
        m = out["minutes_to_damp"]
        out["sentence"] = f"Down to damp in about {m:.0f} minutes."

    return out


def predict_at(baseline: float, amplitude: float, tau_seconds: float,
               t_seconds: float) -> float:
    """The fitted curve's value at any time. Used to draw the projection."""
    return baseline + amplitude * math.exp(-t_seconds / tau_seconds)
