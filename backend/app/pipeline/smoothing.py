"""PIPELINE STEP 3 - Stop the readout flickering.

The problem
-----------
Per-frame scores are noisy. A cloud passes, the camera auto-exposes, a car
throws spray - and the raw score jumps by 0.05. If the score happens to be
sitting on the 0.55 boundary, the label flips WET / DAMP / WET / DAMP and the
whole product looks broken, even though nothing about the track has changed.

Two fixes, applied in order.

1. EXPONENTIALLY WEIGHTED MOVING AVERAGE over the last 5 frames.
   The newest frame counts most, older frames fade out geometrically. This
   removes single-frame spikes while still reacting within a couple of frames to
   a genuine change - much better than a plain average, which lags badly, or no
   average at all, which is jumpy.

2. HYSTERESIS on the label.
   Even a smoothed score can hover exactly on a boundary. So the displayed label
   is only allowed to change once the smoothed score has been in the new band for
   TWO frames in a row. One frame across the line is noise; two is a trend.

Everything here is recomputed from what is stored in the database rather than
kept in memory, so the answer is identical after a restart and identical on
every run.
"""

from __future__ import annotations

import numpy as np

# Band boundaries on the 0-1 wetness scale.
DRY_MAX = 0.25    # below this: DRY
DAMP_MAX = 0.55   # 0.25 - 0.55: DAMP.  above: WET

EWMA_WINDOW = 5   # how many raw frames the average looks at
EWMA_ALPHA = 0.45 # weight given to the newest frame
CONFIRM_FRAMES = 2  # consecutive frames required before the label may change

BAND_ORDER = ("DRY", "DAMP", "WET")


def band_for(score: float) -> str:
    """Which band does this score fall in? Pure function, no memory."""
    if score < DRY_MAX:
        return "DRY"
    if score < DAMP_MAX:
        return "DAMP"
    return "WET"


def band_bounds(band: str) -> tuple[float, float]:
    """The (lower, upper) edge of a band. Used to estimate time-to-crossing."""
    return {
        "DRY": (0.0, DRY_MAX),
        "DAMP": (DRY_MAX, DAMP_MAX),
        "WET": (DAMP_MAX, 1.0),
    }[band]


def ewma(raw_history: list[float]) -> float:
    """Exponentially weighted average of the last EWMA_WINDOW raw scores.

    `raw_history` is oldest-first and must already include the current frame.

    Weight for the newest frame is alpha, the one before it alpha*(1-alpha), and
    so on. Weights are renormalised so short histories (frame 1, frame 2) are
    still correct rather than being dragged toward zero.
    """
    if not raw_history:
        return 0.0
    window = raw_history[-EWMA_WINDOW:]
    n = len(window)
    # newest gets index 0 -> largest weight
    weights = np.array([EWMA_ALPHA * (1.0 - EWMA_ALPHA) ** k for k in range(n)], dtype=np.float64)
    values = np.array(window[::-1], dtype=np.float64)  # reverse to newest-first
    return float(np.clip((values * weights).sum() / weights.sum(), 0.0, 1.0))


def stabilise_band(
    smoothed_history: list[float],
    previous_band: str | None,
) -> str:
    """Decide which band to *display*, applying the two-frame hysteresis rule.

    smoothed_history : oldest-first list of smoothed scores, including this frame.
    previous_band    : the band that was displayed on the previous frame
                       (None for the first frame of a session).

    The rule: switch only if the current frame AND the previous frame both land
    in the same new band. One frame across the boundary is ignored.
    """
    if not smoothed_history:
        return "DRY"

    candidate = band_for(smoothed_history[-1])

    # First frame of a session - nothing to be stable about yet.
    if previous_band is None:
        return candidate

    if candidate == previous_band:
        return previous_band

    # A change is being proposed. Require CONFIRM_FRAMES consecutive frames
    # sitting in the candidate band before we accept it.
    needed = CONFIRM_FRAMES
    if len(smoothed_history) < needed:
        return previous_band

    recent = smoothed_history[-needed:]
    if all(band_for(v) == candidate for v in recent):
        return candidate

    return previous_band
