"""PIPELINE STEP 4 - Which way is it going, and what should the team do?

The idea this whole product rests on
------------------------------------
You cannot tell from a single photograph whether a damp track is drying out or
getting wetter. The pixels are identical. A road at 0.40 wetness on its way down
looks exactly like a road at 0.40 wetness on its way up.

Drying is not something a surface *looks like*. It is a direction.

So CLIP is never asked "is it drying?" - it can only ever answer dry, damp or
wet, because those are the only three things visible in one frame. DRYING is
computed here, from the shape of the last ten smoothed scores. That separation
is the whole point: perception is per-frame, direction is per-sequence.

How the direction is measured
-----------------------------
Fit a straight line (ordinary least squares) through the last 10 smoothed scores
against frame index and take its slope, in wetness units per frame. A line fit
is used rather than "compare first and last" because it uses every point and is
far less sensitive to one bad frame at either end.

    slope < -0.015  ->  IMPROVING     (wetness falling: the track is drying)
    slope > +0.015  ->  DETERIORATING (wetness rising: more water arriving)
    otherwise       ->  STABLE

And then the one substitution that makes the readout useful:

    band == DAMP  and  trend == IMPROVING   ->  display "DRYING"

Tyre advice
-----------
Deliberately a plain lookup table on (band, trend), not a model. Anything that
decides whether a driver goes out on slicks has to be readable and arguable by a
human being. You can see every threshold in RECOMMENDATIONS below and change one
without retraining anything.
"""

from __future__ import annotations

import numpy as np

from .smoothing import band_bounds, band_for, DAMP_MAX, DRY_MAX

TREND_WINDOW = 10          # how many smoothed scores the line is fitted through
MIN_POINTS_FOR_TREND = 3   # below this, a slope is meaningless
IMPROVING_SLOPE = -0.015   # wetness units per frame
DETERIORATING_SLOPE = 0.015
MAX_ETA_FRAMES = 60        # do not project further ahead than this


def fit_slope(smoothed_history: list[float]) -> float:
    """Least-squares slope of the last TREND_WINDOW smoothed scores, per frame."""
    window = smoothed_history[-TREND_WINDOW:]
    n = len(window)
    if n < MIN_POINTS_FOR_TREND:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(window, dtype=np.float64)
    # polyfit degree 1 -> [slope, intercept]
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def classify_trend(slope: float, n_points: int) -> str:
    """Turn a slope into IMPROVING / STABLE / DETERIORATING."""
    if n_points < MIN_POINTS_FOR_TREND:
        return "STABLE"
    if slope < IMPROVING_SLOPE:
        return "IMPROVING"
    if slope > DETERIORATING_SLOPE:
        return "DETERIORATING"
    return "STABLE"


def display_state(band: str, trend: str) -> str:
    """The label the user sees. DRYING is a damp track with a downward slope."""
    if band == "DAMP" and trend == "IMPROVING":
        return "DRYING"
    return band


def estimate_crossing(current: float, band: str, slope: float,
                      trend: str = "STABLE") -> tuple[int | None, str | None]:
    """How many frames until the score leaves its current band, at this rate?

    Returns (frames, human sentence) or (None, None) when nothing is projected -
    the track is stable, or the crossing is too far away to be honest about.

    Nothing is ever projected from a STABLE trend. Having just decided that a
    slope that small is noise rather than movement, extrapolating from it would
    contradict our own judgement - and it produces genuinely silly readouts like
    "holding steady, full wets needed in 4 frames". If the direction is not
    trusted, no forecast is offered.
    """
    if trend == "STABLE" or abs(slope) < 1e-6:
        return None, None

    lower, upper = band_bounds(band)

    if slope < 0:                      # drying: heading for the lower edge
        if band == "DRY":
            return None, None          # already at the bottom band
        target = lower
        next_band = "DAMP" if band == "WET" else "DRY"
    else:                              # wetting: heading for the upper edge
        if band == "WET":
            return None, None          # already at the top band
        target = upper
        next_band = "DAMP" if band == "DRY" else "WET"

    frames = (target - current) / slope
    if frames <= 0 or frames > MAX_ETA_FRAMES:
        return None, None

    n = int(round(frames))
    n = max(1, n)

    if slope < 0:
        phrase = {
            "DAMP": f"intermediates viable in about {n} frames",
            "DRY": f"slicks viable in about {n} frames",
        }[next_band]
    else:
        phrase = {
            "DAMP": f"crossing into damp in about {n} frames",
            "WET": f"full wets needed in about {n} frames",
        }[next_band]
    return n, phrase


# ---------------------------------------------------------------------------
# The safety table. Plain rules, human readable, no model involved.
# Keyed by (band, trend) -> (action, urgency, why)
# ---------------------------------------------------------------------------

RECOMMENDATIONS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("DRY", "IMPROVING"): (
        "SLICKS", "routine",
        "Surface is dry and still improving. Full dry-weather grip available.",
    ),
    ("DRY", "STABLE"): (
        "SLICKS", "routine",
        "Surface is dry and holding steady. Full dry-weather grip available.",
    ),
    ("DRY", "DETERIORATING"): (
        "SLICKS — MONITOR", "caution",
        "Still dry, but wetness is climbing. Keep intermediates ready in the pit lane.",
    ),
    ("DAMP", "IMPROVING"): (
        "INTERMEDIATES — PREPARE SLICKS", "caution",
        "Track is drying. Intermediates now, but a slick crossover is approaching.",
    ),
    ("DAMP", "STABLE"): (
        "INTERMEDIATES", "caution",
        "Persistently damp with no clear direction. Intermediates are the safe call.",
    ),
    ("DAMP", "DETERIORATING"): (
        "INTERMEDIATES — PREPARE WETS", "urgent",
        "Damp and getting worse. Water is arriving faster than it is clearing.",
    ),
    ("WET", "IMPROVING"): (
        "FULL WETS — PREPARE INTERMEDIATES", "urgent",
        "Standing water is clearing but grip is still low. Hold wets for now.",
    ),
    ("WET", "STABLE"): (
        "FULL WETS", "urgent",
        "Heavy standing water with no sign of clearing. Aquaplaning risk is high.",
    ),
    ("WET", "DETERIORATING"): (
        "FULL WETS — CONSIDER RED FLAG", "critical",
        "Standing water is still increasing. Conditions may become undriveable.",
    ),
}


def recommend(band: str, trend: str) -> tuple[str, str, str]:
    """Look up the tyre call. Falls back to the most cautious sensible option."""
    return RECOMMENDATIONS.get((band, trend), ("INTERMEDIATES", "caution",
                                               "Conditions unclear. Defaulting to the cautious call."))


# Plain English for people who are not reading the numbers. The dashboard leads
# with this sentence, because "The surface is damp but drying out" is what a
# person actually needs, and "0.375, slope -0.029" is the evidence for it.
_CONDITION_WORDS = {
    "DRY": "The surface is dry",
    "DAMP": "The surface is damp",
    "WET": "The surface is soaking wet",
}

_DIRECTION_WORDS = {
    "IMPROVING": "and drying out",
    "DETERIORATING": "and getting wetter",
    "STABLE": "and holding steady",
}


def plain_summary(state: str, band: str, trend: str, eta_text: str | None) -> str:
    """One sentence anyone can act on, with no numbers in it."""
    if state == "DRYING":
        sentence = "The surface is damp but drying out"
    elif band == "DRY" and trend == "IMPROVING":
        # "dry and drying out" is nonsense - it has already arrived.
        sentence = "The surface is dry"
    else:
        sentence = f"{_CONDITION_WORDS.get(band, 'The surface is damp')} " \
                   f"{_DIRECTION_WORDS.get(trend, 'and holding steady')}"

    if eta_text:
        # "slicks viable in about 6 frames" -> "Slicks viable in about 6 frames."
        return f"{sentence}. {eta_text[0].upper()}{eta_text[1:]}."
    return f"{sentence}."


# What to say instead of a tyre call when the frame cannot be trusted. A
# confident recommendation from an unreadable image is worse than no
# recommendation, because someone might act on it.
LOW_LIGHT_PLAIN = {
    "low": "Too dark to read this surface reliably.",
    "dark": "There is not enough light to see the road at all.",
}

LOW_LIGHT_ADVICE = {
    "low": ("HOLD — VERIFY CONDITIONS",
            "caution",
            "Light is too poor to read the surface reliably. Keep the current "
            "tyres and confirm by another means before changing."),
    "dark": ("NO READING — LIGHT TOO LOW",
             "caution",
             "The surface is not visible in this frame. This camera cannot "
             "advise until it has light or illumination."),
}


def build_advice(band: str, trend: str, slope: float, smoothed: float,
                 light_level: str = "ok") -> dict:
    """Everything step 4 produces, in one dictionary.

    `light_level` can veto the recommendation. The wetness number is still
    reported - hiding it would lose information - but the ADVICE becomes a hold,
    because a confident tyre call derived from an unreadable frame is the one
    output of this system that could actually get somebody hurt.
    """
    state = display_state(band, trend)
    action, urgency, reason = recommend(band, trend)
    if light_level in LOW_LIGHT_ADVICE:
        action, urgency, reason = LOW_LIGHT_ADVICE[light_level]
    eta_frames, eta_text = estimate_crossing(smoothed, band, slope, trend)

    # One-line status for under the big number.
    direction = {
        "IMPROVING": "Drying",
        "DETERIORATING": "Getting wetter",
        "STABLE": "Holding steady",
    }[trend]
    headline = f"{direction} — {eta_text}" if eta_text else f"{direction} at {slope:+.3f} per frame"

    return {
        "state": state,
        "band": band,
        "trend": trend,
        "slope": slope,
        "recommendation": action,
        "urgency": urgency,
        "reason": reason,
        "eta_frames": eta_frames,
        "eta_text": eta_text,
        "headline": headline,
        "plain": (LOW_LIGHT_PLAIN[light_level]
                  if light_level in LOW_LIGHT_PLAIN
                  else plain_summary(state, band, trend, eta_text)),
        "light_level": light_level,
    }
