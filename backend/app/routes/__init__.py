"""Shared helpers for the API routes."""

from __future__ import annotations

import json
import logging

from ..models import Frame
from ..pipeline.orchestrator import derive_detail

log = logging.getLogger("tracksense.routes")

MEDIA_PREFIX = "/media"

# A gap this wide between the two signals means we have learned nothing -
# one method says dry and the other says soaked. Anything at or beyond it
# scores zero agreement.
DISAGREEMENT_FULL = 0.50


def _agreement(frame: Frame) -> tuple[float | None, float | None, float | None]:
    """Turn the signal gap into (agreement, band_low, band_high).

    The band is not an invented statistic. Its width IS the distance between
    the two independent estimates, so a narrow band literally means "CLIP and
    the optics landed in the same place" and a wide one means "these two
    methods do not agree, treat this reading with care".

    Returns (None, None, None) when the frame ran without CLIP, because with a
    single signal there is nothing to agree or disagree with.
    """
    if frame.disagreement is None:
        return None, None, None

    gap = max(0.0, float(frame.disagreement))
    agreement = 1.0 - min(1.0, gap / DISAGREEMENT_FULL)
    half = gap / 2.0
    low = max(0.0, frame.wetness_smoothed - half)
    high = min(1.0, frame.wetness_smoothed + half)
    return round(agreement, 4), round(low, 4), round(high, 4)


def frame_to_dict(frame: Frame, detail: dict | None = None) -> dict:
    """Turn a Frame row into the JSON the frontend consumes.

    `detail` holds the fields that are derived rather than stored (the reason
    sentence, the ETA, the CV sub-scores). When it is not supplied - i.e. on a
    GET rather than the POST that created the frame - the derivable parts are
    recomputed from the stored numbers.
    """
    detail = detail or derive_detail(frame)
    agreement, band_low, band_high = _agreement(frame)

    # Zones are stored as a JSON string. A frame written before the grid existed
    # simply has none - the UI hides the overlay rather than breaking.
    zone_cells, zone_summary = [], None
    if frame.zones_json:
        try:
            parsed = json.loads(frame.zones_json)
            zone_cells = parsed.get("cells", [])
            zone_summary = parsed.get("summary")
        except (ValueError, AttributeError) as exc:
            log.warning("Frame %s has unreadable zone data: %s", frame.id, exc)

    return {
        "id": frame.id,
        "session_id": frame.session_id,
        "frame_index": frame.frame_index,
        "image_url": f"{MEDIA_PREFIX}/{frame.image_path}",
        "timestamp_s": frame.timestamp_s,
        "p_dry": frame.p_dry,
        "p_damp": frame.p_damp,
        "p_wet": frame.p_wet,
        "clip_wetness": frame.clip_wetness,
        "physical_wetness": frame.physical_wetness,
        # How far apart the two signals are, and what that means for confidence.
        # All three are null when the frame ran without CLIP.
        "disagreement": frame.disagreement,
        "agreement": agreement,
        "band_low": band_low,
        "band_high": band_high,
        "wetness_raw": frame.wetness_raw,
        "wetness_smoothed": frame.wetness_smoothed,
        "state": frame.state,
        "band": detail.get("band", frame.state),
        "trend": frame.trend,
        "slope": frame.slope,
        "recommendation": frame.recommendation,
        "urgency": frame.urgency,
        "reason": detail.get("reason", ""),
        "eta_frames": detail.get("eta_frames"),
        "eta_text": detail.get("eta_text"),
        "headline": detail.get("headline", ""),
        "plain": detail.get("plain", ""),
        "zones": zone_cells,
        "zone_summary": zone_summary,
        "model_used": frame.model_used,
        "latency_ms": frame.latency_ms,
        "cv_subscores": detail.get("cv_subscores"),
        "cv_raw": detail.get("cv_raw"),
        "clip_error": detail.get("clip_error"),
    }
