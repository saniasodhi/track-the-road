"""The pipeline conductor - runs steps 1 to 4 in order for a single frame.

    image bytes
        |
        v
    [1] clip_classifier   ->  P(dry), P(damp), P(wet)   ->  clip_wetness
        |
    [2] cv_features       ->  physical_wetness          (independent of CLIP)
        |
        +--> fuse: wetness_raw = 0.65*clip + 0.35*physical
        |
    [3] smoothing         ->  wetness_smoothed (EWMA over 5) + hysteresis band
        |
    [4] trend             ->  slope, trend, DRYING label, tyre call, ETA
        |
        v
    one row in the `frames` table

History is read back out of the database rather than held in memory, so the
result for a given session is identical after a restart, and re-running the demo
always produces exactly the same curve.

Degraded mode
-------------
If CLIP could not be loaded, or inference throws, the frame is still processed
using the classical computer-vision score alone and `model_used` is set to
"cv-fallback". The demo never fully breaks; it just tells you it is running with
one signal instead of two.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from ..db import CONFIG_DIR
from ..models import Frame, Session as TrackSession
from . import cv_features, smoothing, trend as trend_mod, zones as zones_mod
from .clip_classifier import ClipTrackClassifier, clip_wetness_from_probs

log = logging.getLogger("tracksense.pipeline")

CLIP_INPUT_SIZE = 224      # CLIP ViT-B/32 works at 224x224
CLIP_WEIGHT = 0.65         # weight of the CLIP signal in the fused score

# ---------------------------------------------------------------------------
# Single shared model instance. Loaded once at application startup.
# ---------------------------------------------------------------------------
_classifier: Optional[ClipTrackClassifier] = None


def get_classifier() -> ClipTrackClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ClipTrackClassifier(CONFIG_DIR / "prompts.json")
    return _classifier


def startup_load() -> ClipTrackClassifier:
    """Called once from the FastAPI lifespan handler."""
    clf = get_classifier()
    if not clf.available and clf.load_error is None:
        clf.load()
    return clf


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def decode_image(data: bytes) -> tuple[Image.Image, np.ndarray]:
    """Bytes -> (PIL RGB image for CLIP, OpenCV BGR array for the CV step)."""
    pil = Image.open(io.BytesIO(data))
    pil = pil.convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, bgr


def resize_for_clip(pil: Image.Image) -> Image.Image:
    """Downscale so the short side is 224px before handing the image to CLIP.

    The processor would resize anyway, but doing it here keeps CPU cost and
    timing predictable regardless of how large the uploaded photo was.
    """
    w, h = pil.size
    if min(w, h) <= CLIP_INPUT_SIZE:
        return pil
    scale = CLIP_INPUT_SIZE / float(min(w, h))
    return pil.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      Image.LANCZOS)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _history(db: DbSession, session_id: int) -> tuple[list[float], list[float], str | None, list[float]]:
    """Read this session's frames back out of the database, oldest first.

    Returns (raw scores, smoothed scores, previously displayed band, gaps),
    where `gaps` holds the per-frame distance between the two signals for every
    frame that actually had two signals. Frames that ran in cv-fallback are
    skipped rather than counted as a zero gap, because "no second opinion" is
    not the same thing as "the two opinions agreed".
    """
    rows = db.execute(
        select(Frame.wetness_raw, Frame.wetness_smoothed, Frame.state,
               Frame.clip_wetness, Frame.physical_wetness, Frame.model_used)
        .where(Frame.session_id == session_id)
        .order_by(Frame.frame_index)
    ).all()
    raws = [r[0] for r in rows]
    smoothed = [r[1] for r in rows]
    gaps = [abs(r[3] - r[4]) for r in rows if r[5] != "cv-fallback"]
    # "DRYING" is a display label for a DAMP band, so map it back when we need
    # the band the hysteresis rule was last operating on.
    prev_band = None
    if rows:
        last_state = rows[-1][2]
        prev_band = "DAMP" if last_state == "DRYING" else last_state
    return raws, smoothed, prev_band, gaps


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------

def process_frame(
    db: DbSession,
    session: TrackSession,
    image_bytes: bytes,
    image_path: str,
    frame_index: int,
    timestamp_s: float = 0.0,
) -> tuple[Frame, dict]:
    """Run all four steps on one image and persist the result.

    Returns (the saved Frame row, a dict of extra detail that is derived rather
    than stored - sub-scores, raw measurements, the ETA sentence and the reason).
    """
    started = time.perf_counter()
    pil, bgr = decode_image(image_bytes)

    # ---------------- STEP 1: CLIP -------------------------------------------
    clf = get_classifier()
    model_used = "cv-fallback"
    clip_error: str | None = None
    probs = {"p_dry": 0.0, "p_damp": 0.0, "p_wet": 0.0}
    clip_w = None

    if clf.available:
        try:
            probs_full = clf.classify(resize_for_clip(pil))
            probs = {k: probs_full[k] for k in ("p_dry", "p_damp", "p_wet")}
            clip_w = clip_wetness_from_probs(**probs)
            model_used = clf.model_id
        except Exception as exc:                      # inference blew up mid-demo
            clip_error = f"{type(exc).__name__}: {exc}"
            log.warning("CLIP inference failed on frame %s: %s", frame_index, clip_error)
    else:
        clip_error = clf.load_error

    # ---------------- STEP 2: classical computer vision ----------------------
    cv_result = cv_features.analyze_surface(bgr)
    physical = cv_result["physical_wetness"]

    # ---------------- Fuse the two independent opinions ----------------------
    if clip_w is None:
        # Degraded mode: the physical score carries the frame on its own.
        wetness_raw = physical
        clip_w = 0.0
    else:
        wetness_raw = cv_features.fuse(clip_w, physical, clip_weight=CLIP_WEIGHT)

    # ---------------- STEP 3: smooth, then stabilise the band ----------------
    raw_hist, smooth_hist, prev_band, gap_hist = _history(db, session.id)
    raw_hist = raw_hist + [wetness_raw]
    wetness_smoothed = smoothing.ewma(raw_hist)
    smooth_hist = smooth_hist + [wetness_smoothed]
    band = smoothing.stabilise_band(smooth_hist, prev_band)

    # ---------------- How much do the two signals actually agree? ------------
    # The gap between them is smoothed the same way the score is, so a single
    # odd frame does not make the whole readout look unreliable. With no CLIP
    # there is no second opinion, so this is left NULL rather than faked.
    if model_used == "cv-fallback":
        disagreement = None
    else:
        disagreement = smoothing.ewma(gap_hist + [abs(clip_w - physical)])

    # ---------------- STEP 4: direction and advice ---------------------------
    slope = trend_mod.fit_slope(smooth_hist)
    trend_label = trend_mod.classify_trend(slope, len(smooth_hist))
    advice = trend_mod.build_advice(band, trend_label, slope, wetness_smoothed)

    # ---------------- STEP 2b: where on the road is it wet? ------------------
    # Scored against the smoothed value so the grid does not flicker frame to
    # frame any more than the headline number does.
    zone_cells = zones_mod.analyse_zones(bgr, wetness_smoothed, physical)
    zone_summary = zones_mod.summarise(zone_cells, wetness_smoothed)

    latency_ms = (time.perf_counter() - started) * 1000.0

    frame = Frame(
        session_id=session.id,
        frame_index=frame_index,
        image_path=image_path,
        timestamp_s=float(timestamp_s),
        p_dry=probs["p_dry"],
        p_damp=probs["p_damp"],
        p_wet=probs["p_wet"],
        clip_wetness=float(clip_w),
        physical_wetness=float(physical),
        disagreement=None if disagreement is None else float(disagreement),
        wetness_raw=float(wetness_raw),
        wetness_smoothed=float(wetness_smoothed),
        state=advice["state"],
        trend=advice["trend"],
        slope=float(slope),
        recommendation=advice["recommendation"],
        urgency=advice["urgency"],
        model_used=model_used,
        latency_ms=round(latency_ms, 1),
        zones_json=json.dumps({"cells": zone_cells, "summary": zone_summary}),
    )
    db.add(frame)
    session.frame_count = len(smooth_hist)
    db.commit()
    db.refresh(frame)

    detail = {
        "band": advice["band"],
        "reason": advice["reason"],
        "eta_frames": advice["eta_frames"],
        "eta_text": advice["eta_text"],
        "headline": advice["headline"],
        "plain": advice["plain"],
        "cv_subscores": cv_result["subscores"],
        "cv_raw": cv_result["raw"],
        "clip_error": clip_error,
    }
    return frame, detail


def derive_detail(frame: Frame) -> dict:
    """Recompute the derived (non-stored) fields for a frame read back from the DB.

    build_advice is a pure function of (band, trend, slope, smoothed), so a GET
    request can reconstruct the exact same sentences the POST returned without
    storing them.
    """
    band = "DAMP" if frame.state == "DRYING" else frame.state
    advice = trend_mod.build_advice(band, frame.trend, frame.slope, frame.wetness_smoothed)
    return {
        "band": band,
        "reason": advice["reason"],
        "eta_frames": advice["eta_frames"],
        "eta_text": advice["eta_text"],
        "headline": advice["headline"],
        "plain": advice["plain"],
    }
