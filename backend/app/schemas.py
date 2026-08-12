"""Request and response shapes (Pydantic v2).

Every frame response carries `model_used` so the frontend can always tell the
user which engine produced the number - the real CLIP model or the
computer-vision fallback. That field is never optional.

A note on how these are used
----------------------------
`SessionCreate` is wired into the POST /api/sessions route, so incoming request
bodies really are validated.

The *output* models are the documented contract, but the routes return plain
dicts rather than declaring `response_model=`. That is deliberate. A response
model runs validation after the pipeline has already done its work, so a single
unexpected value would turn a completed analysis into a 500 in front of an
audience - and it silently drops any field not declared here. Given the demo
must never break, the routes hand back exactly what they computed. Keep these
classes in step with `routes/__init__.py:frame_to_dict` when you change a field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- input

class SessionCreate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)
    source_type: Literal["demo", "images", "video"] = "images"


# -------------------------------------------------------------------- output

class FrameOut(BaseModel):
    id: int
    session_id: int
    frame_index: int
    image_url: str
    timestamp_s: float

    # step 1
    p_dry: float
    p_damp: float
    p_wet: float
    clip_wetness: float
    # step 2
    physical_wetness: float
    # agreement between the two signals - all null in cv-fallback mode, because
    # a single signal has nothing to agree with
    disagreement: Optional[float] = None
    agreement: Optional[float] = None
    band_low: Optional[float] = None
    band_high: Optional[float] = None
    # step 2b: the 3x3 road grid. `zones` holds one entry per cell (wetness,
    # band, and the corner coordinates it was measured at, as fractions of the
    # image). `zone_summary` names the wettest and driest cells.
    zones: list[dict[str, Any]] = []
    zone_summary: Optional[dict[str, Any]] = None
    # step 3
    wetness_raw: float
    wetness_smoothed: float
    # step 4
    state: Literal["DRY", "DAMP", "WET", "DRYING"]
    band: Literal["DRY", "DAMP", "WET"]
    trend: Literal["IMPROVING", "STABLE", "DETERIORATING"]
    slope: float
    recommendation: str
    urgency: Literal["routine", "caution", "urgent", "critical"]
    reason: str
    eta_frames: Optional[int] = None
    eta_text: Optional[str] = None
    headline: str

    # provenance
    model_used: str
    latency_ms: float

    # only present on the POST that created the frame
    cv_subscores: Optional[dict[str, float]] = None
    cv_raw: Optional[dict[str, float]] = None
    clip_error: Optional[str] = None


class SessionOut(BaseModel):
    id: int
    name: str
    source_type: str
    created_at: datetime
    frame_count: int


class SessionSummary(SessionOut):
    latest: Optional[FrameOut] = None
    mean_wetness: Optional[float] = None
    min_wetness: Optional[float] = None
    max_wetness: Optional[float] = None
    state_counts: dict[str, int] = {}
    model_used: Optional[str] = None
    total_latency_ms: float = 0.0


class FrameListOut(BaseModel):
    session_id: int
    frame_count: int
    frames: list[FrameOut]


class DemoOut(BaseModel):
    session: SessionOut
    frames: list[FrameOut]
    source_dir: str
    seconds: float


class HealthOut(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    model: dict[str, Any]
    database: dict[str, Any]
    samples: dict[str, Any]
    config: dict[str, Any]
    # Plain-English explanations of anything that is not "ok". The dashboard
    # shows the first of these in its degraded banner.
    notes: list[str] = []


class DemoStepOut(BaseModel):
    """One frame of a stepped demo run (POST /api/sessions/{id}/demo/step)."""
    frame: FrameOut
    index: int
    total: int
    done: bool
    using_real_photos: bool
