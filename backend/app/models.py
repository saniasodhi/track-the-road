"""Database tables.

Two tables only:

  sessions - one run of the analyser (a demo run, a video, or a set of uploads)
  frames   - one analysed image, with every intermediate number the pipeline produced

Storing the intermediate numbers (rather than only the final answer) is deliberate:
it is what lets the UI show the CLIP signal and the classical computer-vision signal
side by side, and it makes the whole pipeline auditable after the fact.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    # "demo" | "images" | "video"
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, default="images")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    frames: Mapped[list["Frame"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Frame.frame_index",
    )


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(String(400), nullable=False)
    timestamp_s: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # --- step 1: CLIP ---
    p_dry: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p_damp: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    p_wet: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    clip_wetness: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # --- step 2: classical computer vision ---
    physical_wetness: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # How far apart the two independent signals were, smoothed over recent
    # frames. This is the honest measure of uncertainty: when a vision-language
    # model and a physical measurement disagree, we genuinely do not know.
    # NULL means there was no second opinion to compare against, i.e. the frame
    # ran in cv-fallback mode.
    disagreement: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    # The 3x3 road grid from pipeline/zones.py, stored as a JSON string because
    # SQLite has no array type. Holds one entry per cell with its wetness, band
    # and the corner coordinates it was measured at.
    zones_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    # Zero-shot hazard detections for this frame (pipeline/hazards.py), stored
    # as a JSON list. NULL when no detectors are registered or CLIP is down.
    hazards_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    # --- step 3: fusion + smoothing ---
    wetness_raw: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    wetness_smoothed: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # --- step 4: state, trend, advice ---
    state: Mapped[str] = mapped_column(String(12), default="DRY", nullable=False)
    trend: Mapped[str] = mapped_column(String(16), default="STABLE", nullable=False)
    slope: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    recommendation: Mapped[str] = mapped_column(String(80), default="SLICKS", nullable=False)
    urgency: Mapped[str] = mapped_column(String(16), default="routine", nullable=False)

    # --- provenance ---
    model_used: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    session: Mapped[Session] = relationship(back_populates="frames")


Index("ix_frames_session_index", Frame.session_id, Frame.frame_index)
