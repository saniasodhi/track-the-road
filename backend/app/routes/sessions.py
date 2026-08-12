"""Session lifecycle and read endpoints.

  POST /api/sessions              start a session
  GET  /api/sessions              list recent sessions
  GET  /api/sessions/{id}         summary (with the latest frame)
  GET  /api/sessions/{id}/frames  every frame, used by the chart and timeline
  DELETE /api/sessions/{id}       tidy up
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from ..db import get_db
from ..models import Frame, Session as TrackSession
from ..schemas import SessionCreate
from . import frame_to_dict

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_session_or_404(db: DbSession, session_id: int) -> TrackSession:
    session = db.get(TrackSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


def _default_name(source_type: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%H:%M")
    return {
        "demo": f"Demo run {stamp}",
        "video": f"Video session {stamp}",
        "images": f"Session {stamp}",
    }.get(source_type, f"Session {stamp}")


def session_to_dict(session: TrackSession) -> dict:
    return {
        "id": session.id,
        "name": session.name,
        "source_type": session.source_type,
        "created_at": session.created_at,
        "frame_count": session.frame_count,
    }


@router.post("")
def create_session(payload: SessionCreate | None = None,
                   db: DbSession = Depends(get_db)) -> dict:
    payload = payload or SessionCreate()
    session = TrackSession(
        name=(payload.name or _default_name(payload.source_type)).strip()[:120],
        source_type=payload.source_type,
        frame_count=0,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session_to_dict(session)


@router.get("")
def list_sessions(limit: int = 20, db: DbSession = Depends(get_db)) -> list[dict]:
    rows = db.execute(
        select(TrackSession).order_by(TrackSession.id.desc()).limit(max(1, min(limit, 100)))
    ).scalars().all()
    return [session_to_dict(s) for s in rows]


@router.get("/{session_id}")
def get_session(session_id: int, db: DbSession = Depends(get_db)) -> dict:
    session = get_session_or_404(db, session_id)
    frames = db.execute(
        select(Frame).where(Frame.session_id == session_id).order_by(Frame.frame_index)
    ).scalars().all()

    out = session_to_dict(session)
    if not frames:
        out.update({
            "latest": None, "mean_wetness": None, "min_wetness": None,
            "max_wetness": None, "state_counts": {}, "model_used": None,
            "total_latency_ms": 0.0,
        })
        return out

    smoothed = [f.wetness_smoothed for f in frames]
    counts: dict[str, int] = {}
    for f in frames:
        counts[f.state] = counts.get(f.state, 0) + 1

    out.update({
        "latest": frame_to_dict(frames[-1]),
        "mean_wetness": round(sum(smoothed) / len(smoothed), 4),
        "min_wetness": round(min(smoothed), 4),
        "max_wetness": round(max(smoothed), 4),
        "state_counts": counts,
        "model_used": frames[-1].model_used,
        "total_latency_ms": round(sum(f.latency_ms for f in frames), 1),
    })
    return out


@router.get("/{session_id}/frames")
def list_frames(session_id: int, db: DbSession = Depends(get_db)) -> dict:
    session = get_session_or_404(db, session_id)
    frames = db.execute(
        select(Frame).where(Frame.session_id == session_id).order_by(Frame.frame_index)
    ).scalars().all()
    return {
        "session_id": session.id,
        "frame_count": len(frames),
        "frames": [frame_to_dict(f) for f in frames],
    }


@router.delete("/{session_id}")
def delete_session(session_id: int, db: DbSession = Depends(get_db)) -> dict:
    session = get_session_or_404(db, session_id)
    db.delete(session)
    db.commit()
    return {"deleted": session_id}
