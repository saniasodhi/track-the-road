"""The endpoints that actually run the pipeline.

  POST /api/sessions/{id}/frames  one image  -> one analysed frame
  POST /api/sessions/{id}/video   one video  -> ~1 frame per second, max 60
  POST /api/sessions/{id}/demo    the bundled sample frames, start to finish

The demo endpoint is the safety net for a live presentation: it needs no
upload, no camera and no network, and it always produces the same curve.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.orm import Session as DbSession

from ..db import UPLOAD_DIR, DATA_DIR, get_db
from ..models import Frame, Session as TrackSession
from ..pipeline.orchestrator import (
    decode_image, get_classifier, get_hazards, process_frame, resize_for_clip,
)
from ..sample_source import active_source_name, resolve_sample_dir
from . import frame_to_dict
from .sessions import get_session_or_404, session_to_dict

log = logging.getLogger("tracksense.analyze")
router = APIRouter(prefix="/api/sessions", tags=["analyze"])

MAX_VIDEO_FRAMES = 60
VIDEO_SAMPLE_HZ = 1.0            # pull roughly one frame per second
# Generous, because real dashcam clips are big - a 3-minute 1080p file off the
# Hugging Face dataset is about 300 MB and should just work.
MAX_UPLOAD_BYTES = 600 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


def _next_index(db: DbSession, session_id: int) -> int:
    rows = db.execute(
        select(Frame.frame_index).where(Frame.session_id == session_id)
    ).scalars().all()
    return (max(rows) + 1) if rows else 0


def _session_upload_dir(session_id: int) -> Path:
    d = UPLOAD_DIR / str(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _relative_media_path(path: Path) -> str:
    """Path relative to backend/data, which is what /media serves.

    If someone points TRACKSENSE_SAMPLES_DIR at a folder outside the data
    directory the image cannot be served over /media, so fall back to the parent
    folder plus filename. The analysis still runs and the API still responds -
    only the thumbnail in the UI would 404, which is a far better failure than a
    500 from relative_to().
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        log.warning("%s is outside %s - its image will not be servable over /media",
                    resolved, DATA_DIR)
        return f"{resolved.parent.name}/{resolved.name}"


# --------------------------------------------------------------- single image

@router.post("/{session_id}/frames")
async def upload_frame(
    session_id: int,
    file: UploadFile = File(...),
    db: DbSession = Depends(get_db),
) -> dict:
    session = get_session_or_404(db, session_id)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 200 MB")

    index = _next_index(db, session_id)
    suffix = Path(file.filename or "frame.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        suffix = ".jpg"
    dest = _session_upload_dir(session_id) / f"{index:04d}{suffix}"
    dest.write_bytes(data)

    try:
        frame, detail = process_frame(
            db, session, data, _relative_media_path(dest), index,
            timestamp_s=float(index),
        )
    except Exception as exc:
        # The real traceback goes to the log; the user gets something readable.
        # Pillow's own message quotes a BytesIO object, which is noise to anyone
        # who is simply holding the wrong file.
        log.exception("Pipeline failed on uploaded frame")
        raise HTTPException(
            status_code=422,
            detail=f"That file could not be read as an image. "
                   f"JPEG, PNG, BMP and WebP all work. ({dest.name})",
        ) from exc

    return frame_to_dict(frame, detail)


# ---------------------------------------------------------------------- video

@router.post("/{session_id}/video")
async def upload_video(
    session_id: int,
    file: UploadFile = File(...),
    db: DbSession = Depends(get_db),
) -> dict:
    """Extract ~1 frame per second with OpenCV, cap at 60, run them in order."""
    session = get_session_or_404(db, session_id)
    started = time.perf_counter()

    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    # mkstemp hands back an OPEN file descriptor. On Windows an open handle
    # locks the file, so it has to be closed before anything else touches it -
    # otherwise the delete at the end of this function raises.
    handle_fd, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="tracksense_")
    os.close(handle_fd)
    tmp = Path(tmp_name)
    try:
        size = 0
        with tmp.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413,
                                        detail="Video is larger than 200 MB")
                handle.write(chunk)

        capture = cv2.VideoCapture(str(tmp))
        if not capture.isOpened():
            raise HTTPException(
                status_code=422,
                detail="OpenCV could not open that video. MP4 (H.264) is the safest format.",
            )

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0.1 or fps > 240:
            fps = 25.0                                   # sane default for odd files
        step = max(1, int(round(fps / VIDEO_SAMPLE_HZ)))

        out_dir = _session_upload_dir(session_id)
        index = _next_index(db, session_id)
        results: list[dict] = []
        src_pos = 0
        extracted = 0

        while extracted < MAX_VIDEO_FRAMES:
            ok, bgr = capture.read()
            if not ok:
                break
            if src_pos % step == 0:
                dest = out_dir / f"{index:04d}.jpg"
                cv2.imwrite(str(dest), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                data = dest.read_bytes()
                frame, detail = process_frame(
                    db, session, data, _relative_media_path(dest), index,
                    timestamp_s=round(src_pos / fps, 2),
                )
                results.append(frame_to_dict(frame, detail))
                index += 1
                extracted += 1
            src_pos += 1

        capture.release()

        if not results:
            raise HTTPException(status_code=422,
                                detail="No frames could be read from that video.")

        session.source_type = "video"
        db.commit()
        db.refresh(session)

        return {
            "session": session_to_dict(session),
            "frames": results,
            "extracted": len(results),
            "source_fps": round(fps, 2),
            "sample_every_n_frames": step,
            "seconds": round(time.perf_counter() - started, 2),
        }
    finally:
        # Never let a failed cleanup turn a successful analysis into a 500.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete temporary video %s: %s", tmp, exc)


# ----------------------------------------------------------------------- demo

def _real_timestamps(folder) -> dict:
    """Map filename -> real elapsed seconds, from a capture manifest if present.

    A timed capture (scripts/import_drying_experiment.py) records when each
    frame was actually taken. Using that instead of the frame index is what
    lets the forecast answer in minutes rather than frames.
    """
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return {f["file"]: float(f["elapsed_s"])
                for f in data.get("frames", []) if "elapsed_s" in f}
    except Exception as exc:
        log.warning("Could not read timestamps from %s: %s", manifest, exc)
        return {}


def _resolve_or_400(source: str | None):
    try:
        return resolve_sample_dir(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _no_frames_detail(folder, source: str | None) -> str:
    if source == "hf":
        return (f"No Hugging Face frames in {folder}. "
                "Run: python scripts/import_hf_dashcam.py")
    if source == "real":
        return (f"No photos in {folder}. Drop at least 4 images in there - "
                "see the README in that folder.")
    return (f"No sample frames found in {folder}. "
            "Run: python scripts/generate_samples.py")


@router.post("/{session_id}/demo")
def run_demo(session_id: int, source: str | None = None,
             db: DbSession = Depends(get_db)) -> dict:
    """Run every sample frame through the pipeline, in order.

    `source` picks which set of frames to use - "bundled" (the synthetic track
    sequence), "hf" (real frames from a Hugging Face dataset), or "real" (your
    own photos). Omit it to let the backend choose.

    Any frames already attached to the session are cleared first, so hitting
    the button twice gives the same answer rather than appending a second run.
    """
    session = get_session_or_404(db, session_id)
    started = time.perf_counter()

    folder, images, using_real = _resolve_or_400(source)
    if not images:
        raise HTTPException(status_code=503, detail=_no_frames_detail(folder, source))

    db.execute(delete(Frame).where(Frame.session_id == session_id))
    session.frame_count = 0
    session.source_type = "demo"
    db.commit()

    stamps = _real_timestamps(folder)
    results = []
    for index, path in enumerate(images):
        data = path.read_bytes()
        frame, detail = process_frame(
            db, session, data, _relative_media_path(path), index,
            timestamp_s=stamps.get(path.name, float(index)),
        )
        results.append(frame_to_dict(frame, detail))

    db.refresh(session)
    return {
        "session": session_to_dict(session),
        "frames": results,
        "source_dir": str(folder),
        "source": source or active_source_name(),
        "using_real_photos": using_real,
        "seconds": round(time.perf_counter() - started, 2),
    }


@router.post("/{session_id}/rescore-hazards")
def rescore_hazards(session_id: int, db: DbSession = Depends(get_db)) -> dict:
    """Re-run zero-shot hazard detection over every frame already in a session.

    Why this exists: detectors are created at runtime, so a detector added at
    12:05 has never seen the frames analysed at 12:04. Without this the new
    detector appears on screen with no reading against it until fresh frames
    arrive - which is fine for a live camera, and useless for a recorded
    session you are looking at right now.

    Only the hazard layer is recomputed. Wetness, trend, zones and the tyre
    call are untouched, so re-scoring can never change the reading - it only
    fills in the column that was missing.

    Cost is one CLIP image pass per frame (~85 ms), because the image vector is
    not stored. For a 20-frame session that is under two seconds.
    """
    session = get_session_or_404(db, session_id)

    clf = get_classifier()
    watch = get_hazards()
    if not clf.available or not watch.ready:
        raise HTTPException(
            status_code=503,
            detail="CLIP is not loaded, so hazards cannot be scored.",
        )

    frames = db.execute(
        select(Frame).where(Frame.session_id == session_id).order_by(Frame.frame_index)
    ).scalars().all()

    started = time.perf_counter()
    updated = 0
    for frame in frames:
        path = (DATA_DIR / frame.image_path).resolve()
        if not path.is_file():
            # Frame image was cleaned up or lives outside the media root.
            continue
        try:
            pil, _ = decode_image(path.read_bytes())
            probs = clf.classify(resize_for_clip(pil))
            hits = watch.detect(probs["image_vector"], clf.logit_scale)
            frame.hazards_json = json.dumps(hits) if hits else None
            updated += 1
        except Exception as exc:
            log.warning("Could not re-score frame %s: %s", frame.id, exc)

    db.commit()

    refreshed = db.execute(
        select(Frame).where(Frame.session_id == session_id).order_by(Frame.frame_index)
    ).scalars().all()

    return {
        "session_id": session_id,
        "rescored": updated,
        "seconds": round(time.perf_counter() - started, 2),
        "frames": [frame_to_dict(f) for f in refreshed],
    }


@router.post("/{session_id}/demo/step")
def run_demo_step(session_id: int, index: int = 0, source: str | None = None,
                  db: DbSession = Depends(get_db)) -> dict:
    """Analyse a single bundled sample frame.

    Same pipeline, same results as /demo - it just returns after one frame so
    the dashboard can fill its timeline with real results as they land, and show
    real per-frame latency, instead of waiting on one long request.

    Calling it with index=0 resets the session, so a replay is identical to a
    first run. If anything about this route misbehaves the frontend falls back
    to the batch /demo endpoint.
    """
    session = get_session_or_404(db, session_id)
    folder, images, using_real = _resolve_or_400(source)
    if not images:
        raise HTTPException(status_code=503, detail=_no_frames_detail(folder, source))
    if index < 0 or index >= len(images):
        raise HTTPException(status_code=400,
                            detail=f"index must be between 0 and {len(images) - 1}")

    if index == 0:
        db.execute(delete(Frame).where(Frame.session_id == session_id))
        session.frame_count = 0
        session.source_type = "demo"
        db.commit()

    path = images[index]
    stamps = _real_timestamps(folder)
    frame, detail = process_frame(
        db, session, path.read_bytes(), _relative_media_path(path), index,
        timestamp_s=stamps.get(path.name, float(index)),
    )
    return {
        "frame": frame_to_dict(frame, detail),
        "index": index,
        "total": len(images),
        "done": index + 1 >= len(images),
        "source": source or active_source_name(),
        "using_real_photos": using_real,
    }
