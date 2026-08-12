"""GET /api/health - the real state of everything, never a hardcoded OK.

Reports on:
  * the CLIP model (loaded? how long did it take? what went wrong if not?)
  * the SQLite database (present? actually writable?)
  * the bundled sample frames (how many, from which folder)
  * the prompt config file

`status` is:
  ok       - CLIP loaded, database writable, samples present
  degraded - the app will still run, but with something missing (usually CLIP,
             in which case every frame falls back to the CV-only score)
  error    - the database is not usable; nothing will work
"""

from __future__ import annotations

from fastapi import APIRouter

from ..db import CONFIG_DIR, db_status
from ..pipeline.orchestrator import get_classifier
from ..sample_source import describe_samples

router = APIRouter(prefix="/api", tags=["health"])

VERSION = "1.0.0"


@router.get("/health")
def health() -> dict:
    clf = get_classifier()
    model = clf.status()
    database = db_status()
    samples = describe_samples()

    prompts_file = CONFIG_DIR / "prompts.json"
    config = {
        "prompts_path": str(prompts_file),
        "prompts_file_present": prompts_file.exists(),
    }

    if not database["writable"]:
        status = "error"
    elif not model["loaded"] or samples["count"] == 0:
        status = "degraded"
    else:
        status = "ok"

    notes = []
    if not model["loaded"]:
        notes.append(
            "CLIP is not loaded - frames are being scored by the classical "
            "computer-vision signal alone (model_used = 'cv-fallback')."
        )
    if samples["count"] == 0:
        notes.append(
            "No sample frames found - the /demo endpoint will not run. "
            "Run: python scripts/generate_samples.py"
        )

    return {
        "status": status,
        "version": VERSION,
        "model": model,
        "database": database,
        "samples": samples,
        "config": config,
        "notes": notes,
    }
