"""Zero-shot hazard detectors, managed at runtime.

  GET    /api/hazards          list the detectors currently active
  POST   /api/hazards          create one from a name (and optional phrasings)
  DELETE /api/hazards/{id}     remove one

Creating a detector takes about a tenth of a second: the phrasings are embedded
once with CLIP's text encoder and cached. There is no training step, because
there is nothing to train - see app/pipeline/hazards.py.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..pipeline.orchestrator import get_classifier, get_hazards

router = APIRouter(prefix="/api/hazards", tags=["hazards"])


class HazardCreate(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    # Optional. Left empty, sensible phrasings are generated from the label.
    prompts: list[str] = Field(default_factory=list, max_length=4)


@router.get("")
def list_hazards() -> dict:
    watch = get_hazards()
    return {
        "hazards": watch.list_all(),
        "status": watch.status(),
    }


@router.post("")
def create_hazard(payload: HazardCreate) -> dict:
    watch = get_hazards()
    clf = get_classifier()

    if not clf.available:
        raise HTTPException(
            status_code=503,
            detail="CLIP is not loaded, so new detectors cannot be created. "
                   "The wetness pipeline is still running on the fallback.",
        )
    if not watch.ready:
        watch.load(clf)
    try:
        hazard = watch.add(clf, payload.label, payload.prompts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"hazard": hazard, "status": watch.status()}


@router.delete("/{hazard_id}")
def delete_hazard(hazard_id: str) -> dict:
    watch = get_hazards()
    if not watch.remove(hazard_id):
        raise HTTPException(status_code=404, detail=f"No detector called {hazard_id!r}")
    return {"deleted": hazard_id, "status": watch.status()}
