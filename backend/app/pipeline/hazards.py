"""ZERO-SHOT HAZARD WATCH - add a new detector by typing a sentence.

The point
---------
The wetness pipeline classifies dry / damp / wet because those are the three
things we wrote descriptions for. Nothing about the architecture is limited to
three. CLIP compares an image against *any* sentence, so a brand-new detector
costs one sentence and zero training data.

That is the genuinely useful property of building on a vision-language model
rather than training a classifier, and it is the thing you cannot do with a
model trained on a fixed label set. Type "a road covered in black ice" and a
black-ice detector exists a second later. No images to collect, no labelling,
no retraining, no redeploy.

How a detector is scored
------------------------
Each hazard is a BINARY question, asked independently of the others:

    "Does this image look more like <the hazard> or like an ordinary road?"

We embed the hazard's phrasings into one prototype, embed a fixed set of
"ordinary road" reference phrasings into another, and softmax the two
similarities against each other using CLIP's own temperature. Out comes a
probability between 0 and 1.

Scoring each hazard against a neutral baseline - rather than putting all the
hazards in one softmax together - matters:

  * hazards do not compete with each other, so adding a fifth cannot change the
    reading of the other four
  * a frame can legitimately trigger two at once (wet AND foggy)
  * the number stays interpretable: 0.8 means "looks 4x more like black ice
    than like ordinary road", not "the strongest of whatever I was offered"

Everything here reuses the image vector CLIP already computed for the wetness
classes, so a hazard costs two dot products per frame - effectively free.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("tracksense.hazards")

# What "nothing unusual" looks like. Every hazard is scored against this.
NEUTRAL_PROMPTS = [
    "an ordinary asphalt road surface",
    "a normal road with nothing unusual about it",
    "a plain stretch of tarmac in ordinary conditions",
]

# Above this probability the hazard is called: it goes red on screen and into
# the event log. Deliberately high - a false alarm on a safety readout is worse
# than a near miss you can still see the number for.
TRIGGER_THRESHOLD = 0.60

MAX_HAZARDS = 12
MAX_PROMPTS_PER_HAZARD = 4


class HazardWatch:
    """Runtime registry of zero-shot detectors."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self._hazards: list[dict] = []          # {id, label, prompts, vector, builtin}
        self._neutral: Optional[np.ndarray] = None
        self.ready = False
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ setup

    def _slug(self, label: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "hazard"
        existing = {h["id"] for h in self._hazards}
        if base not in existing:
            return base
        n = 2
        while f"{base}-{n}" in existing:
            n += 1
        return f"{base}-{n}"

    def load(self, classifier) -> bool:
        """Embed the neutral baseline and any hazards defined in config."""
        if not classifier.available:
            self.ready = False
            self.error = "CLIP is not loaded, so zero-shot detection is unavailable"
            return False
        try:
            self._neutral = classifier.embed_prompts(NEUTRAL_PROMPTS)
            self._hazards = []
            for entry in self._read_config():
                self._add(classifier, entry["label"], entry["prompts"],
                          builtin=entry.get("builtin", True))
            self.ready = True
            self.error = None
        except Exception as exc:
            self.ready = False
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("Hazard watch failed to load: %s", self.error)
        return self.ready

    def _read_config(self) -> list[dict]:
        fallback = [
            {"label": "Black ice",
             "prompts": ["a road covered in black ice",
                         "an icy road surface with a thin transparent glaze",
                         "a frozen road in freezing conditions"]},
            {"label": "Snow",
             "prompts": ["a road covered in snow",
                         "a snowy road surface",
                         "a road under a layer of fresh snow"]},
            {"label": "Fog",
             "prompts": ["a road in thick fog with very low visibility",
                         "a foggy road where the distance disappears into haze"]},
        ]
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
            items = raw.get("hazards", [])
            clean = [
                {"label": i["label"], "prompts": i["prompts"], "builtin": True}
                for i in items
                if isinstance(i.get("label"), str) and isinstance(i.get("prompts"), list)
                and i["prompts"]
            ]
            return clean or fallback
        except Exception:
            return fallback

    def _persist(self) -> None:
        """Best effort. A failed write must never break a running demo."""
        try:
            self.config_path.write_text(
                json.dumps({
                    "_comment": "Zero-shot hazard detectors. Each one is just a label and "
                                "a few descriptions - CLIP does the rest, with no training "
                                "data. Edit freely, or add them live from the dashboard.",
                    "hazards": [{"label": h["label"], "prompts": h["prompts"]}
                                for h in self._hazards],
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Could not save hazards to %s: %s", self.config_path, exc)

    # --------------------------------------------------------------- mutation

    def _add(self, classifier, label: str, prompts: list[str], builtin: bool = False) -> dict:
        vector = classifier.embed_prompts(prompts)
        hazard = {
            "id": self._slug(label),
            "label": label.strip()[:40],
            "prompts": prompts,
            "vector": vector,
            "builtin": builtin,
            "added_at": time.time(),
        }
        self._hazards.append(hazard)
        return hazard

    def add(self, classifier, label: str, prompts: list[str] | None = None) -> dict:
        """Create a detector from a label and, optionally, extra phrasings.

        With no phrasings supplied we generate a few from the label itself,
        because "black ice" alone is a weaker probe than three sentences that
        describe black ice on a road.
        """
        if not self.ready:
            raise RuntimeError(self.error or "Hazard watch is not ready")
        if len(self._hazards) >= MAX_HAZARDS:
            raise ValueError(f"Limit is {MAX_HAZARDS} detectors. Remove one first.")

        label = (label or "").strip()
        if not label:
            raise ValueError("A hazard needs a name")

        # Two detectors with the same name are indistinguishable on screen and
        # score identically, so there is no reason to allow it.
        if any(h["label"].lower() == label.lower() for h in self._hazards):
            raise ValueError(f"A detector called {label!r} already exists")

        prompts = [p.strip() for p in (prompts or []) if p and p.strip()]
        if not prompts:
            noun = label.lower()
            prompts = [
                f"a road covered in {noun}",
                f"a road surface with {noun}",
                f"{noun} on the road",
            ]
        prompts = prompts[:MAX_PROMPTS_PER_HAZARD]

        hazard = self._add(classifier, label, prompts, builtin=False)
        self._persist()
        log.info("Added zero-shot detector %r from %d prompt(s)", label, len(prompts))
        return self.describe(hazard)

    def remove(self, hazard_id: str) -> bool:
        before = len(self._hazards)
        self._hazards = [h for h in self._hazards if h["id"] != hazard_id]
        if len(self._hazards) != before:
            self._persist()
            return True
        return False

    # -------------------------------------------------------------- inference

    def detect(self, image_vector: np.ndarray, logit_scale: float) -> list[dict]:
        """Score every registered hazard against this frame.

        Two dot products and a two-way softmax per hazard. The image vector was
        already computed for the wetness classes, so this is essentially free.
        """
        if not self.ready or self._neutral is None or not self._hazards:
            return []

        neutral_sim = float(self._neutral @ image_vector)
        results = []
        for h in self._hazards:
            hazard_sim = float(h["vector"] @ image_vector)
            # Two-way softmax at CLIP's own temperature: hazard vs ordinary road.
            logits = np.array([neutral_sim, hazard_sim]) * logit_scale
            logits -= logits.max()
            exp = np.exp(logits)
            probability = float(exp[1] / exp.sum())
            results.append({
                "id": h["id"],
                "label": h["label"],
                "probability": round(probability, 4),
                "triggered": probability >= TRIGGER_THRESHOLD,
                "builtin": h["builtin"],
            })
        results.sort(key=lambda r: r["probability"], reverse=True)
        return results

    # ------------------------------------------------------------------ views

    def describe(self, hazard: dict) -> dict:
        return {
            "id": hazard["id"],
            "label": hazard["label"],
            "prompts": hazard["prompts"],
            "builtin": hazard["builtin"],
        }

    def list_all(self) -> list[dict]:
        return [self.describe(h) for h in self._hazards]

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "count": len(self._hazards),
            "threshold": TRIGGER_THRESHOLD,
            "labels": [h["label"] for h in self._hazards],
        }
