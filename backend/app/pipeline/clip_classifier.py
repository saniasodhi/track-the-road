"""PIPELINE STEP 1 - What does CLIP think this surface is?

The short version
----------------
CLIP is a Hugging Face model that was trained on hundreds of millions of
(image, caption) pairs. It learned to put a picture and its caption in the same
place in a 512-dimensional space. That means you can hand it a photo and a list
of candidate captions and ask "which of these sentences best describes this
picture?" - without ever training anything yourself.

There is no ready-made "is this road wet" model, so we build a classifier out of
CLIP by writing the captions ourselves:

    "a dry asphalt race track"        -> dry
    "a damp race track surface"       -> damp
    "a soaking wet race track"        -> wet

Why several captions per category (prompt ensembling)
-----------------------------------------------------
A single sentence is a noisy probe. "a wet road" might latch onto rain, or onto
night-time, or onto a particular camera angle. So for each category we write 3-4
different phrasings, embed all of them, average the resulting vectors, and
re-normalise. The average sits in the middle of everything those sentences have
in common - the *wetness* - and the accidental extras partly cancel out. This is
the same trick the original CLIP paper used to lift ImageNet accuracy, and it is
why this step is real work rather than one API call.

The prompts live in backend/config/prompts.json so they can be tuned without
touching this file.

Cost
----
The text side is computed exactly once, at startup. Per frame we only run the
image encoder (a ViT-B/32) and then three dot products. On CPU that is roughly
50-150 ms for a 224x224 image.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger("tracksense.clip")

# The three categories CLIP is ever allowed to answer with.
# Note there is deliberately no "drying" here - see trend.py for why.
CATEGORY_ORDER = ("dry", "damp", "wet")

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


class ClipTrackClassifier:
    """Wraps a Hugging Face CLIP model as a dry / damp / wet classifier.

    Load it once at application startup and keep it in memory - loading the
    weights takes a few seconds, running a frame takes a fraction of a second.
    """

    def __init__(self, prompts_path: Path, model_id: Optional[str] = None):
        self.prompts_path = Path(prompts_path)
        self.model_id = model_id or DEFAULT_MODEL_ID

        self.available = False           # True once the weights are loaded and usable
        self.load_error: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self.prompts: dict[str, list[str]] = {}

        self._model = None
        self._processor = None
        self._torch = None
        # (3, 512) matrix: one averaged, normalised text embedding per category.
        self._text_prototypes: Optional[np.ndarray] = None
        self._logit_scale: float = 100.0  # CLIP's learned temperature, ~100

    # ------------------------------------------------------------------ setup

    def load_prompts(self) -> dict[str, list[str]]:
        """Read the caption bank from config/prompts.json (with a safe fallback)."""
        fallback = {
            "dry": ["a dry asphalt race track", "dry road surface in sunlight"],
            "damp": ["a damp race track surface", "slightly wet asphalt"],
            "wet": ["a soaking wet race track", "standing water on asphalt"],
        }
        try:
            raw = json.loads(self.prompts_path.read_text(encoding="utf-8"))
            cats = raw.get("categories", {})
            prompts = {}
            for name in CATEGORY_ORDER:
                phrases = [p for p in cats.get(name, []) if isinstance(p, str) and p.strip()]
                prompts[name] = phrases or fallback[name]
            if raw.get("model_id"):
                self.model_id = raw["model_id"]
            self.prompts = prompts
        except Exception as exc:
            log.warning("Could not read %s (%s) - using built-in prompts", self.prompts_path, exc)
            self.prompts = fallback
        return self.prompts

    def load(self) -> bool:
        """Load the CLIP weights and pre-compute the text prototypes.

        Returns True on success. On failure it records the reason and returns
        False - the caller then falls back to the computer-vision-only path so
        the demo still runs.
        """
        started = time.perf_counter()
        self.load_prompts()
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            # Determinism: same input -> same output, every run.
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(False)  # not needed on CPU, and safer
            torch.set_grad_enabled(False)
            # Keep CPU thread count modest so a laptop stays responsive during a demo.
            try:
                torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))
            except Exception:
                pass

            self._torch = torch
            self._model = CLIPModel.from_pretrained(self.model_id)
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model.eval()

            self._build_text_prototypes()
            self._logit_scale = float(self._model.logit_scale.exp().item())

            self.available = True
            self.load_error = None
        except Exception as exc:
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("CLIP failed to load: %s", self.load_error)
        finally:
            self.load_seconds = round(time.perf_counter() - started, 2)
        return self.available

    def _build_text_prototypes(self) -> None:
        """Embed every caption, average per category, re-normalise. Runs once."""
        torch = self._torch
        prototypes = []
        for name in CATEGORY_ORDER:
            phrases = self.prompts[name]
            inputs = self._processor(text=phrases, return_tensors="pt", padding=True)
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)      # (n_phrases, 512)
            feats = feats / feats.norm(dim=-1, keepdim=True)         # unit length each
            mean = feats.mean(dim=0)                                 # the ensemble
            mean = mean / mean.norm()                                # back onto the sphere
            prototypes.append(mean.cpu().numpy())
        self._text_prototypes = np.stack(prototypes).astype(np.float32)  # (3, 512)
        log.info(
            "CLIP text prototypes built from %d captions",
            sum(len(v) for v in self.prompts.values()),
        )

    # ------------------------------------------------------------- inference

    def classify(self, image: Image.Image) -> dict:
        """Return {'p_dry', 'p_damp', 'p_wet'} for one PIL image.

        The processor resizes to 224x224 and applies CLIP's normalisation, so
        input images can be any size - we downscale first anyway to keep CPU
        cost predictable.
        """
        if not self.available:
            raise RuntimeError(self.load_error or "CLIP is not loaded")

        torch = self._torch
        image = image.convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)          # (1, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        img_vec = feats.cpu().numpy()[0].astype(np.float32)

        # Cosine similarity to each prototype (both sides are unit vectors, so a
        # dot product *is* the cosine), scaled by CLIP's learned temperature and
        # turned into probabilities with a softmax.
        sims = self._text_prototypes @ img_vec                        # (3,)
        logits = sims * self._logit_scale
        logits = logits - logits.max()                                # numerical safety
        exp = np.exp(logits)
        probs = exp / exp.sum()

        return {
            "p_dry": float(probs[0]),
            "p_damp": float(probs[1]),
            "p_wet": float(probs[2]),
            "similarities": {c: float(s) for c, s in zip(CATEGORY_ORDER, sims)},
        }

    # ------------------------------------------------------------------ misc

    def status(self) -> dict:
        """Honest status for GET /api/health - never a hardcoded OK."""
        return {
            "model_id": self.model_id,
            "loaded": self.available,
            "error": self.load_error,
            "load_seconds": self.load_seconds,
            "prompt_counts": {k: len(v) for k, v in self.prompts.items()},
            "total_prompts": sum(len(v) for v in self.prompts.values()),
        }


def clip_wetness_from_probs(p_dry: float, p_damp: float, p_wet: float) -> float:
    """Collapse three probabilities into one 0-1 wetness number.

    Damp counts half, wet counts full, dry counts nothing:

        clip_wetness = P(damp) * 0.5 + P(wet) * 1.0

    So a confident "damp" lands near 0.5 and a confident "wet" lands near 1.0.
    """
    return float(np.clip(p_damp * 0.5 + p_wet * 1.0, 0.0, 1.0))
