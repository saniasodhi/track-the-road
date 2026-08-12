"""PIPELINE STEP 2 - Measure the physics of the image, without any model.

Why this step exists
--------------------
If the whole product were "send the picture to CLIP", it would be one call to one
off-the-shelf tool. So we add a second, completely independent opinion, built
from classical computer vision (OpenCV + NumPy) and basic optics. It has no
training data and no weights - it just measures the image.

Two independent methods that agree is a much stronger claim than one method that
is confident. When they disagree, that itself is information, and both numbers
are stored and shown in the UI.

What actually changes when a road gets wet
------------------------------------------
1. SHINE. Dry asphalt scatters light in all directions (it is a diffuse
   surface). A film of water turns it into a mirror, so it throws back sharp
   specular highlights. We count the fraction of near-blown-out bright pixels.

2. DARKNESS. Water fills the pores in the surface and light that enters mostly
   does not come back out, so wet tarmac reads much darker. We measure mean
   brightness (the V channel in HSV).

3. COLOUR. Wet surfaces also lose colour - the reflected light is mostly a
   neutral sky reflection sitting on top of the surface colour. We measure mean
   saturation (the S channel).

4. TEXTURE. Dry asphalt is visibly grainy: thousands of small stones each with
   their own shadow. Water fills the gaps and smooths the surface out. We
   measure this with the variance of the Laplacian, the standard sharpness
   metric - high variance means lots of fine detail, low variance means smooth.

Each measurement is mapped onto 0-1 and the four are blended into one
`physical_wetness` score.

Region of interest
------------------
For a forward-facing camera the top of the frame is usually sky, which would
wreck the brightness and shine statistics. So all four measurements are taken on
the bottom ROI_FRACTION of the image. For a close-up of tarmac this changes
nothing; for a track shot with sky in it, it matters a lot.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Calibration constants.
#
# These are the "dry end" and "wet end" anchors for each measurement. Anything
# at or past the wet anchor scores 1.0, anything at or past the dry anchor
# scores 0.0, and everything in between is linear. They are tuned against the
# bundled sample frames; nudge them if you swap in your own footage.
# ---------------------------------------------------------------------------

ROI_FRACTION = 0.55        # use the bottom 55% of the frame (the road)
WORK_MAX_DIM = 512         # downscale before measuring, for consistent speed

SPECULAR_THRESHOLD = 228   # a pixel this bright (0-255 V) counts as a highlight
SPECULAR_FULL = 0.018      # 1.8% of the road blown out == maximum shine

BRIGHTNESS_DRY = 0.56      # mean V of a dry surface
BRIGHTNESS_WET = 0.20      # mean V of a soaked surface

SATURATION_DRY = 0.21      # mean S of a dry surface
SATURATION_WET = 0.08      # mean S of a soaked surface

TEXTURE_DRY_LOG = 2.35     # log10(Laplacian variance) for grainy dry asphalt (~220)
TEXTURE_WET_LOG = 1.55     # log10(Laplacian variance) for smoothed-over wet asphalt (~35)

# How much each measurement contributes to the final physical score.
WEIGHTS = {
    "shine": 0.32,
    "darkness": 0.30,
    "desaturation": 0.12,
    "smoothness": 0.26,
}


def _ramp(value: float, at_zero: float, at_one: float) -> float:
    """Linearly map `value` onto 0-1, where at_zero -> 0.0 and at_one -> 1.0.

    Works in either direction (at_one may be lower than at_zero, which is the
    case for brightness: darker means wetter).
    """
    span = at_one - at_zero
    if abs(span) < 1e-9:
        return 0.0
    return float(np.clip((value - at_zero) / span, 0.0, 1.0))


def _prepare(bgr: np.ndarray) -> np.ndarray:
    """Downscale, then crop to the road region of interest."""
    h, w = bgr.shape[:2]
    scale = WORK_MAX_DIM / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    top = int(h * (1.0 - ROI_FRACTION))
    roi = bgr[top:, :]
    # Guard against degenerate crops on very small images.
    return roi if roi.size else bgr


def analyze_surface(bgr: np.ndarray) -> dict:
    """Measure one frame. Input is an OpenCV BGR array. Returns raw stats + scores.

    Crops to the road region of interest first, then measures.
    """
    return measure_region(_prepare(bgr))


def measure_region(roi: np.ndarray) -> dict:
    """Run the four measurements over exactly the pixels given, no cropping.

    Split out from analyze_surface so the zone grid (zones.py) can measure a
    single cell of the road with identical maths. Whatever calls this is
    responsible for handing over road pixels and nothing else.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --- 1. Shine: how much of the surface is throwing back a specular highlight?
    specular_fraction = float((v >= SPECULAR_THRESHOLD).mean())

    # --- 2. Darkness: mean brightness, normalised to 0-1.
    mean_v = float(v.mean() / 255.0)

    # --- 3. Colour: mean saturation, normalised to 0-1.
    mean_s = float(s.mean() / 255.0)

    # --- 4. Texture: variance of the Laplacian. Blur very slightly first so we
    #        measure surface grain rather than sensor noise or JPEG artefacts.
    #
    #        Important subtlety: the edges of a specular highlight are themselves
    #        very high-frequency, so a wet road full of glints can score as
    #        "textured" for entirely the wrong reason - the two signals fight
    #        each other. So highlights (and a small margin around them) are
    #        excluded, and the variance is taken over the remaining surface only.
    #        What is left is genuine surface grain.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    highlight = (v >= SPECULAR_THRESHOLD - 28).astype(np.uint8)
    highlight = cv2.dilate(highlight, np.ones((5, 5), np.uint8), iterations=1)
    surface_pixels = laplacian[highlight == 0]
    # Only trust the masked measurement if a decent amount of surface survives.
    if surface_pixels.size > 0.15 * laplacian.size:
        laplacian_var = float(surface_pixels.var())
    else:
        laplacian_var = float(laplacian.var())

    # Map each measurement onto a 0 (bone dry) to 1 (soaked) score.
    shine = _ramp(specular_fraction, 0.0, SPECULAR_FULL)
    darkness = _ramp(mean_v, BRIGHTNESS_DRY, BRIGHTNESS_WET)
    desaturation = _ramp(mean_s, SATURATION_DRY, SATURATION_WET)
    smoothness = _ramp(np.log10(laplacian_var + 1.0), TEXTURE_DRY_LOG, TEXTURE_WET_LOG)

    subscores = {
        "shine": shine,
        "darkness": darkness,
        "desaturation": desaturation,
        "smoothness": smoothness,
    }
    physical = sum(subscores[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "physical_wetness": float(np.clip(physical, 0.0, 1.0)),
        "subscores": {k: round(float(val), 4) for k, val in subscores.items()},
        "raw": {
            "specular_fraction": round(specular_fraction, 5),
            "mean_brightness": round(mean_v, 4),
            "mean_saturation": round(mean_s, 4),
            "laplacian_variance": round(laplacian_var, 2),
        },
    }


def fuse(clip_wetness: float, physical_wetness: float,
         clip_weight: float = 0.65) -> float:
    """Combine the two independent opinions into one number.

        wetness_score = 0.65 * clip_wetness + 0.35 * physical_wetness

    CLIP gets the larger share because it understands *scene context* - it knows
    the difference between a wet road and a dark road at dusk. The physical
    score gets a real vote because it is grounded in optics and cannot be fooled
    by an unusual camera angle or an out-of-distribution scene.
    """
    cw = float(np.clip(clip_wetness, 0.0, 1.0))
    pw = float(np.clip(physical_wetness, 0.0, 1.0))
    return float(np.clip(clip_weight * cw + (1.0 - clip_weight) * pw, 0.0, 1.0))
