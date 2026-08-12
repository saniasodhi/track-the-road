"""Generate the bundled sample frames: one race track, drying out.

    python scripts/generate_samples.py

These frames are SYNTHETIC. They are rendered here rather than downloaded so the
repository is self-contained, so the demo is byte-for-byte identical on every
machine, and so there is no licensing question about somebody else's photographs.

To use real footage instead, drop 4 or more photos into backend/data/samples_real/
named so alphabetical order is time order (01.jpg, 02.jpg, ...). The demo picks
that folder up automatically.

How the render works
--------------------
It is one fixed camera looking down a track, so every frame shows the same piece
of asphalt with the same stones and the same puddles - only the water changes.
That matters: a drying sequence has to look like the same place over time, not
sixteen unrelated pictures.

Geometry is a real pinhole projection of a flat plane. For a camera at height h
looking at a flat road, a point at depth d projects to screen row

    y = horizon + k / d

so screen row maps back to depth as d = k / (y - horizon). Texture is sampled in
*world* coordinates through that mapping, which is why the asphalt grain gets
finer towards the horizon by itself instead of being faked.

The wet look is built from the four things water actually does, which are exactly
the four things pipeline step 2 measures:

  * darkens the surface (water fills the pores, light does not come back out)
  * kills saturation (a neutral sky reflection sits on top of the surface colour)
  * smooths the texture (water fills the gaps between stones)
  * adds specular highlights (a mirror instead of a diffuse surface), strongest
    at grazing angles near the horizon, which is genuine Fresnel behaviour

Everything is seeded, so re-running this script reproduces the same 16 images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BACKEND_DIR / "data" / "samples"

# ---------------------------------------------------------------- parameters

W, H = 960, 640
HORIZON = int(H * 0.36)
VP_X = W * 0.51                 # vanishing point, nudged off centre so it reads as a real photo
ROAD_K = 0.74 * W               # road half-width at the bottom of the frame
TEX_H, TEX_W = 2048, 1024

SEED_TEXTURE = 7                # asphalt grain
SEED_PUDDLE = 11                # where the water sits
SEED_SPARKLE = 23               # specular glints
SEED_FINE = 41                  # resolved chippings close to the camera

# The story: heavy standing water, a plateau while the rain stops, then a fast dry-out.
WETNESS_SCHEDULE = [
    0.99, 0.97, 0.95, 0.92,
    0.88, 0.86, 0.85, 0.83,     # plateau - rain has stopped, water has not gone yet
    0.74, 0.63, 0.52, 0.41,
    0.30, 0.20, 0.11, 0.04,
]

# Colours (linear-ish RGB, 0-1), interpolated between the dry and soaked ends.
SKY_TOP_DRY, SKY_TOP_WET = (0.55, 0.66, 0.80), (0.30, 0.32, 0.36)
SKY_HOR_DRY, SKY_HOR_WET = (0.82, 0.85, 0.88), (0.52, 0.53, 0.55)
ASPHALT_DRY, ASPHALT_WET = (0.480, 0.466, 0.440), (0.115, 0.121, 0.134)
GRASS_DRY, GRASS_WET = (0.38, 0.44, 0.22), (0.155, 0.175, 0.135)


def lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ------------------------------------------------------------------ textures

def multiscale_noise(rng, h, w, scales=((0.8, 1.0), (2.2, 0.8), (6.0, 0.55), (16.0, 0.35))):
    """Zero-mean, unit-variance noise built from several blur scales."""
    out = np.zeros((h, w), np.float32)
    for sigma, amp in scales:
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= n.std() + 1e-6
        out += n * amp
    out -= out.mean()
    out /= out.std() + 1e-6
    return out


def build_asphalt_texture():
    """Grainy tarmac in world space: fine grit plus scattered lighter aggregate."""
    rng = np.random.default_rng(SEED_TEXTURE)
    grain = multiscale_noise(rng, TEX_H, TEX_W)

    # Individual chippings: small bright specks with a touch of shadow underneath.
    stones = rng.random((TEX_H, TEX_W)).astype(np.float32)
    stones = (stones > 0.9975).astype(np.float32)
    stones = cv2.GaussianBlur(stones, (0, 0), 1.1) * 22.0
    shadow = np.roll(stones, 2, axis=0) * 0.5

    tex = grain * 0.85 + stones - shadow
    tex -= tex.mean()
    tex /= tex.std() + 1e-6
    return tex


def build_puddle_field():
    """Low-frequency field. High values are the low points where water collects."""
    rng = np.random.default_rng(SEED_PUDDLE)
    field = multiscale_noise(rng, TEX_H, TEX_W, scales=((18.0, 1.0), (46.0, 0.9), (90.0, 0.7)))
    return field


def build_sparkle_field():
    """High-frequency field used to break puddle reflections into glints."""
    rng = np.random.default_rng(SEED_SPARKLE)
    return multiscale_noise(rng, TEX_H, TEX_W, scales=((0.7, 1.0), (1.8, 0.6)))


def build_fine_grain():
    """Screen-space grain: the individual chippings you can actually resolve.

    This layer is deliberately NOT perspective-magnified. Close to the camera a
    real photograph resolves single stones; far away they blur into a flat tone.
    So the layer is faded out with distance, and faded out with water, because a
    film of water is exactly what stops you seeing the grain.
    """
    rng = np.random.default_rng(SEED_FINE)
    return multiscale_noise(rng, H, W, scales=((0.55, 1.0), (1.2, 0.55), (2.6, 0.3)))


# -------------------------------------------------------------- perspective

def build_perspective_maps():
    """Screen pixel -> world texture coordinate, plus depth and road geometry."""
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]

    below = np.maximum(ys - HORIZON, 1e-3)
    # depth: 1.0 at the bottom of the frame, growing towards the horizon
    depth = (H - HORIZON) / below
    depth = np.clip(depth, 1.0, 45.0)
    depth = np.broadcast_to(depth, (H, W)).astype(np.float32)

    # Half-width of the road on screen shrinks linearly with 1/depth.
    half_width = (ROAD_K / depth).astype(np.float32)
    lateral = (xs - VP_X).astype(np.float32)          # signed pixels from the centre line
    lateral = np.broadcast_to(lateral, (H, W)).astype(np.float32)
    # position across the road, -1 = left edge, +1 = right edge
    across = lateral / np.maximum(half_width, 1e-3)

    # World coordinates for texture lookup.
    map_x = (across * 420.0 + TEX_W * 0.5) % TEX_W
    map_y = (depth * 110.0) % TEX_H

    return map_x.astype(np.float32), map_y.astype(np.float32), depth, across


def sample(texture, map_x, map_y):
    return cv2.remap(texture, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


# ------------------------------------------------------------------- render

def render_frame(w: float, maps, textures) -> np.ndarray:
    """Render one frame at wetness `w` (0 = bone dry, 1 = standing water)."""
    map_x, map_y, depth, across = maps
    asphalt_tex, puddle_tex, sparkle_tex, fine_tex = textures

    inv_depth = (1.0 / depth).astype(np.float32)      # 1 at the bottom, ~0 at the horizon
    road_mask = (np.abs(across) <= 1.0) & (np.arange(H)[:, None] > HORIZON)
    road_mask = road_mask.astype(np.float32)
    road_mask = cv2.GaussianBlur(road_mask, (0, 0), 0.8)

    img = np.zeros((H, W, 3), np.float32)

    # ---- sky -------------------------------------------------------------
    sky_top = np.array(lerp(SKY_TOP_DRY, SKY_TOP_WET, w), np.float32)
    sky_hor = np.array(lerp(SKY_HOR_DRY, SKY_HOR_WET, w), np.float32)
    t = np.clip(np.arange(H, dtype=np.float32) / max(HORIZON, 1), 0, 1)[:, None, None]
    img[:] = sky_top[None, None, :] * (1 - t) + sky_hor[None, None, :] * t

    # ---- distant scenery on the horizon ----------------------------------
    scenery = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(31)
    x = 0
    while x < W:
        width = int(rng.integers(30, 110))
        height = int(rng.integers(6, 26))
        scenery[max(0, HORIZON - height):HORIZON, x:x + width] = 1.0
        x += width + int(rng.integers(0, 24))
    scenery = cv2.GaussianBlur(scenery, (0, 0), 1.6)[..., None]
    tree = np.array(lerp((0.22, 0.26, 0.20), (0.19, 0.20, 0.21), w), np.float32)
    img = img * (1 - scenery) + tree[None, None, :] * scenery

    # ---- grass verges ----------------------------------------------------
    verge_noise = sample(asphalt_tex, (map_x * 1.7) % TEX_W, (map_y * 1.7) % TEX_H)
    grass = np.array(lerp(GRASS_DRY, GRASS_WET, w), np.float32)[None, None, :]
    grass = grass * (1.0 + 0.16 * verge_noise[..., None])
    ground = (np.arange(H)[:, None] > HORIZON).astype(np.float32)[..., None]
    img = img * (1 - ground) + grass * ground

    # ---- asphalt ---------------------------------------------------------
    grain = sample(asphalt_tex, map_x, map_y)
    # Water fills the gaps between the stones, so grain amplitude falls with wetness.
    grain_amp = 0.115 * (1.0 - 0.80 * w)
    base = np.array(lerp(ASPHALT_DRY, ASPHALT_WET, w), np.float32)[None, None, :]
    surface = base * (1.0 + grain_amp * grain[..., None])

    # A slightly darker, polished racing line down the middle.
    racing_line = np.exp(-((across - 0.05) ** 2) / (2 * 0.30 ** 2)).astype(np.float32)
    surface *= (1.0 - 0.09 * racing_line[..., None])

    # ---- puddles ---------------------------------------------------------
    puddle_field = sample(puddle_tex, map_x, map_y)
    road_values = puddle_field[road_mask > 0.5]
    coverage = float(np.clip(0.40 * (w ** 1.6), 0.0, 0.40))
    if coverage > 0.002 and road_values.size:
        threshold = float(np.quantile(road_values, 1.0 - coverage))
        puddle = smoothstep((puddle_field - threshold) / 0.55)
    else:
        puddle = np.zeros_like(puddle_field)
    puddle *= road_mask

    # Water reflects the sky. Reflectivity rises steeply at grazing angles
    # (Fresnel), so distant water is much brighter than water at your feet.
    grazing = np.clip(1.0 - inv_depth, 0.0, 1.0) ** 1.5
    reflect_colour = np.array(lerp(SKY_HOR_DRY, SKY_HOR_WET, w), np.float32)[None, None, :]
    reflect_strength = (puddle * (0.22 + 0.55 * grazing))[..., None]
    surface = surface * (1 - reflect_strength) + reflect_colour * reflect_strength

    # Specular glints: where the reflected sky breaks up on a rippled surface.
    # Kept sparse on purpose - a wet road is mostly dark with a few bright
    # highlights, which is exactly the signature step 2 looks for. The layer is
    # built here but composited later, after the softening pass, because a
    # specular highlight is sharper than the surface it sits on.
    sparkle = sample(sparkle_tex, map_x, map_y * 0.5)
    glint_layer = np.zeros((H, W), np.float32)
    glint_area = 0.13 * (w ** 1.3)
    if glint_area > 0.001:
        vals = sparkle[(puddle > 0.35)]
        if vals.size > 32:
            q = float(np.quantile(vals, 1.0 - min(0.24, glint_area)))
            glint = smoothstep((sparkle - q) / 0.16) * (puddle > 0.35)
            # Reflections stretch along the viewing direction, so smear vertically.
            glint = cv2.GaussianBlur(glint.astype(np.float32), (3, 7), 0)
            glint_layer = glint * (0.85 + 0.75 * grazing) * (w ** 1.1)

    # A thin water film also makes the whole surface slightly mirror-like.
    sheen = (0.26 * w * grazing * road_mask)[..., None]
    surface = surface * (1 - sheen) + reflect_colour * sheen

    img = img * (1 - road_mask[..., None]) + surface * road_mask[..., None]

    # ---- track markings --------------------------------------------------
    line_w = np.clip(0.055 * inv_depth + 0.010, 0.012, 0.075)
    for side in (-1.0, 1.0):
        edge = np.exp(-((np.abs(across) - 0.93) ** 2) / (2 * (line_w * 0.55) ** 2))
        edge = edge * (np.sign(across) == side) * road_mask
        paint = np.array([0.80, 0.80, 0.78], np.float32) * (1.0 - 0.34 * w)
        a = np.clip(edge, 0, 1)[..., None] * 0.95
        img = img * (1 - a) + paint[None, None, :] * a

    # Red / white kerbs just outside the white lines, striped in world space.
    stripe = ((map_y / 46.0).astype(np.int32) % 2 == 0).astype(np.float32)
    kerb_band = np.exp(-((np.abs(across) - 1.055) ** 2) / (2 * 0.045 ** 2))
    kerb_band *= (np.arange(H)[:, None] > HORIZON + 4)
    kerb_col = (np.array([0.78, 0.16, 0.13], np.float32)[None, None, :] * stripe[..., None]
                + np.array([0.79, 0.79, 0.77], np.float32)[None, None, :] * (1 - stripe[..., None]))
    kerb_col *= (1.0 - 0.30 * w)
    a = np.clip(kerb_band, 0, 1)[..., None] * 0.92
    img = img * (1 - a) + kerb_col * a

    # ---- atmosphere ------------------------------------------------------
    # Spray and haze hanging over the circuit while it is wet; plus ordinary
    # aerial perspective washing out the distance.
    haze_colour = np.array(lerp((0.80, 0.83, 0.86), (0.48, 0.49, 0.51), w), np.float32)
    haze = np.clip((1.0 - inv_depth) ** 2.1, 0, 1) * (0.24 + 0.26 * w)
    haze = haze * (np.arange(H)[:, None] > HORIZON)
    img = img * (1 - haze[..., None]) + haze_colour[None, None, :] * haze[..., None]

    # ---- optics ----------------------------------------------------------
    # Wet surfaces genuinely read softer: the fine grain is under water.
    if w > 0.05:
        blurred = cv2.GaussianBlur(img, (0, 0), 0.5 + 1.2 * w)
        mix = float(np.clip(w * 0.80, 0, 0.85))
        img = img * (1 - mix) + blurred * mix

    # ---- specular highlights --------------------------------------------
    # Composited after the softening pass and pushed hard enough to genuinely
    # clip the sensor. Blown-out pixels are the single clearest optical
    # fingerprint of standing water, and counting them is step 2's shine metric.
    if glint_layer.max() > 1e-4:
        img += (glint_layer * 1.35)[..., None]

    # ---- resolved surface grain -----------------------------------------
    # Added after the softening pass, because this is the detail a camera can
    # still resolve. It is strongest close to the lens and it disappears under
    # water, which is precisely the texture signal step 2 measures.
    grain_visibility = (inv_depth ** 0.55) * road_mask * (1.0 - 0.90 * w)
    img *= (1.0 + 0.30 * (fine_tex * grain_visibility)[..., None])

    # Slight vignette and a fixed grain, so it reads as a photograph.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1.0 - 0.16 * np.clip(r - 0.55, 0, None) ** 1.4)[..., None]

    sensor = np.random.default_rng(99).standard_normal((H, W, 1)).astype(np.float32)
    img += sensor * 0.0045

    # Dry tarmac keeps a little warmth; wet tarmac goes cold and neutral.
    warm = np.array([1.0 + 0.045 * (1 - w), 1.0, 1.0 - 0.035 * (1 - w)], np.float32)
    img *= warm[None, None, :]

    img = np.clip(img, 0.0, 1.0)
    # Gentle S-curve, the way a camera would apply it.
    img = np.clip(1.06 * (img ** 0.94) - 0.02, 0.0, 1.0)

    bgr = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("frame_*.jpg"):
        old.unlink()

    maps = build_perspective_maps()
    textures = (build_asphalt_texture(), build_puddle_field(),
                build_sparkle_field(), build_fine_grain())

    manifest = []
    for i, w in enumerate(WETNESS_SCHEDULE):
        bgr = render_frame(w, maps, textures)
        path = OUT_DIR / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        manifest.append({"file": path.name, "render_wetness": round(w, 3)})
        print(f"  frame_{i:02d}.jpg   render wetness {w:.2f}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "generator": "scripts/generate_samples.py",
                "note": "Ground-truth wetness used to RENDER each frame. The pipeline "
                        "never sees this - it is only here so you can sanity-check the "
                        "estimates against what was actually drawn.",
                "frames": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {len(manifest)} frames to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
