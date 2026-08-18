"""Turn a timed drying capture into the validation figure.

    python scripts/analyse_drying.py                 # uses data/samples_real
    python scripts/analyse_drying.py --source hf

Runs the whole pipeline over a captured sequence and produces two things:

  1. A table of every reading against real elapsed time.
  2. drying_curve.png - the measured curve, the fitted exponential, the DRY
     threshold, and where the forecast said it would cross.

That chart is the single strongest slide this project can have. It is the
difference between "we believe drying is a direction" and "we wet a real road,
watched it dry, and predicted the crossover four minutes before it happened".

Drawn with Pillow rather than matplotlib so it needs no extra dependency and
comes out in the same visual language as the dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image, ImageDraw, ImageFont          # noqa: E402

from app.pipeline import cv_features                 # noqa: E402
from app.pipeline.clip_classifier import (           # noqa: E402
    ClipTrackClassifier, clip_wetness_from_probs,
)
from app.pipeline.forecast import forecast_drying, predict_at   # noqa: E402
from app.pipeline.orchestrator import decode_image, resize_for_clip   # noqa: E402
from app.pipeline.smoothing import DRY_MAX, DAMP_MAX, band_for, ewma  # noqa: E402
from app.sample_source import resolve_sample_dir     # noqa: E402

# Same palette as the dashboard.
CANVAS = (250, 249, 247)
INK = (18, 16, 14)
INK_MUTED = (107, 102, 96)
INK_FAINT = (156, 150, 142)
HAIRLINE = (230, 227, 222)
DRY_C = (31, 122, 84)
WET_C = (192, 39, 29)
ACCENT = (225, 6, 0)

W, H = 1500, 860
PAD_L, PAD_R, PAD_T, PAD_B = 96, 210, 108, 96


def _font(size, bold=False):
    for name in (("segoeuisb.ttf", "seguisb.ttf") if bold else ("segoeui.ttf",)):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_times(folder: Path, images: list[Path]) -> list[float]:
    """Real elapsed seconds per frame, from the manifest when present."""
    manifest = folder / "manifest.json"
    stamps = {}
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            stamps = {f["file"]: float(f["elapsed_s"])
                      for f in data.get("frames", []) if "elapsed_s" in f}
        except Exception:
            stamps = {}
    if stamps:
        return [stamps.get(p.name, float(i * 60)) for i, p in enumerate(images)]
    print("No capture manifest with timestamps - assuming one frame per minute.")
    return [float(i * 60) for i in range(len(images))]


def analyse(folder: Path, images: list[Path]) -> dict:
    print("Loading CLIP ...", flush=True)
    clf = ClipTrackClassifier(BACKEND_DIR / "config" / "prompts.json")
    ok = clf.load()
    print(f"CLIP loaded: {ok}\n")

    times = load_times(folder, images)
    raws, smoothed = [], []

    header = f"{'t (min)':>9}{'CLIP':>8}{'optics':>8}{'raw':>8}{'smoothed':>10}{'state':>8}   forecast"
    print(header)
    print("-" * (len(header) + 12))

    forecasts = []
    for i, (path, t) in enumerate(zip(images, times)):
        pil, bgr = decode_image(path.read_bytes())
        if ok:
            probs = clf.classify(resize_for_clip(pil))
            cw = clip_wetness_from_probs(probs["p_dry"], probs["p_damp"], probs["p_wet"])
        else:
            cw = 0.0
        phys = cv_features.analyze_surface(bgr)["physical_wetness"]
        raw = cv_features.fuse(cw, phys) if ok else phys

        raws.append(raw)
        smoothed.append(ewma(raws))

        fc = forecast_drying(times[:i + 1], smoothed)
        forecasts.append(fc)
        note = fc["sentence"] if fc else ""
        print(f"{t/60:>9.1f}{cw:>8.3f}{phys:>8.3f}{raw:>8.3f}"
              f"{smoothed[-1]:>10.3f}{band_for(smoothed[-1]):>8}   {note}")

    # Once the road is dry there is nothing left to forecast, so the fit over
    # the whole sequence is correctly None. For the chart we want the fit that
    # was actually used to make the call - the last one that had something to
    # say - because that is the claim being validated.
    final = forecast_drying(times, smoothed)
    if final is None:
        final = next((f for f in reversed(forecasts) if f), None)

    return {"times": times, "smoothed": smoothed, "raws": raws,
            "forecasts": forecasts, "final": final}


def crossing_minute(times, smoothed) -> float | None:
    """When the measured curve actually first crossed into DRY."""
    for t, w in zip(times, smoothed):
        if w < DRY_MAX:
            return t / 60.0
    return None


def draw_chart(result: dict, out_path: Path, observed_dry: float | None) -> None:
    times = [t / 60.0 for t in result["times"]]
    sm, raws = result["smoothed"], result["raws"]
    fit = result["final"]

    img = Image.new("RGB", (W, H), CANVAS)
    d = ImageDraw.Draw(img)

    f_title, f_lbl = _font(34, True), _font(15)
    f_micro, f_num = _font(13, True), _font(14)

    x0, x1 = PAD_L, W - PAD_R
    y0, y1 = PAD_T, H - PAD_B
    t_max = max(times) * 1.06
    def px(t): return x0 + (t / t_max) * (x1 - x0)
    def py(w): return y1 - min(max(w, 0), 1.0) * (y1 - y0)

    d.text((PAD_L, 38), "A real surface, drying", font=f_title, fill=INK)
    sub = (f"{len(times)} readings over {max(times):.0f} minutes"
           + (f"   ·   time constant {fit['tau_minutes']:.1f} min   ·   "
              f"R² {fit['r_squared']:.3f}" if fit else ""))
    d.text((PAD_L, 80), sub, font=f_lbl, fill=INK_MUTED)

    # band boundaries
    for level, name, col in ((DAMP_MAX, "WET", WET_C), (DRY_MAX, "DRY", DRY_C)):
        y = py(level)
        for xx in range(x0, x1, 9):
            d.line([(xx, y), (xx + 4, y)], fill=HAIRLINE, width=1)
        d.text((x1 + 14, y - 8), f"{name} below {level:.2f}", font=f_micro, fill=INK_FAINT)

    # axes
    d.line([(x0, y1), (x1, y1)], fill=HAIRLINE, width=1)
    d.line([(x0, y0), (x0, y1)], fill=HAIRLINE, width=1)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        d.text((x0 - 46, py(w) - 9), f"{w:.2f}", font=f_num, fill=INK_FAINT)
    step = 5 if t_max <= 45 else 10
    for m in range(0, int(t_max) + 1, step):
        d.text((px(m) - 10, y1 + 14), f"{m}", font=f_num, fill=INK_FAINT)
    d.text(((x0 + x1) // 2 - 60, y1 + 44), "minutes since the water went down",
           font=f_lbl, fill=INK_MUTED)

    # fitted exponential, extended past the data
    if fit:
        base = fit["baseline"]
        tau_s = fit["tau_minutes"] * 60.0
        amp = (sm[0] - base) if sm[0] > base else 0.0
        if amp > 0:
            pts = []
            for k in range(240):
                t_min = (k / 239.0) * t_max
                pts.append((px(t_min), py(predict_at(base, amp, tau_s, t_min * 60))))
            for i in range(0, len(pts) - 1, 2):     # dashed
                d.line([pts[i], pts[i + 1]], fill=DRY_C, width=2)

    # raw + smoothed
    d.line([(px(t), py(w)) for t, w in zip(times, raws)], fill=INK_FAINT, width=1)
    d.line([(px(t), py(w)) for t, w in zip(times, sm)], fill=INK, width=3)
    for t, w in zip(times, sm):
        d.ellipse([px(t) - 3, py(w) - 3, px(t) + 3, py(w) + 3], fill=CANVAS, outline=INK)

    # the moment it actually crossed
    actual = crossing_minute(result["times"], sm)
    if actual is not None:
        x = px(actual)
        d.line([(x, y0), (x, y1)], fill=DRY_C, width=2)
        d.text((x + 10, y0 + 6), f"crossed DRY at {actual:.0f} min",
               font=f_micro, fill=DRY_C)

    if observed_dry is not None:
        x = px(observed_dry)
        for yy in range(y0, y1, 10):
            d.line([(x, yy), (x, yy + 5)], fill=ACCENT, width=2)
        d.text((x + 10, y0 + 30), f"looked dry to the eye at {observed_dry:.0f} min",
               font=f_micro, fill=ACCENT)

    # the earliest useful forecast, called out
    for t_s, fc in zip(result["times"], result["forecasts"]):
        if fc and fc.get("dry_at_minutes"):
            called_at, predicted = t_s / 60.0, fc["dry_at_minutes"]
            lead = predicted - called_at
            if lead > 0.5:
                d.text((PAD_L, y1 + 66),
                       f"At minute {called_at:.0f} the forecast said dry at "
                       f"{predicted:.0f} min - {lead:.0f} minutes before it happened"
                       + (f", actual {actual:.0f} min." if actual else "."),
                       font=f_lbl, fill=INK)
                break

    key_y = y0 + 4
    for label, col, wdt in (("smoothed", INK, 3), ("raw", INK_FAINT, 1),
                            ("fitted decay", DRY_C, 2)):
        d.line([(x1 + 14, key_y + 7), (x1 + 44, key_y + 7)], fill=col, width=wdt)
        d.text((x1 + 52, key_y), label, font=f_micro, fill=INK_MUTED)
        key_y += 26

    img.save(out_path, quality=95)
    print(f"\nChart -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real",
                    help="bundled | hf | real   (default: real)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    folder, images, _ = resolve_sample_dir(args.source)
    if len(images) < 5:
        print(f"Only {len(images)} frames in {folder}. Capture a sequence first:")
        print("    python scripts/import_drying_experiment.py <folder-of-photos>")
        return 1

    observed = None
    manifest = folder / "manifest.json"
    if manifest.is_file():
        try:
            observed = json.loads(manifest.read_text(encoding="utf-8")).get(
                "observed_dry_at_min")
        except Exception:
            pass

    print(f"\nSource: {folder}  ({len(images)} frames)\n")
    result = analyse(folder, images)

    fit = result["final"]
    print("\n" + "=" * 62)
    if fit:
        print(f"Time constant (tau)        {fit['tau_minutes']:.1f} minutes")
        print(f"Fitted over                {fit['points']} readings / "
              f"{fit['observed_minutes']:.0f} min")
        print(f"Fit quality (R²)           {fit['r_squared']:.4f}")
        print(f"Predicted DRY crossing     {fit.get('dry_at_minutes')} min")
    else:
        print("No forecast: the capture does not support one. Needs a real")
        print("timeline of 2+ minutes, a falling curve, and a clean fit.")
    actual = crossing_minute(result["times"], result["smoothed"])
    if actual is not None:
        print(f"Actually crossed DRY at    {actual:.1f} min")
    if observed is not None:
        print(f"Looked dry to the eye at   {observed:.1f} min")
    print("=" * 62)

    out = Path(args.out) if args.out else BACKEND_DIR / "data" / "drying_curve.png"
    draw_chart(result, out, observed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
