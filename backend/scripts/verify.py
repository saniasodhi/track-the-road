"""Assert that the pipeline still behaves the way the demo claims it does.

    python scripts/verify.py            # everything (loads CLIP, ~1 min)
    python scripts/verify.py --quick    # pure functions only, instant

This is not a unit-test suite for its own sake. It is a guard on the things
that are actually said out loud to a judge:

  * the bundled sequence really does go WET -> DRYING -> DRY
  * running it twice really does give identical numbers
  * a soaked daytime road is NOT mistaken for night
  * a forecast is refused unless the data supports one
  * a hazard detector really does separate wet from dry

Any of those can be broken by a one-character change to a constant, and none of
them would raise an exception - the app would just quietly start lying. So they
are checked here rather than trusted.

Run it after changing anything in app/pipeline/, and once more before
presenting. preflight.py answers "can this machine run it"; this answers "does
it still do the right thing".

Exit code 0 if everything passed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np                                            # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((bool(condition), name, detail))
    mark = "pass" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * max(46, len(title)))


# ---------------------------------------------------------------- pure logic

def verify_pure() -> None:
    from app.pipeline.smoothing import (
        band_for, ewma, stabilise_band, DRY_MAX, DAMP_MAX, CONFIRM_FRAMES,
    )
    from app.pipeline import trend as tr
    from app.pipeline import zones as zn
    from app.pipeline.clip_classifier import clip_wetness_from_probs
    from app.pipeline.forecast import forecast_drying
    from app.pipeline import cv_features as cv

    section("Bands and smoothing")
    check("band boundaries", band_for(0.0) == "DRY" and band_for(0.24) == "DRY"
          and band_for(DRY_MAX) == "DAMP" and band_for(0.54) == "DAMP"
          and band_for(DAMP_MAX) == "WET" and band_for(1.0) == "WET")
    check("EWMA of a constant is that constant", abs(ewma([0.4] * 5) - 0.4) < 1e-9)
    check("EWMA leans on the newest frame",
          ewma([0.0, 0.0, 0.0, 0.0, 1.0]) > 0.4,
          f"got {ewma([0.0, 0.0, 0.0, 0.0, 1.0]):.3f}")
    # 0.40 is DAMP, so a WET session dropping to it is a one-band change.
    check("one frame over the line does NOT switch the label",
          stabilise_band([0.60, 0.40], "WET") == "WET")
    check(f"{CONFIRM_FRAMES} frames over the line DOES switch",
          stabilise_band([0.60, 0.40, 0.40], "WET") == "DAMP")
    check("skipping a band still needs confirmation",
          stabilise_band([0.60, 0.20, 0.20], "WET") == "DRY")

    section("Direction")
    falling = list(np.linspace(0.8, 0.2, 10))
    rising = list(np.linspace(0.2, 0.8, 10))
    flat = [0.5] * 10
    check("falling reads IMPROVING",
          tr.classify_trend(tr.fit_slope(falling), 10) == "IMPROVING")
    check("rising reads DETERIORATING",
          tr.classify_trend(tr.fit_slope(rising), 10) == "DETERIORATING")
    check("flat reads STABLE", tr.classify_trend(tr.fit_slope(flat), 10) == "STABLE")
    check("too few points never claims a direction",
          tr.classify_trend(-0.9, 2) == "STABLE")
    check("DRYING is damp plus improving",
          tr.display_state("DAMP", "IMPROVING") == "DRYING"
          and tr.display_state("DAMP", "STABLE") == "DAMP"
          and tr.display_state("WET", "IMPROVING") == "WET")
    check("a STABLE trend forecasts nothing",
          tr.estimate_crossing(0.4, "DAMP", 0.004, "STABLE") == (None, None))
    check("every band/trend pair has a tyre call",
          all((b, t) in tr.RECOMMENDATIONS
              for b in ("DRY", "DAMP", "WET")
              for t in ("IMPROVING", "STABLE", "DETERIORATING")))

    section("Low light")
    daytime_wet = {"crushed_fraction": 0.000, "p95_brightness": 0.569}
    daytime_dry = {"crushed_fraction": 0.000, "p95_brightness": 0.706}
    dusk = {"crushed_fraction": 0.294, "p95_brightness": 0.231}
    night = {"crushed_fraction": 0.914, "p95_brightness": 0.090}
    check("a SOAKED daytime road is not called dark",
          cv.assess_light(daytime_wet)["level"] == "ok",
          "the failure that would break the product")
    check("a dry daytime road is not called dark",
          cv.assess_light(daytime_dry)["level"] == "ok")
    check("dusk degrades", cv.assess_light(dusk)["level"] == "low")
    check("night refuses", cv.assess_light(night)["level"] == "dark")
    for level, expect_hold in (("ok", False), ("low", True), ("dark", True)):
        a = tr.build_advice("WET", "STABLE", 0.0, 0.8, light_level=level)
        held = a["recommendation"] != "FULL WETS"
        check(f"light={level} {'vetoes' if expect_hold else 'keeps'} the tyre call",
              held == expect_hold, a["recommendation"])

    section("Forecast")
    tau, w0, wdry, iv = 11.0, 0.82, 0.09, 90.0
    n = 20
    t = [i * iv for i in range(n)]
    w = [wdry + (w0 - wdry) * math.exp(-x / (tau * 60)) for x in t]
    fit = forecast_drying(t[:12], w[:12])
    check("fits a real decay", fit is not None)
    if fit:
        check("recovers the time constant", abs(fit["tau_minutes"] - tau) < 3.0,
              f"{fit['tau_minutes']:.1f} min vs {tau} true")
        check("fit quality is high", fit["r_squared"] > 0.95, f"R2 {fit['r_squared']}")
        truth = -tau * 60 * math.log((0.25 - wdry) / (w0 - wdry)) / 60
        check("predicts the crossing within 3 min",
              abs(fit["dry_at_minutes"] - truth) < 3.0,
              f"{fit['dry_at_minutes']:.1f} vs {truth:.1f} true")
        check("exposes the curve for drawing",
              all(k in fit for k in ("amplitude", "baseline", "tau_minutes")))
    check("refuses frame numbers as a timeline",
          forecast_drying(list(range(16)), list(np.linspace(0.65, 0.11, 16))) is None,
          "this is why the bundled demo shows no forecast")
    check("refuses a road getting wetter",
          forecast_drying(t[:12], list(np.linspace(0.2, 0.7, 12))) is None)
    check("refuses a flat curve",
          forecast_drying(t[:12], [0.42] * 12) is None)
    check("refuses too few readings",
          forecast_drying([0.0, 90.0, 180.0], [0.8, 0.6, 0.4]) is None)

    section("Geometry and mapping")
    quads = [zn.cell_quad(r, c) for r in range(zn.GRID_ROWS) for c in range(zn.GRID_COLS)]
    check("grid has 9 cells", len(quads) == 9)
    check("every corner is inside the frame",
          all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for q in quads for x, y in q))
    check("the road narrows towards the horizon",
          zn._road_width_at(0.0) < zn._road_width_at(1.0))
    check("wetness mapping: damp counts half, wet counts full",
          abs(clip_wetness_from_probs(0, 1, 0) - 0.5) < 1e-9
          and abs(clip_wetness_from_probs(0, 0, 1) - 1.0) < 1e-9
          and abs(clip_wetness_from_probs(1, 0, 0)) < 1e-9)


# ------------------------------------------------------- the demo, end to end

def verify_demo() -> None:
    from app.pipeline import cv_features as cv
    from app.pipeline.clip_classifier import ClipTrackClassifier, clip_wetness_from_probs
    from app.pipeline.orchestrator import decode_image, resize_for_clip
    from app.pipeline.smoothing import band_for, ewma
    from app.pipeline import trend as tr
    from app.sample_source import resolve_sample_dir

    section("Loading CLIP")
    clf = ClipTrackClassifier(BACKEND_DIR / "config" / "prompts.json")
    ok = clf.load()
    check("CLIP loads", ok, clf.load_error or clf.model_id)
    if not ok:
        return

    def run(source: str):
        folder, images, _ = resolve_sample_dir(source)
        raws, states, vectors = [], [], []
        prev = None
        for path in images:
            pil, bgr = decode_image(path.read_bytes())
            probs = clf.classify(resize_for_clip(pil))
            vectors.append(probs["image_vector"])
            cw = clip_wetness_from_probs(probs["p_dry"], probs["p_damp"], probs["p_wet"])
            ph = cv.analyze_surface(bgr)["physical_wetness"]
            raws.append(cv.fuse(cw, ph))
            sm = [ewma(raws[:i + 1]) for i in range(len(raws))]
            from app.pipeline.smoothing import stabilise_band
            band = stabilise_band(sm, prev)
            prev = band
            states.append(tr.display_state(band, tr.classify_trend(tr.fit_slope(sm), len(sm))))
        return images, raws, states, vectors

    section("The bundled sequence tells the story")
    images, raws, states, vectors = run("bundled")
    check("16 frames", len(images) == 16, f"{len(images)} found")
    check("starts WET", states[0] == "WET", states[0])
    check("ends DRY", states[-1] == "DRY", states[-1])
    check("passes through DRYING", "DRYING" in states,
          " ".join(dict.fromkeys(states)))
    check("wetness genuinely falls", raws[0] - raws[-1] > 0.4,
          f"{raws[0]:.3f} -> {raws[-1]:.3f}")

    section("Determinism")
    _, raws2, states2, _ = run("bundled")
    check("identical scores on a second run",
          all(abs(a - b) < 1e-12 for a, b in zip(raws, raws2)))
    check("identical states on a second run", states == states2)

    section("Real Hugging Face footage")
    hf_images, hf_raws, hf_states, hf_vectors = run("hf")
    check("20 frames", len(hf_images) == 20, f"{len(hf_images)} found")
    # The claim made to a judge is about the DISPLAYED state, which is the
    # smoothed value after hysteresis - not the raw per-frame score. A raw
    # reading can tip past a band edge for one frame; that is exactly the
    # flicker step 3 exists to absorb.
    check("reads damp throughout", all(s == "DAMP" for s in hf_states),
          " ".join(dict.fromkeys(hf_states)))
    check("raw scores stay in the damp region", 0.3 < min(hf_raws) and max(hf_raws) < 0.7,
          f"{min(hf_raws):.2f}-{max(hf_raws):.2f} raw")

    section("Zero-shot hazards separate wet from dry")
    from app.pipeline.hazards import HazardWatch
    watch = HazardWatch(BACKEND_DIR / "config" / "hazards.json")
    check("hazard watch loads", watch.load(clf), watch.error or f"{watch.status()['count']} detectors")
    if watch.ready:
        wet = {h["label"]: h["probability"] for h in watch.detect(vectors[0], clf.logit_scale)}
        dry = {h["label"]: h["probability"] for h in watch.detect(vectors[-1], clf.logit_scale)}
        check("black ice scores higher on the soaked frame",
              wet.get("Black ice", 0) > dry.get("Black ice", 1),
              f"wet {wet.get('Black ice', 0):.3f} vs dry {dry.get('Black ice', 0):.3f}")
        check("the dry frame triggers nothing",
              not any(h["triggered"] for h in watch.detect(vectors[-1], clf.logit_scale)))

    section("The landing screen quotes real numbers")
    # The landing screen shows a wetness readout beside the cross-fading
    # frames. Those values are hard-coded, which is fine only for as long as
    # they still match what the pipeline produces - otherwise the first thing a
    # judge sees is a number the app no longer agrees with. Checked here so it
    # cannot drift silently after a calibration change.
    import re
    landing = (BACKEND_DIR.parent / "frontend" / "src" / "components" / "Landing.jsx")
    if not landing.is_file():
        check("landing screen file found", False, str(landing))
    else:
        block = re.search(r"BUNDLED_WETNESS\s*=\s*\[(.*?)\]", landing.read_text(encoding="utf-8"), re.S)
        quoted = [float(x) for x in re.findall(r"[\d.]+", block.group(1))] if block else []
        actual = [round(ewma(raws[:i + 1]), 2) for i in range(len(raws))]
        check("landing quotes one value per bundled frame",
              len(quoted) == len(actual), f"{len(quoted)} quoted vs {len(actual)} frames")
        if len(quoted) == len(actual):
            worst = max(abs(a - b) for a, b in zip(quoted, actual))
            check("landing values still match the pipeline", worst <= 0.02,
                  f"largest difference {worst:.3f}")

    section("Light: no false alarms on real frames")
    bad = []
    for path in list(images) + list(hf_images):
        _, bgr = decode_image(path.read_bytes())
        if cv.assess_light(cv.analyze_surface(bgr)["raw"])["level"] != "ok":
            bad.append(path.name)
    check("all 36 demo frames read as well lit", not bad, ", ".join(bad) or "none flagged")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="pure functions only; skip anything needing CLIP")
    args = ap.parse_args()

    print("\nTrackSense AI - behaviour check")
    print("=" * 46)
    verify_pure()
    if args.quick:
        print("\n(--quick: skipped the end-to-end checks)")
    else:
        verify_demo()

    failed = [name for ok, name, _ in _results if not ok]
    print("\n" + "=" * 46)
    print(f"{len(_results) - len(failed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"  FAILED: {name}")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
