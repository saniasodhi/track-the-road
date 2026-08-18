"""Turn a real drying experiment into frames the pipeline can read.

    python scripts/import_drying_experiment.py photos/
    python scripts/import_drying_experiment.py clip.mp4 --minutes 40

Why this script exists
----------------------
The central claim of this project is that drying is a direction, not a look.
Every frame that currently demonstrates that is synthetic. This closes that gap
with a real surface, real water and real evaporation - and because you control
when the water goes down, you get GROUND TRUTH FOR FREE.

Unlike the bundled synthetic set, the frames this produces carry REAL elapsed
time rather than frame numbers. That is what lets the forecast talk in minutes.


                        HOW TO RUN THE EXPERIMENT
                        =========================

You need a bucket of water, a phone, something to prop it on, and 40 minutes.
Rain is not required, and you should not wait for it - you want control.

 1. Pick a patch of tarmac or concrete in the open. Not under a tree: moving
    shade wrecks the exposure and looks like a change in the surface.

 2. Prop the phone so it looks at the patch at a shallow angle, the way a road
    camera would - not straight down. The road should fill most of the frame,
    and the bottom half should be nothing but surface.

 3. LOCK THE EXPOSURE. This is the one thing that ruins the experiment.

    iPhone: tap and hold on the road until AE/AF LOCK appears.
    Android: tap the surface, then use the camera's exposure lock.

    Skip this and the phone brightens the image as the road dries and darkens
    it as it wets - actively cancelling the signal you are trying to measure.
    You would be photographing the camera's compensation, not the road.

 4. Take one photo. That is your dry reference. Do not move the phone again.

 5. Pour the water. Cover the whole visible patch, generously.

 6. Take a photo every 60-90 seconds until the patch looks dry to your eye,
    then take three more. 25 to 40 photos is ideal.

 7. WRITE DOWN the minute at which it looked dry to you. That is your ground
    truth, and it turns "here is a curve" into "we predicted 26 minutes and it
    was actually dry at 29".

 8. Do not stand between the sun and the patch. Your own shadow crossing the
    frame is the second most common way to ruin this.

Then point this script at the folder.


What it does
------------
Photos are read in filename order, and if they carry EXIF capture times the
real elapsed seconds are recovered automatically - you do not have to tell it
anything. Video is sampled evenly across the clip; pass --minutes so real time
is known.

Output lands in backend/data/samples_real/ with a manifest carrying the true
elapsed time of every frame. The dashboard picks that folder up on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BACKEND_DIR / "data" / "samples_real"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_WIDTH = 960
MAX_FRAMES = 45


def exif_seconds(paths: list[Path]) -> list[float] | None:
    """Real elapsed seconds from EXIF capture times, or None if unavailable.

    This is the reason to shoot stills rather than video: the phone already
    recorded exactly when each frame was taken, so the true timeline comes back
    for free and nobody has to remember how long the experiment ran.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    stamps = []
    for p in paths:
        try:
            with Image.open(p) as img:
                exif = img.getexif()
                raw = exif.get(36867) or exif.get(306)   # DateTimeOriginal, DateTime
            if not raw:
                return None
            stamps.append(datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S"))
        except Exception:
            return None

    if len(stamps) != len(paths) or len(set(stamps)) < 2:
        return None
    origin = min(stamps)
    return [(s - origin).total_seconds() for s in stamps]


def write_frame(img, index: int, elapsed_s: float, source: str) -> dict:
    h, w = img.shape[:2]
    if w != TARGET_WIDTH:
        img = cv2.resize(img, (TARGET_WIDTH, int(round(h * TARGET_WIDTH / w))),
                         interpolation=cv2.INTER_AREA)
    name = f"dry_{index:03d}.jpg"
    cv2.imwrite(str(OUT_DIR / name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"  {name}   t = {elapsed_s / 60:6.1f} min")
    return {
        "file": name,
        "elapsed_s": round(elapsed_s, 1),
        "elapsed_min": round(elapsed_s / 60, 2),
        "source": source,
    }


def from_photos(folder: Path, minutes: float | None) -> list[dict]:
    photos = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )
    if len(photos) < 4:
        print(f"Found {len(photos)} images in {folder}. Need at least 4.")
        return []

    if len(photos) > MAX_FRAMES:
        step = len(photos) / MAX_FRAMES
        photos = [photos[int(i * step)] for i in range(MAX_FRAMES)]
        print(f"Using {len(photos)} evenly spaced frames.")

    times = exif_seconds(photos)
    if times:
        print(f"Recovered real capture times from EXIF - "
              f"{times[-1] / 60:.1f} minutes end to end.")
    elif minutes:
        span = minutes * 60.0
        times = [i * span / max(1, len(photos) - 1) for i in range(len(photos))]
        print(f"No EXIF times; spreading {len(photos)} frames evenly over "
              f"{minutes:.0f} minutes as specified.")
    else:
        times = [float(i * 60) for i in range(len(photos))]
        print("No EXIF times and no --minutes given. Assuming one frame per "
              "minute. Pass --minutes for a real timeline.")

    out = []
    for i, (src, t) in enumerate(zip(photos, times)):
        img = cv2.imread(str(src))
        if img is None:
            print(f"  skipping unreadable {src.name}")
            continue
        out.append(write_frame(img, i, t, src.name))
    return out


def from_video(path: Path, minutes: float | None) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"OpenCV could not open {path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_seconds = total / fps if total > 0 else 0.0

    if minutes:
        real_span = minutes * 60.0
        print(f"Clip is {clip_seconds:.0f}s of footage representing "
              f"{minutes:.0f} real minutes.")
    else:
        real_span = clip_seconds
        print(f"No --minutes given; treating the clip as real time "
              f"({clip_seconds / 60:.1f} minutes). Pass --minutes if this was a "
              f"time-lapse.")

    count = min(MAX_FRAMES, max(4, (total // 10) or 4))
    out = []
    for i in range(count):
        frac = i / max(1, count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * max(0, total - 1)))
        ok, img = cap.read()
        if not ok:
            continue
        out.append(write_frame(img, i, frac * real_span, f"{path.name}@{frac:.2f}"))
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Import a real drying experiment. Read the top of this file "
                    "for how to shoot it.",
    )
    ap.add_argument("source", help="folder of photos, or a video file")
    ap.add_argument("--minutes", type=float, default=None,
                    help="real elapsed minutes the capture covers")
    ap.add_argument("--dry-at", type=float, default=None,
                    help="the minute it looked dry to your eye - your ground truth")
    ap.add_argument("--keep", action="store_true",
                    help="keep frames already in samples_real instead of clearing")
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        print(f"Not found: {source}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.keep:
        for old in OUT_DIR.glob("*.jpg"):
            old.unlink()

    print()
    frames = (from_photos(source, args.minutes) if source.is_dir()
              else from_video(source, args.minutes))
    if not frames:
        print("\nNothing imported.")
        return 1

    manifest = {
        "synthetic": False,
        "kind": "drying_experiment",
        "source": str(source),
        "captured_frames": len(frames),
        "duration_min": round(frames[-1]["elapsed_s"] / 60, 2),
        "observed_dry_at_min": args.dry_at,
        "note": "Real controlled drying experiment. elapsed_s is true wall-clock "
                "time since the water went down, which is what lets the forecast "
                "report minutes rather than frames.",
        "frames": frames,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nImported {len(frames)} frames covering "
          f"{manifest['duration_min']:.1f} minutes -> {OUT_DIR}")
    if args.dry_at:
        print(f"Ground truth recorded: looked dry at {args.dry_at:.0f} min.")
    else:
        print("Tip: re-run with --dry-at N to record when it looked dry to you.")
    print("\nThe dashboard will show a 'My photos' button. Then run:")
    print("    python scripts/analyse_drying.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
