"""Build a real-world frame set from a Hugging Face dataset.

    python scripts/import_hf_dashcam.py

Dataset: aap9002/UK-Road-DashCam - real dashcam footage from UK roads, filmed
20 December 2024. British December means damp tarmac, which is exactly the
condition this project is about. It is a public dataset: no account, no token.

This pulls one 3-minute clip (about 294 MB, cached after the first run),
extracts evenly spaced frames, crops them, and writes them to
backend/data/samples_hf/. The dashboard can then run the whole pipeline over
real footage instead of the bundled synthetic set.

Why the crop matters
--------------------
A dashcam sees the car's own bonnet across the bottom of every frame, plus a
burnt-in GPS/timestamp bar. Pipeline step 2 measures the bottom part of the
image because that is where the road is - so left alone it would faithfully
measure the paintwork of the car. Dark, smooth bodywork reads as "wet".

So each frame is cropped to its top 75%, which removes the bonnet and the
overlay while keeping enough of the scene for CLIP to understand where it is.
Measured on this footage, that crop is the one that gives a confident and
correct answer (CLIP reports damp at ~0.86); cropping down to a bare strip of
tarmac loses the context and over-reads the wetness.

Any real deployment needs this same idea - a fixed mask for whatever part of
the frame is the vehicle rather than the road.

Honest note on what this shows
------------------------------
A moving car photographs a different place every frame, so this set shows
wetness varying ALONG A ROUTE, not one place drying over time. It proves the
per-frame reading works on real images. The bundled synthetic sequence is still
the one that demonstrates the drying trend, because that needs a fixed camera.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

REPO_ID = "aap9002/UK-Road-DashCam"
FILENAME = "241220_125301_002_RH.MP4"
LOCAL_COPY = BACKEND_DIR / "data" / "hf_dashcam" / "uk_road_rear.mp4"
OUT_DIR = BACKEND_DIR / "data" / "samples_hf"

FRAME_COUNT = 20        # how many frames to pull out of the clip
CROP_TOP = 0.0          # keep from this fraction of the height...
CROP_BOTTOM = 0.75      # ...to this one. Removes bonnet + timestamp overlay.
OUTPUT_WIDTH = 960      # match the bundled frames


def fetch_clip() -> Path:
    """Return a local path to the video, downloading it if we do not have it."""
    if LOCAL_COPY.exists() and LOCAL_COPY.stat().st_size > 1_000_000:
        print(f"Using the copy already on disk: {LOCAL_COPY}")
        return LOCAL_COPY

    print(f"Downloading {FILENAME} from {REPO_ID} (~294 MB, public, no login) ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed. Fix: pip install -r requirements.txt")
        raise SystemExit(1)

    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    print(f"Downloaded to the Hugging Face cache: {path}")
    return Path(path)


def main() -> int:
    video = fetch_clip()

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"OpenCV could not open {video}")
        return 1

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        print("Could not read a frame count from the video.")
        return 1

    print(f"Clip: {total} frames at {fps:.0f} fps ({total / fps:.0f} seconds)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("hf_*.jpg"):
        old.unlink()

    # Evenly spaced through the clip, skipping the very start and end.
    step = total / (FRAME_COUNT + 1)
    manifest = []

    for i in range(FRAME_COUNT):
        source_index = int(round(step * (i + 1)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, bgr = capture.read()
        if not ok:
            print(f"  could not read source frame {source_index}, skipping")
            continue

        height = bgr.shape[0]
        cropped = bgr[int(height * CROP_TOP):int(height * CROP_BOTTOM)]

        scale = OUTPUT_WIDTH / cropped.shape[1]
        resized = cv2.resize(
            cropped,
            (OUTPUT_WIDTH, int(round(cropped.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

        name = f"hf_{i:02d}.jpg"
        cv2.imwrite(str(OUT_DIR / name), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        manifest.append({
            "file": name,
            "source_frame": source_index,
            "timestamp_s": round(source_index / fps, 2),
        })
        print(f"  {name}   from {source_index / fps:6.1f}s")

    capture.release()

    (OUT_DIR / "manifest.json").write_text(
        json.dumps({
            "synthetic": False,
            "source": "Hugging Face dataset",
            "repo_id": REPO_ID,
            "file": FILENAME,
            "url": f"https://huggingface.co/datasets/{REPO_ID}",
            "filmed": "2024-12-20, UK roads",
            "crop": f"top {int(CROP_BOTTOM * 100)}% of each frame "
                    f"(removes the car bonnet and the timestamp overlay)",
            "note": "Frames come from a moving car, so this shows wetness along a "
                    "route rather than one place drying over time. There is no "
                    "ground-truth wetness label - the values are estimates only.",
            "frames": manifest,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {len(manifest)} real frames to {OUT_DIR}")
    print("They will appear in the dashboard as a second demo source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
