"""Where the demo gets its frames from.

There are three possible sources, and the demo endpoints can ask for any of
them by name:

  bundled  backend/data/samples/       the 16 synthetic frames committed to the
                                       repo, ordered wet -> dry. Always present,
                                       so the demo always has something to run.

  hf       backend/data/samples_hf/    real frames pulled out of a Hugging Face
                                       dataset (UK dashcam footage).
                                       Created by scripts/import_hf_dashcam.py.

  real     backend/data/samples_real/  drop your own photos in here, named so
                                       that sorting them alphabetically puts
                                       them in time order (01.jpg, 02.jpg, ...).

When no source is named, the priority is: your own photos, then the Hugging Face
frames, then the bundled set. A folder has to hold at least MIN_FRAMES images
before it is allowed to take over.

Sorting is alphabetical and the pipeline is deterministic, so any source
produces byte-identical results on every run.
"""

from __future__ import annotations

from pathlib import Path

from .db import DATA_DIR, REAL_SAMPLES_DIR, SAMPLES_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_FRAMES = 4

HF_SAMPLES_DIR = DATA_DIR / "samples_hf"

SOURCES: dict[str, Path] = {
    "bundled": SAMPLES_DIR,
    "hf": HF_SAMPLES_DIR,
    "real": REAL_SAMPLES_DIR,
}

# Which source wins when the caller does not name one.
# Your own photos first, because putting them there is an explicit choice.
# Then the bundled sequence, because it is the one that actually shows a track
# drying out. The Hugging Face dashcam frames are only used when asked for by
# name - they are real, but filmed from a moving car, so they show wetness
# along a route rather than one place changing over time.
AUTO_PRIORITY = ("real", "bundled", "hf")

LABELS = {
    "bundled": "Synthetic track sequence",
    "hf": "Real UK dashcam (Hugging Face)",
    "real": "Your own photos",
}

KINDS = {"bundled": "synthetic", "hf": "real", "real": "real"}


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


def resolve_sample_dir(source: str | None = None) -> tuple[Path, list[Path], bool]:
    """Return (folder, sorted image paths, using_real_photos).

    `source` may be "bundled", "hf", "real", or None/"auto" to pick
    automatically. An explicitly named source is honoured even if it is empty -
    the caller then gets an empty list and can report a proper error, rather
    than silently being given different footage than it asked for.
    """
    if source and source != "auto":
        folder = SOURCES.get(source)
        if folder is None:
            raise ValueError(f"Unknown sample source {source!r}. "
                             f"Expected one of: {', '.join(SOURCES)}")
        return folder, _list_images(folder), KINDS[source] == "real"

    for name in AUTO_PRIORITY:
        images = _list_images(SOURCES[name])
        if len(images) >= MIN_FRAMES:
            return SOURCES[name], images, KINDS[name] == "real"

    return SAMPLES_DIR, _list_images(SAMPLES_DIR), False


def active_source_name() -> str:
    """Which source would be used right now if nobody named one."""
    for name in AUTO_PRIORITY:
        if len(_list_images(SOURCES[name])) >= MIN_FRAMES:
            return name
    return "bundled"


def describe_samples() -> dict:
    """Health-check view of the sample situation, across every source."""
    folder, images, is_real = resolve_sample_dir()
    active = active_source_name()

    available = {}
    for name, path in SOURCES.items():
        found = _list_images(path)
        available[name] = {
            "dir": str(path),
            "label": LABELS[name],
            "kind": KINDS[name],
            "count": len(found),
            "usable": len(found) >= MIN_FRAMES,
            "files": [f"/media/{path.name}/{p.name}" for p in found],
        }

    return {
        # Kept flat for backwards compatibility with the startup banner.
        "dir": str(folder),
        "count": len(images),
        "using_real_photos": is_real,
        "kind": KINDS[active],
        "active_source": active,
        "first": images[0].name if images else None,
        "last": images[-1].name if images else None,
        "files": [f"/media/{folder.name}/{p.name}" for p in images],
        "sources": available,
        "real_photo_dir": str(REAL_SAMPLES_DIR),
        "real_photo_count": len(_list_images(REAL_SAMPLES_DIR)),
    }
