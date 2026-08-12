"""Download the CLIP weights ahead of time, so the demo never waits on wifi.

    python scripts/download_model.py

The model is about 600 MB. It is cached in ~/.cache/huggingface and reused
forever after. Run this once, on a connection you trust, well before you need to
present anything. If the weights are already cached this exits immediately.

No account, token or licence acceptance is required - openai/clip-vit-base-patch32
is a public model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    try:
        from app.pipeline.clip_classifier import ClipTrackClassifier, DEFAULT_MODEL_ID
    except ImportError as exc:
        print(f"Could not import the pipeline: {exc}")
        print("Fix: pip install -r requirements.txt")
        return 1

    print(f"\nFetching {DEFAULT_MODEL_ID} (~600 MB on first run) ...")
    print("This is a one-off. Later runs read it straight from the cache.\n")

    started = time.perf_counter()
    clf = ClipTrackClassifier(BACKEND_DIR / "config" / "prompts.json")
    ok = clf.load()
    elapsed = time.perf_counter() - started

    if not ok:
        print(f"FAILED after {elapsed:.1f}s: {clf.load_error}\n")
        print("Common causes:")
        print("  * no internet connection (the first run has to download)")
        print("  * a corporate proxy blocking huggingface.co")
        print("  * a full disk - the cache needs about 1 GB free")
        print("\nThe app still runs without this, in cv-fallback mode, using the")
        print("classical computer-vision signal alone.")
        return 1

    # Prove the weights work, rather than just reporting that files exist.
    try:
        import numpy as np
        from PIL import Image

        probe = Image.fromarray(np.full((224, 224, 3), 90, dtype=np.uint8))
        probs = clf.classify(probe)
        check = (f"sanity check ok - dry {probs['p_dry']:.2f} / "
                 f"damp {probs['p_damp']:.2f} / wet {probs['p_wet']:.2f} on a grey test image")
    except Exception as exc:
        check = f"weights loaded but inference failed: {exc}"

    from huggingface_hub.constants import HF_HUB_CACHE

    print(f"DONE in {elapsed:.1f}s")
    print(f"  model    {clf.model_id}")
    print(f"  cache    {HF_HUB_CACHE}")
    print(f"  prompts  {clf.status()['total_prompts']} across 3 categories")
    print(f"  {check}")
    print("\nThe weights are cached. You can now run the app offline.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
