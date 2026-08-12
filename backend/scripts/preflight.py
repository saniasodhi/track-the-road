"""Check this machine can run TrackSense AI, before you need it to.

    python scripts/preflight.py

Every check prints PASS, WARN or FAIL, and every failure prints the exact
command that fixes it. Run this once when you sit down, not when the judges are
watching.

Exit code is 0 if nothing FAILED (warnings are fine), 1 otherwise.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str, str]] = []   # (level, name, detail, fix)


def record(level: str, name: str, detail: str, fix: str = "") -> None:
    results.append((level, name, detail, fix))


# --------------------------------------------------------------------- checks

def check_python() -> None:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 9):
        record(FAIL, "Python version", f"{version} is too old",
               "Install Python 3.11 and recreate the virtual environment.")
    elif v >= (3, 14):
        record(WARN, "Python version", f"{version} - PyTorch wheels may not exist yet",
               "If pip install fails, use Python 3.11: py -3.11 -m venv .venv")
    else:
        record(PASS, "Python version", version)


def check_packages() -> None:
    needed = {
        "fastapi": "FastAPI web framework",
        "uvicorn": "ASGI server",
        "sqlalchemy": "database layer",
        "cv2": "OpenCV (classical computer vision, step 2)",
        "numpy": "numerics",
        "PIL": "Pillow, image decoding",
        "torch": "PyTorch (runs CLIP)",
        "transformers": "Hugging Face transformers (CLIP)",
        "multipart": "python-multipart, needed for file uploads",
    }
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:
        record(FAIL, "Python packages",
               "missing: " + ", ".join(missing),
               "pip install -r requirements.txt")
    else:
        try:
            import torch, transformers  # noqa: E401
            detail = f"all present (torch {torch.__version__}, transformers {transformers.__version__})"
        except Exception:
            detail = "all present"
        record(PASS, "Python packages", detail)


def check_model_weights() -> None:
    """Are the CLIP weights already downloaded? ~600 MB, and you do not want
    that starting on campus wifi while a judge waits."""
    if importlib.util.find_spec("huggingface_hub") is None:
        record(FAIL, "CLIP weights", "huggingface_hub is not installed",
               "pip install -r requirements.txt")
        return

    model_id = "openai/clip-vit-base-patch32"
    try:
        from app.pipeline.clip_classifier import DEFAULT_MODEL_ID
        model_id = DEFAULT_MODEL_ID
    except Exception:
        pass

    cache = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = cache / "hub" if (cache / "hub").exists() else cache
    folder = hub / ("models--" + model_id.replace("/", "--"))

    if not folder.exists():
        record(FAIL, "CLIP weights", f"not cached ({model_id})",
               "python scripts/download_model.py   (~600 MB, needs internet once)")
        return

    size_mb = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
    if size_mb < 300:
        record(WARN, "CLIP weights", f"cache looks incomplete ({size_mb:.0f} MB)",
               "python scripts/download_model.py")
    else:
        record(PASS, "CLIP weights", f"cached, {size_mb:.0f} MB in {folder}")


def check_model_loads() -> None:
    try:
        from app.pipeline.clip_classifier import ClipTrackClassifier
        clf = ClipTrackClassifier(BACKEND_DIR / "config" / "prompts.json")
        if clf.load():
            record(PASS, "CLIP loads",
                   f"{clf.model_id} in {clf.load_seconds}s, "
                   f"{clf.status()['total_prompts']} prompts")
        else:
            record(WARN, "CLIP loads", clf.load_error or "unknown error",
                   "The app still runs in cv-fallback mode. "
                   "Check internet access, then: python scripts/download_model.py")
    except Exception as exc:
        record(WARN, "CLIP loads", f"{type(exc).__name__}: {exc}",
               "The app still runs in cv-fallback mode.")


def check_samples() -> None:
    try:
        from app.sample_source import describe_samples
        info = describe_samples()
    except Exception as exc:
        record(FAIL, "Sample frames", f"could not inspect: {exc}",
               "python scripts/generate_samples.py")
        return

    if info["count"] == 0:
        record(FAIL, "Sample frames", f"none found in {info['dir']}",
               "python scripts/generate_samples.py")
    elif info["count"] < 8:
        record(WARN, "Sample frames", f"only {info['count']} frames - the trend needs a few more",
               "Add more frames, or: python scripts/generate_samples.py")
    else:
        ready = [f"{name} ({s['count']})"
                 for name, s in info.get("sources", {}).items() if s["usable"]]
        record(PASS, "Sample frames",
               f"{info['count']} {info['kind']} frames active; "
               f"sources ready: {', '.join(ready) if ready else 'none'}")


def check_database() -> None:
    try:
        from app.db import DB_PATH, init_db, db_status
        init_db()
        info = db_status()
        if info["writable"]:
            record(PASS, "SQLite database", f"writable at {DB_PATH}")
        else:
            record(FAIL, "SQLite database", info["error"] or "not writable",
                   f"Check folder permissions for {DB_PATH.parent}")
    except Exception as exc:
        record(FAIL, "SQLite database", f"{type(exc).__name__}: {exc}",
               "Check that backend/data/ exists and is writable.")


def _port_free(port: int) -> bool:
    """True if nothing is listening on the port.

    Both loopback addresses have to be probed: on Windows "localhost" often
    resolves to the IPv6 ::1 first, so a dev server can be live on ::1 while
    127.0.0.1 looks completely free.
    """
    targets = [(socket.AF_INET, ("127.0.0.1", port))]
    if socket.has_ipv6:
        targets.append((socket.AF_INET6, ("::1", port)))

    for family, address in targets:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.4)
                if sock.connect_ex(address) == 0:
                    return False
        except OSError:
            continue
    return True


def check_ports() -> None:
    for port, who in ((8000, "backend"), (5173, "frontend dev server")):
        if _port_free(port):
            record(PASS, f"Port {port}", f"free for the {who}")
        else:
            record(WARN, f"Port {port}", f"already in use - something is on the {who} port",
                   f"Either that is TrackSense already running, or free it: "
                   f"netstat -ano | findstr :{port}")


def check_config() -> None:
    prompts = BACKEND_DIR / "config" / "prompts.json"
    if not prompts.exists():
        record(WARN, "Prompt config", f"{prompts} missing",
               "The app falls back to built-in prompts, but you cannot tune them.")
        return
    try:
        import json
        data = json.loads(prompts.read_text(encoding="utf-8"))
        cats = data.get("categories", {})
        counts = {k: len(v) for k, v in cats.items()}
        if set(cats) != {"dry", "damp", "wet"}:
            record(WARN, "Prompt config",
                   f"expected categories dry/damp/wet, found {sorted(cats)}",
                   "Edit backend/config/prompts.json")
        else:
            record(PASS, "Prompt config", f"dry {counts['dry']}, damp {counts['damp']}, wet {counts['wet']} prompts")
    except Exception as exc:
        record(FAIL, "Prompt config", f"is not valid JSON: {exc}",
               "Fix the JSON in backend/config/prompts.json")


# ----------------------------------------------------------------------- main

def main() -> int:
    print("\nTrackSense AI - preflight\n" + "=" * 62)

    check_python()
    check_packages()
    check_config()
    check_samples()
    check_database()
    check_model_weights()
    check_model_loads()
    check_ports()

    width = max(len(name) for _, name, _, _ in results)
    for level, name, detail, fix in results:
        print(f"  [{level}]  {name.ljust(width)}   {detail}")
        if fix and level != PASS:
            print(f"          {' ' * width}   -> {fix}")

    failures = sum(1 for level, *_ in results if level == FAIL)
    warnings = sum(1 for level, *_ in results if level == WARN)

    print("=" * 62)
    if failures:
        print(f"{failures} check(s) FAILED, {warnings} warning(s). Fix the failures above.\n")
        return 1
    if warnings:
        print(f"All required checks passed, with {warnings} warning(s). "
              "The app will run.\n")
    else:
        print("Everything passed. Start the backend with:\n"
              "    uvicorn app.main:app --reload\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
