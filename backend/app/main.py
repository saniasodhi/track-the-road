"""TrackSense AI - FastAPI application entry point.

Start it with:

    uvicorn app.main:app --reload

from inside the `backend/` folder. Everything else (database file, tables,
model weights) is created or loaded automatically.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .db import CONFIG_DIR, DATA_DIR, DB_PATH, db_status, init_db
from .pipeline.orchestrator import startup_load
from .routes import analyze, hazards, health, sessions
from .sample_source import describe_samples

load_dotenv()

logging.basicConfig(
    level=os.getenv("TRACKSENSE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tracksense")

# Comma-separated list, or "*" for anything. The Vite dev server runs on 5173.
CORS_ORIGINS = os.getenv(
    "TRACKSENSE_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
)


# --------------------------------------------------------------------- banner

def _tick(ok: bool) -> str:
    return "[ OK ]" if ok else "[ -- ]"


def print_startup_banner() -> None:
    """Print, in plain English, what is set up and what is not."""
    clf_status = startup_load().status()
    db_info = db_status()
    samples = describe_samples()

    lines = [
        "",
        "==================================================================",
        "  TrackSense AI  -  road surface wetness from camera frames",
        "==================================================================",
        f"  {_tick(clf_status['loaded'])}  Hugging Face CLIP   {clf_status['model_id']}",
    ]
    if clf_status["loaded"]:
        lines.append(
            f"          loaded in {clf_status['load_seconds']}s from "
            f"{clf_status['total_prompts']} prompts across 3 categories"
        )
    else:
        lines.append(f"          NOT LOADED: {clf_status['error']}")
        lines.append("          Running in cv-fallback mode (classical CV only).")
        lines.append("          Fix: pip install -r requirements.txt, then check your")
        lines.append("          internet connection so the weights can download once.")

    lines += [
        f"  {_tick(db_info['writable'])}  SQLite database     {DB_PATH}",
    ]
    if not db_info["writable"]:
        lines.append(f"          NOT WRITABLE: {db_info['error']}")

    lines += [
        f"  {_tick(samples['count'] > 0)}  Sample frames       "
        f"{samples['count']} {samples['kind']} frames in {samples['dir']}",
    ]
    if samples["count"] == 0:
        lines.append("          Fix: python scripts/generate_samples.py")
    if samples["real_photo_count"] and not samples["using_real_photos"]:
        lines.append(
            f"          ({samples['real_photo_count']} real photo(s) present but at "
            f"least 4 are needed before they replace the synthetic set)"
        )

    lines += [
        f"  {_tick((CONFIG_DIR / 'prompts.json').exists())}  Prompt config       "
        f"{CONFIG_DIR / 'prompts.json'}",
        "  ------------------------------------------------------------",
        "  No API keys, accounts or paid services are required.",
        "  API docs:  http://127.0.0.1:8000/docs",
        "  Health:    http://127.0.0.1:8000/api/health",
        "==================================================================",
        "",
    ]
    print("\n".join(lines), file=sys.stderr, flush=True)


# ------------------------------------------------------------------ lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Loading CLIP takes a few seconds. Doing it here means the first frame a
    # judge analyses is fast, instead of paying the cost mid-demo.
    print_startup_banner()
    yield


app = FastAPI(
    title="TrackSense AI",
    version="1.0.0",
    description=(
        "Estimates how wet a road or race track surface is from camera frames, "
        "using Hugging Face CLIP plus classical computer vision, then works out "
        "which way conditions are heading and what tyres to run."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS.strip() == "*" else
    [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frame images (bundled samples and uploads) are served straight off disk.
app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")

app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(analyze.router)
app.include_router(hazards.router)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({
        "name": "TrackSense AI",
        "docs": "/docs",
        "health": "/api/health",
    })
