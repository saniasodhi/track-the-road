# SETUP — what you personally have to do

## Nothing external is required. No accounts, no API keys, no payment methods.

I checked every dependency in this project against that claim, and it holds:

| Thing | Needs an account? | Needs a key? | Costs money? |
|---|---|---|---|
| Hugging Face CLIP weights | **No** — public model, downloads anonymously | No | No |
| Hugging Face dashcam dataset | **No** — public dataset, downloads anonymously | No | No |
| SQLite database | No — it is just a file on disk | No | No |
| FastAPI / uvicorn backend | No | No | No |
| React / Vite / Tailwind / Recharts | No | No | No |
| Google Fonts (frontend) | No | No | No |

**There is no bolded warning at the top of this file, because there is nothing you need to go and sign up for.** Everything below is either "run this once" or "optional".

---

## The checklist

### 1. Install Python 3.11 — *required, ~5 min, only if you do not have it*

**Why:** PyTorch (which runs CLIP) does not reliably have prebuilt packages for Python 3.14 yet. 3.9 through 3.13 all work; 3.11 is the safest.

You already have 3.11 at `C:\Users\Sania\AppData\Local\Programs\Python\Python311`, and the virtual environment in `backend/.venv` is already built against it. **So there is nothing to do here** — this entry only matters if you move to another machine.

On a new machine: install Python 3.11 from python.org, tick "Add to PATH", then recreate the environment:

```bash
py -3.11 -m venv backend/.venv
```

### 2. Install Node 18+ — *required, only if you do not have it*

You have Node 24.13.1. **Nothing to do.** On a new machine, install the LTS build from nodejs.org.

### 3. Install the Python packages — *already done*

Already installed in `backend/.venv`. On a new machine:

```bash
cd backend && pip install -r requirements.txt
```

### 4. Install the frontend packages — *already done*

Already installed in `frontend/node_modules`. On a new machine:

```bash
cd frontend && npm install
```

### 5. Download the CLIP weights — *already done*

**Why it matters:** the weights are about 600 MB. If they are not cached, the *first* frame you analyse triggers the download. You do not want that starting on campus wifi while a judge is watching.

**Status: already cached on this machine**, in `C:\Users\Sania\.cache\huggingface\hub\models--openai--clip-vit-base-patch32`. Verified working — the backend loads it in about 9 seconds at startup.

To re-check, or on a new machine:

```bash
cd backend && python scripts/download_model.py
```

### 6. Run preflight before you present — *required, 30 seconds, do this on demo day*

**Why:** it checks the eight things that could embarrass you, and prints a plain-English fix for each.

```bash
cd backend && python scripts/preflight.py
```

It checks Python version, installed packages, the prompt config, sample frames, that SQLite can write, that the CLIP weights are cached, that CLIP actually loads, and that ports 8000 and 5173 are free. Exit code 0 means you are fine.

### 7. Environment variables — *optional, skip unless deploying*

**The app runs correctly with no `.env` file at all.** Both `.env.example` files exist purely to document what is available:

- `backend/.env.example` — CORS origins, database path, data folder, log level, Hugging Face cache location
- `frontend/.env.example` — `VITE_API_BASE`, the backend URL

You only need these when the backend and frontend are not both on localhost, i.e. when you deploy. See the deploy section of the README.

### 8. Real Hugging Face footage — *already done*

**Status: already set up.** `backend/data/samples_hf/` holds 20 real frames extracted from the public Hugging Face dataset [`aap9002/UK-Road-DashCam`](https://huggingface.co/datasets/aap9002/UK-Road-DashCam) — UK roads, filmed 20 December 2024. They appear in the dashboard as the **Real dashcam** button.

Verified working: all 20 frames read DAMP / INTERMEDIATES, with a mean signal agreement of 90% (range 80–98%). Correct for a damp December road.

Nothing to do. To rebuild them, or on a new machine:

```bash
cd backend && python scripts/import_hf_dashcam.py
```

That downloads a 294 MB clip on first run. The video is cached in `backend/data/hf_dashcam/`, which is git-ignored — so a fresh clone re-downloads it, but the extracted frames are committed and work immediately.

### 9. Drop in your own photos — *optional, still the best remaining upgrade*

**Why:** you now have real footage, but it is from a moving car, so it shows wetness along a route rather than one place drying. Photos of **one spot** over time would demonstrate the drying trend on real data — which is the one thing nothing else in the project does.

Put 4 or more real photos in:

```
backend/data/samples_real/
```

Name them so alphabetical order is time order — `01.jpg`, `02.jpg`, `03.jpg`. Wet first, dry last. A third button, **My photos**, appears in the dashboard automatically as soon as there are 4 or more. Nothing else is affected — the synthetic and dashcam demos stay exactly as they are.

Then check the calibration still holds:

```bash
cd backend && python scripts/evaluate_samples.py
```

If your photos read brighter or grainier than the bundled set, adjust the anchor constants at the top of `backend/app/pipeline/cv_features.py` — they are all named and commented.

---

## Running it — two terminals

**Terminal 1:**

```bash
cd backend && .venv\Scripts\Activate.ps1
```

```bash
uvicorn app.main:app --reload
```

**Terminal 2:**

```bash
cd frontend && npm run dev
```

Then open <http://localhost:5173>.

---

## If something goes wrong mid-demo

| Symptom | What is happening | What to do |
|---|---|---|
| Model pill says `cv-fallback`, health dot is amber | CLIP did not load. The app is running on the classical computer-vision signal alone. | It still works and still demos. Say so out loud — it is a designed fallback, and it is a good answer to "what if the model fails?" |
| "Cannot reach the backend" | The backend is not running, or is on a different port. | Check terminal 1. The banner prints the real state of everything. |
| Demo button does nothing | No sample frames found. | `cd backend && python scripts/generate_samples.py` |
| Port 8000 already in use | Something else is on it, possibly an old backend. | `netstat -ano \| findstr :8000` then stop that process id. |
| First frame is very slow | CLIP weights are downloading. | You skipped step 5. Let it finish once — it never happens again. |

The `/demo` endpoint is the one to trust on stage: no upload, no camera, no network beyond localhost, and identical results every time.
