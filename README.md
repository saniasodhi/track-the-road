# TrackSense AI

**Looks at photos of a road and tells you how wet it is, which way conditions are heading, and what tyres to run.**

Built with a Hugging Face model (CLIP) plus classical computer vision. It is a four-step pipeline that turns a general-purpose AI model into a road-surface sensor.

---

## What it does

Give it camera frames. It answers four questions:

| | |
|---|---|
| **What is it now?** | DRY, DAMP, WET or DRYING |
| **How wet?** | A number from 0 to 1 |
| **Where is it going?** | Improving, stable, or deteriorating |
| **What do I do?** | Slicks, intermediates, or full wets |
| **How long until it's safe?** | A time-to-dry estimate, in minutes |

It also scores a **3×3 grid of the road**, because tracks do not dry evenly. It can tell you "the track is drying, but the near left is still standing in water."

It refuses to answer when it cannot see — a frame too dark to read gets no tyre call, only a hold.

Three views of the same session: **Dashboard** (the answer), **Network** (every point at once, ranked by severity), **Signals** (every number and the arithmetic behind it).

Two things it does that a trained classifier cannot:

- **Point it at a live camera.** Frames from your webcam go through the exact same four-step pipeline as the bundled ones — no special path in the backend.
- **Teach it a new hazard by typing one sentence.** Type "black ice" and a working detector exists ~150 ms later. No training data, no labelling, no retraining, no redeploy.

---

## Why it matters beyond racing

The demo is a race track because that is where a wet-surface decision is most visible. The real uses are:

- **Road safety cameras.** Councils already have thousands of cameras pointed at roads. Today they mostly report that it is raining. This reports which stretches are *still* wet twenty minutes later — which is where crashes happen.
- **Runway monitoring.** Checking a runway for standing water is currently manual and periodic. A camera that reports continuously is strictly more information.
- **Self-driving cars.** A car needs to lengthen its stopping distance *before* it needs the grip, not after.

---

## Run it

You need **Python 3.11** and **Node 18+**. See [SETUP.md](SETUP.md).

**Terminal 1 — backend**

```bash
cd backend && python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_model.py
```

```bash
uvicorn app.main:app --reload
```

**Terminal 2 — frontend**

```bash
cd frontend && npm install
```

```bash
npm run dev
```

Open <http://localhost:5173> and press **Start session**. The dashboard analyses the bundled sequence by itself.

Check your machine is ready at any time:

```bash
cd backend && python scripts/preflight.py
```

---

## How the AI works

There is no ready-made "is this road wet" model anywhere. Training one would need thousands of labelled photos we do not have. So instead of training a model, we **assemble** one out of four steps.

```
   photo
     |
 [1] CLIP           what does a vision-language model think this is?
     |                    dry / damp / wet
 [2] Optics         measure the physics directly - no model at all
     |                    shine, darkness, colour, texture
     +---> combine: 65% CLIP + 35% optics
     |
 [3] Smoothing      average 5 frames, make labels earn their change
     |
 [4] Trend          fit a line through 10 frames -> direction -> tyre call
     |
   answer
```

### Step 1 — Ask a model that already knows what things look like

**CLIP** is a Hugging Face model trained on hundreds of millions of images paired with their captions. Because of that, it can take a photo and a list of sentences and tell you which sentence fits best.

So we write the sentences ourselves:

> "a dry asphalt race track" … "a damp race track surface" … "a soaking wet race track"

CLIP returns three probabilities: dry, damp, wet.

**The refinement that matters:** we use **four** phrasings per category, not one. A single sentence is a noisy probe — "a wet road" might accidentally latch onto rain, or darkness, or a camera angle. So we convert all four into CLIP's internal number-vectors and **average them**. The average keeps what those sentences have in common — the wetness — and their accidental differences cancel out. This is called prompt ensembling.

All twelve sentences live in `backend/config/prompts.json`. Edit them, restart, done.

### Step 2 — Measure the physics ourselves, with no model

If the whole app were "send the photo to CLIP", it would be one call to one tool. So we build a **second, completely independent opinion** using OpenCV and arithmetic — no training, no weights. It measures four things that physically change when a road gets wet:

| Measured | Why it changes |
|---|---|
| **Shine** — how many pixels are blown-out white | Dry asphalt scatters light. Water turns it into a mirror. |
| **Darkness** — average brightness | Water fills the pores; light goes in and does not come out. |
| **Colour** — average saturation | A neutral sky reflection washes out the surface colour. |
| **Texture** — how grainy it is | Dry asphalt is thousands of tiny stones. Water smooths it over. |

Then the two opinions combine:

```
wetness = 0.65 × (what CLIP thinks) + 0.35 × (what the optics measure)
```

CLIP gets the bigger share because it understands *context* — it knows a wet road from a road at dusk. The optics get a real vote because they are grounded in physics and cannot be fooled by an unusual scene.

**Both numbers are shown separately on screen.** Two independent methods agreeing is a far stronger claim than one method being confident.

**And the gap between them is free confidence.** That gap becomes the shaded band on the chart and the AGREEMENT percentage. Narrow band = both methods landed in the same place. On real footage they agree to within a few percent.

### Step 2b — Where on the road is it wet?

A track does not dry evenly — the racing line dries first, the edges hold water. So the same optics score a **3×3 grid** that follows the perspective (rows are distance, columns are left/centre/right) and the app names the wettest cell.

On real dashcam footage this gives: frame reads DAMP at 0.47 overall, near-left cell reads 0.66 — genuinely wet. The app says *"still wet on the near left."*

*Honest note:* the zones use the optics only, not CLIP. Nine extra CLIP passes per frame would be nine times slower, and CLIP handed a bare tile of tarmac loses the context that makes it good. So CLIP and optics set the overall **level**; the per-cell optics describe the **shape** around it.

### Zero-shot hazard watch — the part a trained model cannot do

The pipeline classifies dry / damp / wet because those are the three things we wrote descriptions for. **Nothing about the architecture is limited to three.**

CLIP compares an image against *any* sentence. So a brand-new detector costs one sentence and zero training data. Type "black ice" into the dashboard and a black-ice detector exists about 150 milliseconds later — the phrasings are embedded once and cached. There is nothing to train, because there is nothing being fitted.

This is the actual reason to build on a vision-language model instead of training a classifier. A model trained on a fixed label set physically cannot gain a new class without new labelled images and a retraining run.

**How each detector is scored.** Every hazard is a binary question asked independently of the others: *does this look more like the hazard, or like an ordinary road?* We embed the hazard's phrasings into one prototype, embed a fixed set of "ordinary road" references into another, and softmax the two similarities at CLIP's own temperature.

Scoring against a neutral baseline — rather than putting all hazards in one softmax — matters: detectors do not compete, a frame can legitimately trigger two at once (wet *and* foggy), and the number stays interpretable.

Measured on the bundled frames:

| | Black ice | Snow | Fog |
|---|---|---|---|
| Soaked frame | 22.6% | 1.9% | 8.9% |
| Dry frame | 0.3% | 0.0% | 0.1% |

The dry road triggers nothing. The wet one shows elevated black-ice similarity — and that is not a bug. Wet and icy asphalt genuinely look alike, which is precisely why black ice is dangerous.

Each hazard costs two dot products per frame, because it reuses the image vector CLIP already computed. Per-frame latency is unchanged at ~85 ms.

### Step 3 — Stop it flickering

Raw per-frame scores jump around. If the number sits near a boundary, the label flips WET / DAMP / WET and the app looks broken even though nothing changed. Two fixes:

1. **Average the last five frames**, weighted so the newest counts most.
2. **Make the label earn its change** — it must cross the boundary *and stay across for two frames*. One frame over the line is noise.

### Step 4 — Work out the direction

**This is the idea the whole product rests on.**

You cannot tell from a single photo whether a damp road is drying or getting wetter. The pixels are identical. A road at 0.40 on the way down looks exactly like a road at 0.40 on the way up.

> Drying is not something a surface *looks like*. It is a **direction**.

So CLIP is never asked "is it drying?" — it can only say dry, damp or wet, because those are the only things visible in one frame. **DRYING is computed**, by fitting a straight line through the last ten smoothed scores:

- falling faster than −0.015 per frame → **IMPROVING**
- rising faster than +0.015 per frame → **DETERIORATING**
- in between → **STABLE**

And then: **damp + improving = DRYING**.

The slope also forecasts: *"slicks viable in about 6 frames."* If the trend is STABLE, no forecast is offered — having decided that slope is noise, extrapolating from it would contradict ourselves.

The tyre call is a **plain lookup table**, deliberately not a model. Anything deciding whether a driver goes out on slicks must be readable and arguable by a human. Every threshold is visible in `backend/app/pipeline/trend.py`.

### Forecasting the time to dry

Step 4 gives a direction. The obvious next question is *how long* — and answering in frames, *"slicks viable in about 6 frames"*, is useless, because frames are not a unit anyone can act on.

So when a session carries real timestamps, a second model is fitted for the forecast:

```
w(t) = w_dry + (w_0 - w_dry) * exp(-t / tau)
```

**Why not just extend the straight line?** A line is the right tool for a *direction* — robust, assumption-free, and a sign is all it has to produce. It is the wrong tool for a *forecast*. Evaporation from a thin film is roughly proportional to how much water is left, so the curve decays toward a dry baseline and flattens as it approaches. A line extrapolated through that always crosses the threshold too early.

Measured against a surface with a known 11-minute time constant, forecasting while the road was still wet:

| observed | exponential | straight line |
|---|---|---|
| 7.5 min | **1.6 min off** | 5.5 min off |
| 13.5 min | **0.0** | 2.9 |
| 16.5 min | **0.4** | 1.1 |

Mean error 1.65 min against 3.35. More important than the average: the linear predictions were *all early*. **A straight line systematically says "safe sooner than it is"** — the wrong direction of error for a safety system.

`tau`, the time constant, is the single number describing how fast a surface sheds water in given weather — the sort of thing an authority would track per site and per season.

A forecast is only offered with 5+ readings over 2+ minutes of real time, a falling curve, and R² ≥ 0.80. Otherwise nothing is shown. The bundled demo is correctly refused: its frames are numbered 0–15, which is not a timeline.

### Knowing when not to answer

The most obvious failure of a camera-based road sensor is darkness, and a confident tyre call from a frame nobody can read is the one output here that could get somebody hurt.

The naive check — "dark means night" — would break the product, because **a soaked road is dark**. That is exactly what the darkness cue measures. The real distinction is not darkness but **information loss**:

| | mean brightness | **crushed to black** |
|---|---|---|
| dry, daytime | 0.516 | **0.000** |
| **soaked, daytime** | **0.283** | **0.000** |
| dusk | 0.099 | 0.294 |
| night | 0.024 | 0.914 |

A wet road at noon is dark but fully resolved. Night clips 91% of its pixels to black, where no processing recovers the surface. Every daytime frame — including the wettest — crushes exactly 0.000.

When the light is poor the wetness number is still reported, but the recommendation becomes **HOLD — VERIFY CONDITIONS**, or **NO READING — LIGHT TOO LOW**. Verified: zero false positives across all 36 bundled and real demo frames.

A night frame still computes about 0.52, "damp" — and that number is wrong, because darkness mimics wetness. That is the entire justification for the veto.

### If something breaks

If CLIP fails to load, the pipeline falls back to the optics alone and reports `model_used: "cv-fallback"`. Every response carries that field and the dashboard always shows it. The demo degrades; it never dies.

---

## Three views

| View | Answers |
|---|---|
| **Dashboard** | What is it now, where is it going, what do I do |
| **Network** | Which point in the survey needs attention first |
| **Signals** | Every number the pipeline computed, and the arithmetic joining them |

**Network** ranks every point in a session by severity. For the dashcam set those are 20 genuinely different places — frames 8.6 seconds apart from a moving car, about 2.2 km of real UK road. That is a mobile road survey, a real product category; it is not a simulation of fixed cameras, and for a fixed-camera session the header says so instead.

**Signals** exists because the dashboard deliberately hides its working. Everything on it is measured — no filler telemetry, no placeholder gauges. Where a value is genuinely unavailable it says so rather than showing a plausible one.

## Tech stack

| Layer | What |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS, Recharts |
| Backend | FastAPI + uvicorn (Python 3.11) |
| Database | SQLite via SQLAlchemy |
| AI | Hugging Face `transformers` + PyTorch, CPU only, running locally |
| Vision | OpenCV + NumPy |

---

## Where the frames come from

Three sources. A button appears in the top bar for each one that has frames.

| Button | What it is |
|---|---|
| **Run demo** | 16 synthetic frames of a track drying, wet → dry. Shows the trend. |
| **Real dashcam** | 20 real frames from a Hugging Face dataset |
| **My photos** | Yours — drop 4+ images into `backend/data/samples_real/` |

**The synthetic frames** are rendered by `scripts/generate_samples.py`, not photographed — so the repo is self-contained and the demo is identical on every machine. They are a real perspective projection of a road, and the wet look is built from the four things water physically does.

**The real frames** come from the public Hugging Face dataset [`aap9002/UK-Road-DashCam`](https://huggingface.co/datasets/aap9002/UK-Road-DashCam) — UK roads, December 2024. Rebuild them with:

```bash
cd backend && python scripts/import_hf_dashcam.py
```

A dashcam sees the car's own bonnet across the bottom of every frame, and step 2 measures the bottom of the image because that is where the road normally is — so left alone it measures the car's paintwork, which is dark and smooth and reads as soaking wet. The import crops that away. Any real deployment needs the same idea.

All 20 real frames read DAMP / INTERMEDIATES, which is correct for a damp December road.

---

## Project layout

```
backend/
  app/
    main.py                  FastAPI app, loads CLIP once at startup
    pipeline/
      clip_classifier.py     STEP 1  CLIP + prompt ensembling
      cv_features.py         STEP 2  shine, darkness, colour, texture
      zones.py               STEP 2b 3x3 road grid
      hazards.py             zero-shot detectors, added at runtime
      forecast.py            exponential time-to-dry fit
      smoothing.py           STEP 3  averaging + hysteresis
      trend.py               STEP 4  direction, DRYING, tyre table
      orchestrator.py        runs 1 to 4 in order
    routes/                  analyze, sessions, health
  config/prompts.json        the twelve CLIP sentences
  data/samples/              16 synthetic frames
  data/samples_hf/           20 real frames from Hugging Face
  scripts/                   download_model, preflight, generate_samples,
                             import_hf_dashcam, evaluate_samples,
                             import_drying_experiment, analyse_drying
frontend/
  src/components/            Landing, FrameViewer, TrendChart, ZoneOverlay,
                             NetworkView, SignalsView, HazardWatch, ...
  src/hooks/useSession.js    all dashboard state
```

---

## Determinism

Every random seed is fixed, and smoothing state is recomputed from the database rather than held in memory. **The same input gives the same output on every run**, and running the demo twice produces an identical curve.

You can check the pipeline against ground truth:

```bash
cd backend && python scripts/evaluate_samples.py
```

For the synthetic frames we know the wetness each one was *drawn* with. The pipeline never sees that number, so the error column is a genuine measurement.

---

## Deploying

**Backend → Hugging Face Space (Docker).** CLIP needs about 1 GB of RAM, which does not fit Render's free 512 MB tier. A free Space does. Use the Docker SDK, base on `python:3.11-slim`, install `requirements.txt`, run `python scripts/download_model.py` **during the build** so the weights are baked into the image, then start `uvicorn app.main:app --host 0.0.0.0 --port 7860`.

**Frontend → Vercel.** Root directory `frontend`, preset Vite. Set `VITE_API_BASE` to your Space URL.

Deploy the Space first so you know its URL, set `VITE_API_BASE` on Vercel, then set `TRACKSENSE_CORS_ORIGINS` on the Space to the Vercel URL.

---

## Credits

- **CLIP** — [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32), MIT licence, via Hugging Face `transformers`
- **Dashcam footage** — [`aap9002/UK-Road-DashCam`](https://huggingface.co/datasets/aap9002/UK-Road-DashCam) on Hugging Face

Everything else is original work for this project.
