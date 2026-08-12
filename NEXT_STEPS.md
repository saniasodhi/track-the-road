# NEXT STEPS — what to do when you are out of class

In priority order. Everything above the "if you have time" line matters more than everything below it.

Total for the must-do items: **about 1 hour.**

---

## Do these first

### 1. Run it yourself, end to end — 10 min

Nothing here replaces you seeing it work on your own screen.

```bash
cd backend && python scripts/preflight.py
```

Then start both halves (see SETUP.md), open <http://localhost:5173>, and press **Start session**. The dashboard analyses the bundled sequence by itself — you do not have to click anything else. Then click through the timeline segments and watch the label go WET → DRYING → DRY.

**What to check:** the whole dashboard fits on one screen with no scrolling, the numbers change the instant you click a segment, the zone grid sits on the road, and the model pill in the top right says `clip-vit-base-patch32` and not `cv-fallback`.

Also press **Real dashcam** — that runs the same pipeline over 20 real frames from a Hugging Face dataset and should report DAMP with a wet patch on the near left.

### 2. Look at the screen in greyscale — 5 min

I verified the hard design rules programmatically (no bold weights anywhere, no gradients, no shadows, no blur, no purple, no corner radius above 12px, no scrolling at 1440×900) but **I could not take a screenshot in this environment**, so nobody has actually looked at it yet. You need to.

Take a screenshot, drop it into any image editor, desaturate it. If you can still tell what is important, the type is doing its job. If it falls apart, colour was carrying the layout.

**Most likely thing you will want to change:** the big wetness number is 46px. If it does not dominate enough in greyscale, push it to 52px in `frontend/src/App.jsx` (`text-readout` in `tailwind.config.js`).

### 3. Photograph one spot drying — 20 min *(the best remaining upgrade)*

Real footage is already in: press **Real dashcam** and the pipeline runs over 20 genuine frames from a Hugging Face dataset, correctly reporting DAMP. So "does it work on real images?" is already answered.

What is still missing is a **real drying sequence**. The dashcam is on a moving car, so it shows different places, not one place changing. Only the synthetic set demonstrates the trend.

So: photograph *one spot* over time. A phone on a windowsill during and after rain — same angle, one shot every few minutes, 8–16 of them. Put them in `backend/data/samples_real/` named `01.jpg` … `16.jpg`. A third button appears automatically.

Then check the calibration:

```bash
cd backend && python scripts/evaluate_samples.py real
```

Check that `phys` falls as the road dries. If it barely moves, adjust the anchor constants at the top of `backend/app/pipeline/cv_features.py` — `BRIGHTNESS_DRY` / `BRIGHTNESS_WET` and the texture pair matter most. All named and commented.

Nothing is at risk here: adding photos never removes the other two demos.

### 4. Rehearse the explanation out loud — 15 min

The "How the AI works" section of the README is written to be read aloud. Read it once standing up.

The four sentences that do the most work:

> "There is no ready-made wet-road model, so I assembled one out of four steps instead of training one."

> "CLIP gives one opinion from scene understanding, and classical optics gives a completely independent one from physics. Both are stored, both are on screen. When two independent methods agree, that is worth much more than one method being confident."

> "You cannot tell from a single photo whether a damp road is drying or getting wetter — the pixels are identical. Drying is a direction, not a look. So the model only ever says dry, damp or wet, and DRYING is computed from the slope of the last ten frames."

> "And because there are two independent estimates, the gap between them gives us confidence for free. That shaded band is literally how far apart the two methods are — on real footage they agree to within a few percent."

> "One number for a whole track is not useful, because tracks don't dry evenly. So we score a 3×3 grid that follows the perspective. On this real footage the frame says damp overall, but the near left is still genuinely wet — and that's the sentence a race engineer actually acts on."

**Have an answer ready for "isn't this just CLIP?"** — it is not, and the two-signal tile on screen is your proof. Point at it.

---

## If you have time

### 5. Tune the zone grid to your own camera — 10 min *(only if you add your own photos)*

The 3×3 road grid assumes a forward-facing camera on a flat road, and describes that with exactly two numbers at the top of `backend/app/pipeline/zones.py`:

```
ROAD_WIDTH_FAR  = 0.26     how wide the road is at the far end of the region
ROAD_WIDTH_NEAR = 0.96     ...and at the near end
```

They are tuned for the bundled frames and they work well on the dashcam footage. If your own photos are shot from a different height or angle, turn the zone overlay on, look at where the trapezoids land, and nudge these two until the grid sits on tarmac rather than on the verge.

**Err narrow.** A grid that is too wide puts the outer columns on grass or kerb and then reports the grass as a wet patch. Too narrow only means measuring less of the road.

Also worth knowing: the FAR row is the least reliable — fewest pixels, most affected by grazing-angle sky reflection, and if the road bends, whatever lies beyond it leaks in. Trust NEAR and MID first.

### 6. Deploy it — 45 min

Details are in the README. Order matters: Hugging Face Space first (so you know the URL), then set `VITE_API_BASE` on Vercel, then set `TRACKSENSE_CORS_ORIGINS` on the Space to your Vercel URL.

The one non-obvious part: in the Dockerfile, run `python scripts/download_model.py` **during the build**, so the 600 MB of weights are baked into the image. Otherwise the Space downloads them on first request and the first visitor waits two minutes.

A live URL is a real advantage with judges. It is also the riskiest thing on this list — do not start it if you have less than an hour, and never make it your primary demo path. Localhost is the demo; the URL is a bonus.

### 7. Self-host the fonts — 15 min

`frontend/index.html` loads Instrument Sans, Inter and JetBrains Mono from Google Fonts. If the venue wifi is bad, the page falls back to system fonts — it still works and still looks reasonable, but it loses some of its character.

If you want certainty: download the three families, put the `.woff2` files in `frontend/public/fonts/`, and swap the `<link>` for local `@font-face` rules in `src/index.css`.

### 8. Show the calibration table to judges — 10 min

`python scripts/evaluate_samples.py` prints estimated wetness against the ground-truth value each synthetic frame was *rendered* with. The pipeline never sees that number, so it is a genuine error measurement.

Screenshot that table and have it open in a tab. "Mean absolute error 0.23 against ground truth, and the band sequence is exactly right" is a much stronger claim than "it looks about right", and almost nobody at a hackathon will have one.

### 9. Tune the prompts — 20 min

`backend/config/prompts.json` holds the twelve sentences. Editing them and restarting is the cheapest possible experiment. Try adding a fourth category, or wording aimed at your real photos ("a wet suburban street at night"). Re-run `evaluate_samples.py` after each change and keep whichever set separates best.

---

## Things I deliberately did not build

Listed so you can decide, not so you feel behind. **None of these are needed for a good demo.**

- **A live webcam feed.** Genuinely impressive, but browser camera permissions on a borrowed laptop behind a projector is exactly the kind of thing that fails on stage. The video upload path covers the same ground safely.
- **Multiple simultaneous sessions in the UI.** The backend fully supports it (`GET /api/sessions` lists them) but the dashboard only shows one at a time. Adding a session switcher would be a fourth panel, and the design brief said three.
- **Tests.** There is no test suite. For a 4-day hackathon build I put that time into making the pipeline deterministic and into `evaluate_samples.py`, which is a better demo artifact than a green CI badge. If you want tests, start with `smoothing.py` and `trend.py` — both are pure functions and would take about 30 minutes to cover properly.
