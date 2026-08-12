# Drop your own track photos in here

This folder is empty on purpose. It is the hand-off point between the bundled
synthetic frames and real footage.

## How to use it

1. Put **4 or more** images in this folder (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
2. Name them so that **alphabetical order is time order**:
   `01.jpg`, `02.jpg`, `03.jpg` … not `1.jpg, 2.jpg, 10.jpg`.
3. Order them **wet first, dry last**, so the demo tells a drying story.
4. Restart nothing. The `/demo` endpoint checks this folder on every call and
   uses it automatically as soon as there are 4 or more images here.

With fewer than 4 images the app keeps using the bundled synthetic set in
`../samples/`.

## What makes a good set

- **Same camera position for every shot.** The point of the product is a
  sequence of one place over time, not a gallery of different roads. A phone on
  a windowsill during and after rain is ideal.
- **8 to 16 frames.** The trend line fits over the last 10, so fewer than about
  6 means the direction never becomes confident.
- **A real transition.** Start while there is still standing water and finish
  when it is properly dry. If every frame is merely damp, the demo has no story.
- **Consistent exposure if you can.** Auto-exposure fighting you between frames
  adds noise — which the smoothing step will absorb, but it makes the raw line
  on the chart messier.

## After you add them

Check the pipeline still reads them correctly:

```bash
cd backend
python scripts/evaluate_samples.py
```

That prints every signal for every frame. What you want to see is `phys` and
`clip` both falling as the road dries.

If `phys` barely moves, your camera's brightness and texture ranges differ from
the bundled set. Adjust the anchor constants at the top of
`backend/app/pipeline/cv_features.py` — `BRIGHTNESS_DRY` / `BRIGHTNESS_WET` and
`TEXTURE_DRY_LOG` / `TEXTURE_WET_LOG` are the two pairs that matter most. They
are named and commented, and the values you need are in the table that script
prints.
