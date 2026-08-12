"""PIPELINE STEP 2b - Where on the road is it wet?

Why one number is not enough
----------------------------
A track does not dry evenly. The racing line dries first, because cars are
driving on it and pushing water off it. The edges, the kerbs and the shaded
corners hold water long after the middle looks fine. "The track is drying" can
be true at the same moment as "there is still standing water on the left at the
far end", and the second sentence is the one that actually decides whether a
driver goes out.

So as well as scoring the whole frame, we score a 3x3 grid of the road and
report the wettest cell.

The grid follows the perspective
--------------------------------
A forward-facing camera does not see the road as a rectangle - it sees a
trapezoid that narrows towards the horizon. So the grid narrows too:

      FAR    |  \\____|____/  |     narrow band, far away
      MID    |  \\____|____/  |
      NEAR   | /_____|_____\\ |     wide band, close to the camera

Rows are DEPTH, because for a camera looking at a flat road, how far down the
image a pixel sits tells you how far away it is. Columns are LATERAL position -
left, centre, right - and the centre column is roughly where the racing line is.

Every cell is measured with exactly the same optics as step 2: shine, darkness,
colour, texture.

An honest note on what the zones use
------------------------------------
Zones are measured with the CLASSICAL signal only, not CLIP. Two reasons, and
both are worth saying out loud:

  1. Cost. Nine extra CLIP passes per frame would take roughly nine times as
     long, which kills a live demo on a CPU.
  2. Accuracy. CLIP reads a whole scene. Handed a small tile of bare tarmac it
     loses the context that makes it good, and over-reads the wetness - we
     measured this.

So the zones are anchored: CLIP and the optics together set the overall LEVEL
for the frame, and the per-cell optics describe the SHAPE of the variation
around it. A cell that measures 0.1 above the frame's physical average is
reported 0.1 above the frame's fused score. That keeps the zone labels
consistent with the big readout instead of contradicting it.
"""

from __future__ import annotations

import numpy as np

from . import cv_features
from .smoothing import band_for

GRID_ROWS = 3
GRID_COLS = 3

ROW_NAMES = ("FAR", "MID", "NEAR")      # top of the road region to bottom
COL_NAMES = ("LEFT", "CENTRE", "RIGHT")

# How wide the road is, as a fraction of image width, at each end of the region
# of interest. Straight-ahead camera on a flat road: wide at your feet,
# narrowing towards the vanishing point.
#
# Err on the NARROW side. A grid that is too wide puts the left and right
# columns on the grass verge, the kerb or the white line, and then reports the
# verge as a wet patch. Too narrow only means measuring less of the road.
# Tune these if your camera is mounted differently - they are the only two
# numbers that describe the geometry.
ROAD_WIDTH_FAR = 0.26
ROAD_WIDTH_NEAR = 0.96

# A cell smaller than this many pixels on a side is not worth measuring.
MIN_CELL_PX = 12

# Only call a cell out as a problem if it is at least this much wetter than the
# frame overall. Below that it is just noise and flagging it would cry wolf.
NOTABLE_MARGIN = 0.08


def _road_width_at(v: float) -> float:
    """Road width as a fraction of image width, at depth `v` through the ROI.

    v = 0.0 is the far edge of the region, v = 1.0 is the near edge.
    """
    return ROAD_WIDTH_FAR + (ROAD_WIDTH_NEAR - ROAD_WIDTH_FAR) * v


def cell_quad(row: int, col: int, roi_fraction: float = cv_features.ROI_FRACTION) -> list[list[float]]:
    """The four corners of one cell, as fractions of the whole image (0-1).

    Returned so the frontend can draw the trapezoid over the frame exactly
    where it was measured, rather than guessing at the geometry.
    Order: top-left, top-right, bottom-right, bottom-left.
    """
    roi_top = 1.0 - roi_fraction

    v_top = row / GRID_ROWS
    v_bottom = (row + 1) / GRID_ROWS

    y_top = roi_top + roi_fraction * v_top
    y_bottom = roi_top + roi_fraction * v_bottom

    w_top = _road_width_at(v_top)
    w_bottom = _road_width_at(v_bottom)

    def x_at(width: float, index: int) -> float:
        left_edge = 0.5 - width / 2.0
        return left_edge + width * (index / GRID_COLS)

    return [
        [x_at(w_top, col), y_top],
        [x_at(w_top, col + 1), y_top],
        [x_at(w_bottom, col + 1), y_bottom],
        [x_at(w_bottom, col), y_bottom],
    ]


def analyse_zones(
    bgr: np.ndarray,
    frame_wetness: float,
    frame_physical: float,
    roi_fraction: float = cv_features.ROI_FRACTION,
) -> list[dict]:
    """Score every cell of the grid.

    frame_wetness  - the frame's final fused score (CLIP + optics)
    frame_physical - the frame's overall physical score, used as the reference
                     point so a cell's offset is measured against the same
                     yardstick it was computed with.
    """
    height, width = bgr.shape[:2]
    zones: list[dict] = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            quad = cell_quad(row, col, roi_fraction)
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]

            # Measure the axis-aligned box that contains the cell. The cells are
            # only mildly slanted, so this is close enough and far cheaper than
            # masking a polygon on every frame.
            x0 = int(max(0, min(xs) * width))
            x1 = int(min(width, max(xs) * width))
            y0 = int(max(0, min(ys) * height))
            y1 = int(min(height, max(ys) * height))

            name = f"{ROW_NAMES[row]} {COL_NAMES[col]}"

            if (x1 - x0) < MIN_CELL_PX or (y1 - y0) < MIN_CELL_PX:
                # Too small to say anything honest about.
                zones.append({
                    "row": row, "col": col, "name": name, "quad": quad,
                    "wetness": None, "band": None, "offset": None, "measured": False,
                })
                continue

            result = cv_features.measure_region(bgr[y0:y1, x0:x1])
            cell_physical = result["physical_wetness"]

            # Anchor the cell to the frame's fused level - see the note at the
            # top of this file.
            offset = cell_physical - frame_physical
            wetness = float(np.clip(frame_wetness + offset, 0.0, 1.0))

            zones.append({
                "row": row,
                "col": col,
                "name": name,
                "quad": quad,
                "wetness": round(wetness, 4),
                "band": band_for(wetness),
                "offset": round(offset, 4),
                "measured": True,
                "physical": round(cell_physical, 4),
            })

    return zones


def summarise(zones: list[dict], frame_wetness: float) -> dict:
    """Pick out the wettest cell and turn it into a sentence, if it is worth one."""
    measured = [z for z in zones if z["measured"]]
    if not measured:
        return {"worst": None, "spread": None, "note": None, "uneven": False}

    worst = max(measured, key=lambda z: z["wetness"])
    best = min(measured, key=lambda z: z["wetness"])
    spread = worst["wetness"] - best["wetness"]

    uneven = (worst["wetness"] - frame_wetness) >= NOTABLE_MARGIN
    note = None
    plain = None
    if uneven:
        where = worst["name"].lower()
        frame_band = band_for(frame_wetness)
        if worst["band"] != frame_band:
            # The interesting case: part of the road is in a worse condition
            # than the headline number suggests. This is the whole point of the
            # grid - "drying, but still standing water on the far left".
            note = (f"still {worst['band'].lower()} on the {where} "
                    f"({worst['wetness']:.2f} against {frame_wetness:.2f} overall)")
            plain = f"Careful — the {where} of the road is still {worst['band'].lower()}."
        else:
            note = (f"wetter on the {where} "
                    f"({worst['wetness']:.2f} against {frame_wetness:.2f} overall)")
            plain = f"The {where} of the road is wetter than the rest."

    return {
        "worst": worst["name"],
        "worst_wetness": worst["wetness"],
        "worst_band": worst["band"],
        "driest": best["name"],
        "driest_wetness": best["wetness"],
        "spread": round(spread, 4),
        "uneven": uneven,
        "note": note,
        "plain": plain,
    }
