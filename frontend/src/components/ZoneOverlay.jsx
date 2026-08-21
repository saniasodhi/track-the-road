/**
 * The 3x3 road grid, drawn over the frame where it was actually measured.
 *
 * The backend sends each cell's four corners as fractions of the image, so the
 * trapezoids land exactly on the pixels that produced the numbers - nothing
 * here re-derives the geometry.
 *
 * Cells are tinted by condition and never filled solid, so the road still
 * reads through. The wettest cell gets a heavier outline, because that is the
 * one worth looking at.
 */

import { conditionColour, conditionTint } from "./ConditionBadge";

const FULL_VIEW = { x: 0, y: 0, w: 1, h: 1 };

export default function ZoneOverlay({ zones, worst, view = FULL_VIEW }) {
  if (!zones?.length) return null;

  const measured = zones.filter((z) => z.measured);
  if (!measured.length) return null;

  /** Image coordinate -> panel percentage, accounting for the crop. */
  const toPanel = ([x, y]) => [
    ((x - view.x) / view.w) * 100,
    ((y - view.y) / view.h) * 100,
  ];

  return (
    <div className="pointer-events-none absolute inset-0">
      {/* The viewBox is the slice of the image that survived the crop, so a
          quad drawn at image coordinate (0.5, 0.9) lands on the pixel that
          actually is at (0.5, 0.9) of the photograph. preserveAspectRatio is
          off because the panel and that slice are the same shape by
          construction. Shapes only - text would be stretched, so the labels
          are positioned separately below. */}
      <svg
        className="h-full w-full"
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {measured.map((z) => {
          const isWorst = z.name === worst;
          return (
            <polygon
              key={z.name}
              points={z.quad.map(([x, y]) => `${x},${y}`).join(" ")}
              /* Only the cell that matters is filled. Tinting all nine buried
                 the photograph and, because everything was coloured, made the
                 one worth looking at indistinguishable from the rest. */
              fill={isWorst ? conditionTint(z.band, 0.22) : "transparent"}
              stroke={isWorst ? conditionColour(z.band) : "rgba(255,255,255,0.38)"}
              strokeWidth={isWorst ? 2 : 1}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>

      {measured.map((z) => {
        // Centroid of the trapezoid, mapped through the same crop as the shapes.
        const mx = z.quad.reduce((sum, p) => sum + p[0], 0) / z.quad.length;
        const my = z.quad.reduce((sum, p) => sum + p[1], 0) / z.quad.length;
        const [cx, cy] = toPanel([mx, my]);
        if (cx < -5 || cx > 105 || cy < -5 || cy > 105) return null;  // cropped out
        const isWorst = z.name === worst;
        return (
          <span
            key={z.name}
            className="num absolute -translate-x-1/2 -translate-y-1/2 rounded-[3px] px-1 text-[9.5px]"
            style={{
              left: `${cx}%`,
              top: `${cy}%`,
              // Readable over any photograph without a solid chip covering it.
              backgroundColor: isWorst ? "rgba(255,255,255,0.92)" : "rgba(255,255,255,0.7)",
              color: isWorst ? conditionColour(z.band) : "#12100E",
              fontWeight: isWorst ? 500 : 400,
            }}
          >
            {z.wetness.toFixed(2)}
          </span>
        );
      })}
    </div>
  );
}
