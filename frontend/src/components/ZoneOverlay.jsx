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

export default function ZoneOverlay({ zones, worst }) {
  if (!zones?.length) return null;

  const measured = zones.filter((z) => z.measured);
  if (!measured.length) return null;

  return (
    <div className="pointer-events-none absolute inset-0">
      {/* preserveAspectRatio="none" lets a 0-1 viewBox map straight onto the
          frame however it is cropped. Only shapes go in here - text would be
          stretched, so labels are positioned separately below. */}
      <svg
        className="h-full w-full"
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {measured.map((z) => {
          const isWorst = z.name === worst;
          return (
            <polygon
              key={z.name}
              points={z.quad.map(([x, y]) => `${x},${y}`).join(" ")}
              fill={conditionTint(z.band, isWorst ? 0.34 : 0.18)}
              stroke={isWorst ? conditionColour(z.band) : "rgba(255,255,255,0.5)"}
              strokeWidth={isWorst ? 0.004 : 0.0015}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>

      {measured.map((z) => {
        // Centroid of the trapezoid, as a percentage of the frame.
        const cx = (z.quad.reduce((sum, p) => sum + p[0], 0) / z.quad.length) * 100;
        const cy = (z.quad.reduce((sum, p) => sum + p[1], 0) / z.quad.length) * 100;
        const isWorst = z.name === worst;
        return (
          <span
            key={z.name}
            className="num absolute -translate-x-1/2 -translate-y-1/2 rounded-[3px] px-1 py-0.5 text-[10px]"
            style={{
              left: `${cx}%`,
              top: `${cy}%`,
              backgroundColor: "rgba(255,255,255,0.86)",
              color: conditionColour(z.band),
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
