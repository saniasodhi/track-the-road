/**
 * Where it is going.
 *
 * Bold line  = the smoothed score, which is what the label is based on.
 * Faint line = the raw per-frame score behind it, so you can see exactly how
 *              much noise step 3 is absorbing.
 * Shaded band = how far apart the two independent signals are. Its width is
 *              literally the distance between what CLIP says and what the
 *              optics measure - not an invented confidence interval. Narrow
 *              means both methods landed in the same place. Wide means they
 *              disagree and the reading deserves less trust. It disappears
 *              entirely in cv-fallback mode, because one signal has nothing to
 *              agree with.
 * Dashed lines = the DRY / DAMP / WET band boundaries at 0.25 and 0.55.
 *
 * No gridlines, no legend box. The band labels sit on the right where the eye
 * already is.
 */

import { useEffect, useRef } from "react";
import {
  Area,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

const BAND_LABEL = {
  fill: "#9C968E",
  fontSize: 10,
  fontFamily: "Inter, system-ui, sans-serif",
  fontWeight: 500,
  letterSpacing: "0.16em",
};

export default function TrendChart({ frames, selected, playing }) {
  // The line draws itself in once, on the first complete render. It must not
  // re-animate on every frame that arrives, or it would never settle.
  const hasAnimated = useRef(false);
  useEffect(() => {
    if (!playing && frames.length > 1) hasAnimated.current = true;
  }, [playing, frames.length]);

  const animate = !playing && frames.length > 1 && !hasAnimated.current;

  const data = frames.map((f) => ({
    index: f.frame_index,
    raw: f.wetness_raw,
    smoothed: f.wetness_smoothed,
    // Recharts draws a range area from a [low, high] pair. Null when the frame
    // had no second signal, so the shading simply stops rather than lying.
    band: f.band_low == null || f.band_high == null ? null : [f.band_low, f.band_high],
  }));

  return (
    <div className="h-full min-h-0 w-full">
      {data.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <p className="text-body text-ink-faint">No frames yet.</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 42, bottom: 4, left: 0 }}>
            <XAxis dataKey="index" hide />
            <YAxis domain={[0, 1]} hide />

            {/* Signal agreement, drawn first so it sits behind everything */}
            <Area
              dataKey="band"
              stroke="none"
              fill="#12100E"
              fillOpacity={0.08}
              isAnimationActive={false}
              connectNulls={false}
              activeDot={false}
            />

            {/* Band boundaries */}
            <ReferenceLine y={0.55} stroke="#C9C4BC" strokeDasharray="3 4" strokeWidth={1} />
            <ReferenceLine y={0.25} stroke="#C9C4BC" strokeDasharray="3 4" strokeWidth={1} />

            {/* Band names, right-hand side */}
            <ReferenceLine
              y={0.78}
              stroke="transparent"
              label={{ value: "WET", position: "right", ...BAND_LABEL }}
            />
            <ReferenceLine
              y={0.4}
              stroke="transparent"
              label={{ value: "DAMP", position: "right", ...BAND_LABEL }}
            />
            <ReferenceLine
              y={0.12}
              stroke="transparent"
              label={{ value: "DRY", position: "right", ...BAND_LABEL }}
            />

            {/* Which frame is selected */}
            {selected != null && data.some((d) => d.index === selected) && (
              <ReferenceLine x={selected} stroke="#12100E" strokeWidth={1} strokeOpacity={0.22} />
            )}

            <Line
              type="monotone"
              dataKey="raw"
              stroke="#9C968E"
              strokeWidth={1}
              strokeOpacity={0.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="smoothed"
              stroke="#12100E"
              strokeWidth={1.75}
              dot={false}
              isAnimationActive={animate}
              animationDuration={700}
              animationEasing="ease-out"
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
