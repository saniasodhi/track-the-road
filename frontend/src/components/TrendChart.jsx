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
 * Green curve  = where the surface is going, projected past the last reading
 *              from the fitted exponential decay. Only drawn when the session
 *              has a real timeline and the fit passed its gates, so it is
 *              absent on the bundled demo and appears on a timed capture. The
 *              marker sits where it crosses into DRY.
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

import { DRY_MAX, DAMP_MAX } from "../lib/bands";

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

  // The forecast describes a curve in seconds; the chart is indexed by frame.
  // Converting needs the average spacing between readings, which only means
  // anything when the timestamps are real - the same condition the forecast
  // itself is gated on.
  // Project from the frame being LOOKED AT, not the last one analysed. Stepping
  // back through a session then shows what the forecast said at that moment,
  // which is the claim worth checking: "at minute six it said dry at eleven".
  // The last frame of a completed dry-out has no forecast at all - there is
  // nothing left to predict - so anchoring to it would hide the feature.
  const anchor =
    frames.find((f) => f.frame_index === selected) || frames[frames.length - 1];
  const fc = anchor?.forecast;
  const span = frames.length > 1
    ? (frames[frames.length - 1].timestamp_s - frames[0].timestamp_s) / (frames.length - 1)
    : 0;

  const projection = [];
  if (fc?.amplitude != null && span > 0) {
    const tauS = fc.tau_minutes * 60;
    const elapsed = anchor.timestamp_s - frames[0].timestamp_s;
    const dryAtS = fc.dry_at_minutes != null ? fc.dry_at_minutes * 60 : null;
    // Run to the crossing, or a little past the edge of the data if there is none.
    const endS = dryAtS != null ? dryAtS + span : elapsed + span * 6;
    const steps = 26;
    for (let i = 0; i <= steps; i += 1) {
      const tS = elapsed + ((endS - elapsed) * i) / steps;
      projection.push({
        index: anchor.frame_index + (tS - elapsed) / span,
        projected: fc.baseline + fc.amplitude * Math.exp(-tS / tauS),
      });
    }
  }

  const data = frames.map((f) => ({
    index: f.frame_index,
    raw: f.wetness_raw,
    smoothed: f.wetness_smoothed,
    // Recharts draws a range area from a [low, high] pair. Null when the frame
    // had no second signal, so the shading simply stops rather than lying.
    band: f.band_low == null || f.band_high == null ? null : [f.band_low, f.band_high],
  }));

  // Projection points carry no measured values, so the measured lines simply
  // end where the data ends.
  const series = projection.length ? [...data, ...projection] : data;
  const crossingIndex =
    fc?.dry_at_minutes != null && span > 0
      ? frames[0].frame_index + (fc.dry_at_minutes * 60) / span
      : null;

  // A fixed 0-1 axis wasted most of the plot: a session sitting between 0.42
  // and 0.58 became a flat squiggle in a mostly empty box. Fit the axis to the
  // data instead, but always keep BOTH band boundaries in view - they are what
  // give the line its meaning, and a chart that scrolled them off would be
  // prettier and less useful.
  const values = [
    ...data.flatMap((d) => [d.raw, d.smoothed, ...(d.band || [])]),
    ...projection.map((p) => p.projected),
  ].filter((v) => Number.isFinite(v));
  const lo = Math.max(0, Math.min(...values, DRY_MAX) - 0.08);
  const hi = Math.min(1, Math.max(...values, DAMP_MAX) + 0.08);
  const domain = values.length ? [Number(lo.toFixed(3)), Number(hi.toFixed(3))] : [0, 1];

  // Band captions sit in the middle of whatever part of each band is visible.
  const mid = (a, b) => (Math.max(a, lo) + Math.min(b, hi)) / 2;

  return (
    <div className="h-full min-h-0 w-full">
      {data.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <p className="text-body text-ink-faint">No frames yet.</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={series} margin={{ top: 8, right: 42, bottom: 4, left: 0 }}>
            <XAxis dataKey="index" type="number" domain={["dataMin", "dataMax"]} hide />
            <YAxis domain={domain} hide />

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
            {[
              ["WET", mid(DAMP_MAX, 1)],
              ["DAMP", mid(DRY_MAX, DAMP_MAX)],
              ["DRY", mid(0, DRY_MAX)],
            ].map(([name, y]) =>
              y > lo + 0.01 && y < hi - 0.01 ? (
                <ReferenceLine
                  key={name}
                  y={y}
                  stroke="transparent"
                  label={{ value: name, position: "right", ...BAND_LABEL }}
                />
              ) : null
            )}

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
            {/* Where it is going, and when it gets there */}
            {projection.length > 0 && (
              <Line
                type="monotone"
                dataKey="projected"
                stroke="#1F7A54"
                strokeWidth={1.75}
                strokeDasharray="4 4"
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
            {crossingIndex != null && (
              <ReferenceLine
                x={crossingIndex}
                stroke="#1F7A54"
                strokeWidth={1}
                label={{
                  value: fc.dry_at_minutes != null
                    ? `dry ~${fc.dry_at_minutes.toFixed(0)} min`
                    : "",
                  position: "insideTopRight",
                  fill: "#1F7A54",
                  fontSize: 10,
                  fontFamily: "Inter, system-ui, sans-serif",
                  fontWeight: 500,
                }}
              />
            )}

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
