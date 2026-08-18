/**
 * NETWORK — every point in the session at once, ranked by how bad it is.
 *
 * The dashboard reads one frame at a time, which is the right way to explain
 * the system and the wrong way to operate it. Nobody watching a road network
 * wants to page through cameras; they want to know which stretch needs
 * attention right now.
 *
 * What a "survey point" honestly means
 * ------------------------------------
 * For the dashcam session these are 20 genuinely different places: frames
 * taken 8.6 seconds apart from a moving car, roughly 2.3 km of real UK road.
 * That is a mobile road survey - a real product category, the thing Teconer
 * and Vaisala sell mobile sensors for - not a simulation of fixed cameras.
 *
 * For a fixed-camera session the same points are one place over time. The
 * header says which, because they are different claims and only one of them
 * is true for any given session.
 */

import { mediaUrl } from "../api/client";
import { conditionColour, conditionTint } from "./ConditionBadge";

const ACCENT = "#E10600";

/** ~30 mph is the speed shown on the dashcam's own overlay. */
const SURVEY_SPEED_MS = 13.4;

export default function NetworkView({ frames, selected, onSelect, sessionName }) {
  const points = frames.filter(Boolean);

  if (!points.length) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <p className="text-body text-ink-muted">Analyse a session to see it here.</p>
      </div>
    );
  }

  // Frames served out of samples_hf came from the moving dashcam, so the
  // points are places. Anything else is one place over time.
  const isRoute = points[0]?.image_url?.includes("samples_hf");
  const spanSeconds = (points[points.length - 1]?.timestamp_s ?? 0) - (points[0]?.timestamp_s ?? 0);
  const km = (spanSeconds * SURVEY_SPEED_MS) / 1000;

  // "Needs attention" has to mean something an operator would act on. On a
  // uniformly damp road every point is damp, so flagging all of them says
  // nothing. What matters is standing water, a triggered hazard, a frame that
  // cannot be read, or a point meaningfully worse than the rest of the survey.
  const sortedWetness = [...points].map((p) => p.wetness_smoothed).sort((a, b) => a - b);
  const median = sortedWetness[Math.floor(sortedWetness.length / 2)];
  const flagged = points.filter(
    (p) =>
      p.band === "WET" ||
      // Standing water in PART of the lane is the thing a survey exists to
      // find. A point can read damp overall while its near-left is genuinely
      // wet, and that is exactly the point worth sending someone to.
      p.zone_summary?.worst_band === "WET" ||
      p.hazards_triggered?.length ||
      !p.light_ok ||
      p.wetness_smoothed >= median + 0.08
  );
  const worst = [...points].sort((a, b) => b.wetness_smoothed - a.wetness_smoothed).slice(0, 3);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 p-6">
      {/* ---------------------------------------------------------- summary */}
      <div className="flex shrink-0 flex-wrap items-end justify-between gap-4 border-b border-hairline pb-4">
        <div>
          <span className="label-micro">
            {isRoute ? "Mobile survey" : "Fixed camera"}
          </span>
          <h2
            className="mt-1.5 font-display text-[23px] text-ink"
            style={{ fontWeight: 500, letterSpacing: "-0.02em" }}
          >
            {points.length} points
            {isRoute && km > 0.05 && (
              <span className="text-ink-muted"> · about {km.toFixed(1)} km of road</span>
            )}
            {!isRoute && spanSeconds > 60 && (
              <span className="text-ink-muted">
                {" "}· {(spanSeconds / 60).toFixed(0)} minutes
              </span>
            )}
          </h2>
          <p className="mt-1 text-[11.5px] text-ink-faint">
            {isRoute
              ? "Frames from a moving vehicle, so each point is a different place."
              : "One location sampled over time."}
          </p>
        </div>

        <div className="flex items-end gap-7">
          <div>
            <span className="label-micro">Needs attention</span>
            <div
              className="num mt-1 text-[26px]"
              style={{ fontWeight: 500, color: flagged.length ? ACCENT : "#1F7A54" }}
            >
              {flagged.length}
              <span className="text-[15px] text-ink-faint"> / {points.length}</span>
            </div>
          </div>
          <div>
            <span className="label-micro">Worst point</span>
            <div className="num mt-1 text-[26px]" style={{ fontWeight: 500, color: conditionColour(worst[0]?.state) }}>
              {worst[0]?.wetness_smoothed.toFixed(2)}
              <span className="text-[13px] text-ink-faint">
                {" "}#{String((worst[0]?.frame_index ?? 0) + 1).padStart(2, "0")}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------ grid */}
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
          {points.map((p) => {
            const active = p.frame_index === selected;
            const isWorst = p.frame_index === worst[0]?.frame_index;
            const alert = p.hazards_triggered?.length > 0 || !p.light_ok;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onSelect(p.frame_index)}
                title={`Point ${p.frame_index + 1} — ${p.state} ${p.wetness_smoothed.toFixed(3)}`}
                className="group overflow-hidden rounded-lg border bg-surface text-left transition-transform duration-200"
                style={{
                  borderColor: active ? conditionColour(p.state) : "#E6E3DE",
                  transform: active ? "scale(1.02)" : "scale(1)",
                }}
              >
                <div className="relative aspect-[16/10] w-full overflow-hidden bg-surface-sunk">
                  <img
                    src={mediaUrl(p.image_url)}
                    alt=""
                    loading="lazy"
                    className="h-full w-full object-cover"
                    style={{ objectPosition: "center bottom" }}
                    draggable={false}
                  />
                  <div
                    className="absolute inset-0"
                    style={{ backgroundColor: conditionTint(p.state, 0.16) }}
                  />
                  {isWorst && (
                    <span
                      className="absolute right-1.5 top-1.5 rounded-[3px] px-1.5 py-0.5 text-[8.5px] uppercase"
                      style={{
                        backgroundColor: "rgba(255,255,255,0.9)",
                        color: conditionColour(p.state),
                        fontWeight: 500,
                        letterSpacing: "0.12em",
                      }}
                    >
                      worst
                    </span>
                  )}
                  {alert && (
                    <span
                      className="absolute left-1.5 top-1.5 inline-block h-1.5 w-1.5 rounded-full"
                      style={{ backgroundColor: ACCENT }}
                    />
                  )}
                </div>

                <div className="flex items-baseline justify-between px-2 py-1.5">
                  <span className="num text-[10px] text-ink-faint">
                    {String(p.frame_index + 1).padStart(2, "0")}
                  </span>
                  <span
                    className="num text-[13px]"
                    style={{ color: conditionColour(p.state), fontWeight: 500 }}
                  >
                    {p.wetness_smoothed.toFixed(2)}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ------------------------------------------------------- worst list */}
      <div className="shrink-0 border-t border-hairline pt-3.5">
        <span className="label-micro">Attend to these first</span>
        <div className="mt-2 flex flex-wrap gap-x-8 gap-y-1.5">
          {worst.map((p, i) => (
            <button
              key={p.id}
              type="button"
              onClick={() => onSelect(p.frame_index)}
              className="flex items-baseline gap-2.5 text-left"
            >
              <span className="num text-[11px] text-ink-faint">{i + 1}</span>
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: conditionColour(p.state) }}
              />
              <span className="text-[12.5px] text-ink">
                Point {String(p.frame_index + 1).padStart(2, "0")}
              </span>
              <span className="num text-[12px]" style={{ color: conditionColour(p.state) }}>
                {p.wetness_smoothed.toFixed(3)}
              </span>
              <span className="text-[11.5px] text-ink-muted">
                {p.zone_summary?.plain
                  ? p.zone_summary.plain.replace(/^Careful — /, "")
                  : p.recommendation}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
