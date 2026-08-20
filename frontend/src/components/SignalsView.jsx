/**
 * SIGNALS — every number the pipeline actually computes, and the arithmetic
 * that connects them.
 *
 * The dashboard answers three questions and deliberately hides its working.
 * This view is the opposite: it shows the whole chain from raw pixels to tyre
 * call, in order, with nothing rounded away.
 *
 * Everything on this screen is measured. There is no filler telemetry, no
 * placeholder gauge and no number that exists to look busy — if a value is
 * here, some line of Python produced it for this frame. Where a value is not
 * available it says so rather than showing a plausible one.
 */

import { conditionColour, conditionTint } from "./ConditionBadge";

const INK = "#12100E";
const FAINT = "#9C968E";
const DRY = "#1F7A54";
const DAMP = "#D97534";
const WET = "#C0271D";

function Section({ label, right, children, className = "" }) {
  return (
    <section className={`min-h-0 ${className}`}>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="label-micro">{label}</span>
        {right && <span className="num text-[10px] text-ink-faint">{right}</span>}
      </div>
      {children}
    </section>
  );
}

/** A labelled 0–1 bar. The bar is a scaled 1px rule, so it costs nothing. */
function Bar({ label, value, colour = INK, suffix = "", decimals = 3 }) {
  const v = Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : null;
  return (
    <div className="mb-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] text-ink-muted">{label}</span>
        <span className="num text-[12px]" style={{ color: v == null ? FAINT : colour }}>
          {v == null ? "—" : `${value.toFixed(decimals)}${suffix}`}
        </span>
      </div>
      <div className="mt-1 h-px w-full bg-hairline">
        <div
          className="h-px origin-left transition-transform duration-200 ease-out"
          style={{ backgroundColor: colour, transform: `scaleX(${v ?? 0})` }}
        />
      </div>
    </div>
  );
}

/** A raw measurement with its units. Not normalised — the actual reading. */
function Metric({ label, value, unit = "", hint }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-[3px]" title={hint}>
      <span className="text-[11px] text-ink-muted">{label}</span>
      <span className="num text-[11.5px] text-ink">
        {value == null ? "—" : value}
        {value != null && unit ? <span className="text-ink-faint"> {unit}</span> : null}
      </span>
    </div>
  );
}

function Rule() {
  return <div className="my-3 h-px w-full bg-hairline" />;
}

export default function SignalsView({ frame, frameCount, health }) {
  if (!frame) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <p className="text-body text-ink-muted">
          Analyse a frame to see the pipeline working.
        </p>
      </div>
    );
  }

  const fb = frame.model_used === "cv-fallback";
  const sub = frame.cv_subscores || {};
  const raw = frame.cv_raw || {};
  const fc = frame.forecast;
  const zones = frame.zones?.filter((z) => z.measured) || [];

  const clipPart = frame.clip_wetness * 0.65;
  const opticsPart = frame.physical_wetness * 0.35;

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-x-8 gap-y-5 overflow-auto p-6 lg:grid-cols-3 lg:overflow-hidden">
      {/* ------------------------------------------------ column 1: perception */}
      <div className="flex min-h-0 flex-col">
        <Section
          label="Step 1 — CLIP"
          right={fb ? "unavailable" : "openai/clip-vit-base-patch32"}
        >
          {fb ? (
            <p className="text-[11px] leading-[1.5] text-ink-faint">
              CLIP did not load. Every reading below comes from the optics alone.
            </p>
          ) : (
            <>
              <Bar label="P(dry)" value={frame.p_dry} colour={DRY} />
              <Bar label="P(damp)" value={frame.p_damp} colour={DAMP} />
              <Bar label="P(wet)" value={frame.p_wet} colour={WET} />
              <p className="mt-2 text-[10.5px] leading-[1.5] text-ink-faint">
                Softmax over 12 prompts, ensembled into 3 prototypes.
                Collapsed as <span className="num">P(damp)×0.5 + P(wet)×1.0</span>.
              </p>
            </>
          )}
        </Section>

        <Rule />

        <Section label="Step 2 — Optics" right="no model">
          <Bar label="Shine · specular highlights" value={sub.shine} />
          <Bar label="Darkness · mean luminance" value={sub.darkness} />
          <Bar label="Colour · desaturation" value={sub.desaturation} />
          <Bar label="Texture · smoothness" value={sub.smoothness} />
          <div className="mt-2 border-t border-hairline pt-2">
            <Metric label="specular pixels" value={raw.specular_fraction != null
              ? (raw.specular_fraction * 100).toFixed(2) : null} unit="%" />
            <Metric label="mean brightness" value={raw.mean_brightness?.toFixed(3)} />
            <Metric label="mean saturation" value={raw.mean_saturation?.toFixed(3)} />
            <Metric label="Laplacian variance" value={raw.laplacian_variance?.toFixed(1)}
              hint="Measured with specular highlights masked out, so it reflects surface grain rather than reflection edges" />
          </div>
        </Section>
      </div>

      {/* ------------------------------------------------- column 2: reasoning */}
      <div className="flex min-h-0 flex-col">
        <Section label="Fusion" right="0.65 / 0.35">
          <div className="rounded-lg border border-hairline bg-surface-sunk px-3.5 py-3">
            <div className="flex items-baseline justify-between">
              <span className="text-[11px] text-ink-muted">CLIP × 0.65</span>
              <span className="num text-[12px] text-ink">{clipPart.toFixed(4)}</span>
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-[11px] text-ink-muted">optics × 0.35</span>
              <span className="num text-[12px] text-ink">{opticsPart.toFixed(4)}</span>
            </div>
            <div className="mt-2 flex items-baseline justify-between border-t border-hairline pt-2">
              <span className="text-[11px] text-ink">raw wetness</span>
              <span className="num text-[15px] text-ink" style={{ fontWeight: 500 }}>
                {frame.wetness_raw.toFixed(4)}
              </span>
            </div>
          </div>

          <div className="mt-2.5">
            <Metric label="signal gap" value={frame.disagreement?.toFixed(4)}
              hint="How far apart the two independent estimates are, smoothed" />
            <Metric label="agreement"
              value={frame.agreement != null ? `${(frame.agreement * 100).toFixed(0)}%` : null} />
            <Metric label="uncertainty band"
              value={frame.band_low != null
                ? `${frame.band_low.toFixed(3)} – ${frame.band_high.toFixed(3)}` : null} />
          </div>
        </Section>

        <Rule />

        <Section label="Step 3 — Smoothing" right="EWMA · 5 frames">
          <Metric label="raw this frame" value={frame.wetness_raw.toFixed(4)} />
          <Metric label="smoothed" value={frame.wetness_smoothed.toFixed(4)} />
          <Metric label="band (hysteresis)" value={frame.band} />
        </Section>

        <Rule />

        <Section label="Step 4 — Trend" right="least squares · 10 frames">
          <Metric label="slope" value={`${frame.slope >= 0 ? "+" : ""}${frame.slope.toFixed(4)}`}
            unit="/frame" />
          <Metric label="classification" value={frame.trend} />
          <Metric label="displayed state" value={frame.state}
            hint="DRYING is DAMP plus a falling slope — it is computed, never perceived" />
        </Section>

        <Rule />

        <Section label="Forecast" right={fc ? "exponential decay" : "not offered"}>
          {fc ? (
            <>
              <Metric label="time constant τ" value={fc.tau_minutes} unit="min"
                hint="Time to shed ~63% of the water still present" />
              <Metric label="fit quality R²" value={fc.r_squared} />
              <Metric label="fitted over" value={`${fc.points} readings / ${fc.observed_minutes} min`} />
              <Metric label="predicted dry at" value={fc.dry_at_minutes} unit="min" />
            </>
          ) : (
            <p className="text-[10.5px] leading-[1.5] text-ink-faint">
              Needs 5+ readings over 2+ minutes of real time, a falling curve and
              R² ≥ 0.80. Withheld rather than guessed.
            </p>
          )}
        </Section>
      </div>

      {/* --------------------------------------------- column 3: space & state */}
      <div className="flex min-h-0 flex-col">
        <Section label="Step 2b — Road grid" right={`${zones.length} cells`}>
          {zones.length ? (
            <div className="grid grid-cols-3 gap-1">
              {zones.map((z) => (
                <div
                  key={z.name}
                  title={`${z.name} — ${z.band}`}
                  className="rounded-[4px] border px-2 py-1.5 text-center"
                  style={{
                    borderColor: conditionTint(z.band, 0.4),
                    backgroundColor: conditionTint(z.band, 0.14),
                  }}
                >
                  <div className="num text-[13px]" style={{ color: conditionColour(z.band) }}>
                    {z.wetness.toFixed(2)}
                  </div>
                  <div className="text-[8.5px] uppercase tracking-[0.1em] text-ink-faint">
                    {z.name.toLowerCase()}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[11px] text-ink-faint">No grid for this frame.</p>
          )}
          {frame.zone_summary?.spread != null && (
            <p className="mt-2 text-[10.5px] text-ink-faint">
              spread <span className="num">{frame.zone_summary.spread.toFixed(3)}</span>
              {" · wettest "}
              <span className="num">{frame.zone_summary.worst?.toLowerCase()}</span>
            </p>
          )}
        </Section>

        <Rule />

        <Section label="Exposure" right={frame.light_level}>
          <Metric label="crushed to black"
            value={raw.crushed_fraction != null
              ? (raw.crushed_fraction * 100).toFixed(1) : null} unit="%"
            hint="Pixels below V=16, where surface detail is unrecoverable. This — not mean brightness — is how night is told from a wet road" />
          <Metric label="95th percentile"
            value={raw.p95_brightness?.toFixed(3)} />
          {!frame.light_ok && (
            <p className="mt-1.5 text-[10.5px] leading-[1.5]" style={{ color: "#E10600" }}>
              {frame.light_note}
            </p>
          )}
        </Section>

        <Rule />

        <Section label="Hazard watch" right={`${frame.hazards?.length || 0} zero-shot`}>
          {frame.hazards?.length ? (
            frame.hazards.map((h) => (
              <Bar key={h.id} label={h.label} value={h.probability}
                colour={h.triggered ? "#E10600" : INK} decimals={3} />
            ))
          ) : (
            <p className="text-[11px] text-ink-faint">No detectors registered.</p>
          )}
        </Section>

        <Rule />

        <Section label="Provenance">
          <Metric label="model" value={fb ? "cv-fallback" : "clip-vit-base-patch32"} />
          <Metric label="inference" value={frame.latency_ms.toFixed(0)} unit="ms"
            hint="Everything inside the measurement: CLIP, optics, 9 zone cells, hazard detectors, smoothing, trend and forecast" />
          <Metric label="frame" value={`${frame.frame_index + 1} of ${frameCount}`} />
          <Metric label="capture time" value={
            frame.timestamp_s != null ? `${(frame.timestamp_s / 60).toFixed(1)} min` : null} />
          <Metric label="device" value={health?.model?.loaded ? "local CPU" : "—"} />
        </Section>
      </div>
    </div>
  );
}
