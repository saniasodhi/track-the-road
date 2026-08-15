/**
 * The two independent signals, side by side.
 *
 * This is the most important thing on the screen for a technical judge: the
 * Hugging Face model and the classical computer-vision measurement are computed
 * separately and shown separately. When they agree, the answer is trustworthy.
 * When they disagree, you can see it happening.
 */

function Tile({ label, value, caption, weight }) {
  return (
    <div className="flex-1 rounded-lg border border-hairline bg-surface-sunk px-4 py-2.5">
      <div className="flex items-baseline justify-between">
        <span className="label-micro">{label}</span>
        <span className="num text-[10px] text-ink-faint">{weight}</span>
      </div>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span
          className="num text-[26px] text-ink"
          style={{ fontWeight: 500, letterSpacing: "-0.035em" }}
        >
          {(value ?? 0).toFixed(3)}
        </span>
      </div>
      {/* A 1px rule that fills to the value. Transform-only, so it is free. */}
      <div className="mt-2 h-px w-full bg-hairline">
        <div
          className="h-px origin-left bg-ink transition-transform duration-200 ease-out"
          style={{ transform: `scaleX(${Math.min(Math.max(value ?? 0, 0), 1)})` }}
        />
      </div>
      <p className="mt-1.5 truncate text-[11px] leading-[1.5] text-ink-muted">{caption}</p>
    </div>
  );
}

export default function SignalTiles({ frame }) {
  const usingFallback = frame && frame.model_used === "cv-fallback";

  return (
    <div className="flex gap-3">
      <Tile
        label="Signal 01 — CLIP"
        value={usingFallback ? 0 : frame?.clip_wetness}
        weight="×0.65"
        caption={
          usingFallback
            ? "Unavailable — running on the fallback"
            : "Hugging Face CLIP reading the scene"
        }
      />
      <Tile
        label="Signal 02 — Optics"
        value={frame?.physical_wetness}
        weight={usingFallback ? "×1.00" : "×0.35"}
        caption="Shine, darkness, colour, texture"
      />
    </div>
  );
}
