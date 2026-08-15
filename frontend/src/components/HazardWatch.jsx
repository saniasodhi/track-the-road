/**
 * Zero-shot hazard detectors.
 *
 * This is the part that shows what building on a vision-language model buys
 * you: type a hazard name, press add, and a detector for it exists about a
 * tenth of a second later. No training data, no labelling, no retraining.
 *
 * Each detector reports how much more the frame looks like that hazard than
 * like an ordinary road. Anything over the trigger threshold goes red.
 */

import { useState } from "react";

const RED = "#E10600";
const INK = "#12100E";

export default function HazardWatch({ hazards, detections, onAdd, onRemove, busy, disabled }) {
  const [value, setValue] = useState("");
  const [adding, setAdding] = useState(false);

  const byId = new Map((detections || []).map((d) => [d.id, d]));
  const triggered = (detections || []).filter((d) => d.triggered);

  const submit = async (e) => {
    e.preventDefault();
    const label = value.trim();
    if (!label || adding || disabled) return;
    setAdding(true);
    try {
      await onAdd(label);
      setValue("");
    } finally {
      setAdding(false);
    }
  };

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="label-micro">Hazard watch — zero-shot</span>
        {triggered.length > 0 && (
          <span
            className="text-micro uppercase"
            style={{ color: RED, fontWeight: 500, letterSpacing: "0.16em" }}
          >
            {triggered.length} detected
          </span>
        )}
      </div>

      <form onSubmit={submit} className="mt-2.5 flex gap-2">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled || adding}
          maxLength={40}
          placeholder="Name a new hazard — e.g. black ice"
          aria-label="New hazard name"
          className="min-w-0 flex-1 rounded-md border border-hairline bg-surface px-2.5 py-1.5 text-[12px] text-ink placeholder:text-ink-faint focus:border-ink focus:outline-none disabled:opacity-45"
        />
        <button
          type="submit"
          disabled={disabled || adding || !value.trim()}
          className="shrink-0 rounded-md border border-ink bg-ink px-3 py-1.5 text-[12px] text-canvas transition-opacity duration-200 disabled:opacity-35"
          style={{ fontWeight: 500 }}
        >
          {adding ? "Teaching…" : "Add"}
        </button>
      </form>

      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {hazards.length === 0 && (
          <span className="text-[11px] text-ink-faint">
            {disabled
              ? "Needs CLIP — unavailable in fallback mode."
              : "No detectors yet."}
          </span>
        )}

        {hazards.map((h) => {
          const hit = byId.get(h.id);
          const p = hit?.probability ?? null;
          const on = Boolean(hit?.triggered);
          return (
            <span
              key={h.id}
              title={`${h.label} — ${h.prompts?.[0] || ""}${
                p == null ? "" : `\nthis frame: ${(p * 100).toFixed(0)}%`
              }`}
              className="group inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px]"
              style={{
                borderColor: on ? RED : "#E6E3DE",
                backgroundColor: on ? "rgba(225,6,0,0.10)" : "#FFFFFF",
                color: on ? RED : "#6B6660",
              }}
            >
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: on ? RED : "#C9C4BC" }}
              />
              {h.label}
              {p != null && (
                <span className="num" style={{ color: on ? RED : "#9C968E" }}>
                  {(p * 100).toFixed(0)}%
                </span>
              )}
              <button
                type="button"
                onClick={() => onRemove(h.id)}
                disabled={busy}
                aria-label={`Remove ${h.label}`}
                className="ml-0.5 text-ink-faint opacity-0 transition-opacity duration-200 hover:text-ink group-hover:opacity-100"
              >
                ×
              </button>
            </span>
          );
        })}
      </div>

      <p className="mt-2 text-[11px] leading-[1.5] text-ink-faint">
        Detectors are built from a sentence, not from training data — nothing is retrained.
      </p>
    </div>
  );
}
