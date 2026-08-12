/**
 * One thin clickable segment per frame, tinted by condition.
 *
 * Segments that have not been analysed yet are faded, so the strip fills up in
 * front of you as results land. Clicking a segment selects that frame - and
 * that is pure local state, so every number on screen changes in the same
 * render with no animation in the way.
 */

import { conditionColour, conditionTint } from "./ConditionBadge";

export default function FrameTimeline({ frames, total, selected, onSelect }) {
  const count = Math.max(total, frames.length, 1);
  const slots = Array.from({ length: count }, (_, i) => frames[i] || null);

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="label-micro">Session timeline</span>
        <span className="num text-[10px] text-ink-faint">
          {frames.filter(Boolean).length} / {count} analysed
        </span>
      </div>

      <div className="flex gap-[3px]">
        {slots.map((frame, i) => {
          const done = Boolean(frame);
          const active = i === selected;
          return (
            <button
              key={i}
              type="button"
              onClick={() => done && onSelect(i)}
              disabled={!done}
              title={
                done
                  ? `Frame ${i + 1} — ${frame.state} — ${frame.wetness_smoothed.toFixed(3)}`
                  : `Frame ${i + 1} — not analysed yet`
              }
              aria-label={`Frame ${i + 1}${done ? `, ${frame.state}` : ", pending"}`}
              className="segment group relative h-9 flex-1 rounded-[3px] border disabled:cursor-default"
              style={{
                backgroundColor: done ? conditionTint(frame.state, 0.42) : "#F3F1ED",
                borderColor: active
                  ? conditionColour(frame?.state)
                  : done
                  ? conditionTint(frame.state, 0.5)
                  : "#E6E3DE",
                opacity: done ? 1 : 0.45,
                transform: active ? "scaleY(1.12)" : "scaleY(1)",
              }}
            >
              {active && (
                <span
                  className="absolute inset-x-0 bottom-0 h-[2px]"
                  style={{ backgroundColor: conditionColour(frame?.state) }}
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
