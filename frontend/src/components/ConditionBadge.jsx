/**
 * The condition label, and the shared colour vocabulary for the four states.
 *
 * Condition colour is only ever used as a 12% tint with solid coloured text on
 * top - never as a big block of solid colour. That keeps the screen calm and
 * keeps the colour meaningful when it does appear.
 */

export const CONDITION_COLOURS = {
  DRY: "#1F7A54",
  DRYING: "#C08A00",
  DAMP: "#D97534",
  WET: "#C0271D",
};

export function conditionColour(state) {
  return CONDITION_COLOURS[state] || "#6B6660";
}

/** Same hue at 12% alpha, for backgrounds. */
export function conditionTint(state, alpha = 0.12) {
  const hex = conditionColour(state).replace("#", "");
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default function ConditionBadge({ state, size = "md" }) {
  if (!state) return null;
  const colour = conditionColour(state);
  const pad = size === "lg" ? "px-3.5 py-2" : "px-3 py-1.5";
  const type = size === "lg" ? "text-[15px]" : "text-[12px]";

  return (
    /* key on the state so React remounts the node and the 200ms crossfade
       actually runs when the condition changes */
    <div
      key={state}
      className={`anim-condition inline-flex items-center gap-2 rounded-md border ${pad}`}
      style={{
        backgroundColor: conditionTint(state),
        borderColor: conditionTint(state, 0.28),
        color: colour,
      }}
    >
      <span
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: colour }}
      />
      <span
        className={`font-display ${type} uppercase`}
        style={{ fontWeight: 500, letterSpacing: "0.14em" }}
      >
        {state}
      </span>
    </div>
  );
}
