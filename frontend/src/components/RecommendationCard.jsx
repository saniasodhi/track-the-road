/**
 * What to do about it.
 *
 * The action comes from a plain lookup table on (band, trend) in
 * backend/app/pipeline/trend.py - deliberately not a model, because a call that
 * decides whether a driver goes out on slicks has to be readable and arguable
 * by a human.
 */

const URGENCY_COLOUR = {
  routine: "#1F7A54",
  caution: "#C08A00",
  urgent: "#D97534",
  critical: "#E10600",
};

export default function RecommendationCard({ frame }) {
  if (!frame) {
    return (
      <div>
        <span className="label-micro">What to do</span>
        <p className="mt-3 text-body text-ink-faint">Waiting for the first frame.</p>
      </div>
    );
  }

  const colour = URGENCY_COLOUR[frame.urgency] || "#6B6660";

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="label-micro">What to do</span>
        <span
          className="text-micro uppercase"
          style={{ color: colour, fontWeight: 500, letterSpacing: "0.16em" }}
        >
          {frame.urgency}
        </span>
      </div>

      <h2
        key={frame.recommendation}
        className="anim-condition mt-3 font-display text-section text-ink"
        style={{ fontWeight: 500 }}
      >
        {frame.recommendation}
      </h2>

      <p className="mt-2.5 max-w-[46ch] text-body text-ink-muted">{frame.reason}</p>
    </div>
  );
}
