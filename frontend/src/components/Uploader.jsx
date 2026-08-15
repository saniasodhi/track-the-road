/**
 * The controls: run the bundled demo, add your own frames, or drop in a video.
 *
 * Kept as a compact row rather than a panel - the dashboard is only allowed
 * three panels, one for each question it answers.
 */

import { useRef } from "react";

function Control({ children, onClick, disabled, primary }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md border px-3 py-1.5 text-[12px] transition-opacity duration-200 disabled:opacity-40 ${
        primary
          ? "border-ink bg-ink text-canvas hover:opacity-90"
          : "border-hairline bg-surface text-ink-muted hover:text-ink"
      }`}
      style={{ fontWeight: 500 }}
    >
      {children}
    </button>
  );
}

/** Short button labels for each frame source. */
const SOURCE_LABEL = {
  bundled: "Run demo",
  hf: "Real dashcam",
  real: "My photos",
};

export default function Uploader({
  onDemo, onImages, onVideo, busy, sources, onLive, live, liveStarting,
}) {
  const imageInput = useRef(null);
  const videoInput = useRef(null);

  // Only offer a source that actually has enough frames behind it. The
  // synthetic set is always first and always primary - it is the demo that
  // tells the drying story.
  const usable = ["bundled", "hf", "real"].filter((k) => sources?.[k]?.usable);
  const buttons = usable.length ? usable : ["bundled"];

  return (
    <div className="flex items-center gap-2">
      {/* Live camera first: it is the one control that makes this a tool you
          can point at something rather than a recording you watch. */}
      <button
        type="button"
        onClick={onLive}
        disabled={busy && !live}
        className={`rounded-md border px-3 py-1.5 text-[12px] transition-opacity duration-200 disabled:opacity-40 ${
          live
            ? "border-accent bg-accent text-canvas"
            : "border-hairline bg-surface text-ink-muted hover:text-ink"
        }`}
        style={{ fontWeight: 500 }}
      >
        {live ? "Stop live" : liveStarting ? "Opening…" : "Live camera"}
      </button>

      {buttons.map((key, i) => (
        <Control
          key={key}
          onClick={() => onDemo(key)}
          disabled={busy}
          primary={i === 0}
        >
          {SOURCE_LABEL[key]}
        </Control>
      ))}

      <Control onClick={() => imageInput.current?.click()} disabled={busy}>
        Add frames
      </Control>
      <input
        ref={imageInput}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          onImages(e.target.files);
          e.target.value = "";
        }}
      />

      <Control onClick={() => videoInput.current?.click()} disabled={busy}>
        Video
      </Control>
      <input
        ref={videoInput}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          onVideo(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </div>
  );
}
