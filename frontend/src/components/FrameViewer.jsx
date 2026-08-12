/**
 * The current frame, with the condition badge in one corner and the frame
 * counter in the other.
 */

import { mediaUrl } from "../api/client";
import ConditionBadge from "./ConditionBadge";
import ZoneOverlay from "./ZoneOverlay";

export default function FrameViewer({ frame, index, total, empty, showZones, onToggleZones }) {
  const hasZones = Boolean(frame?.zones?.length);

  return (
    <div className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-hairline bg-surface">
      {frame ? (
        <img
          key={frame.id}
          src={mediaUrl(frame.image_url)}
          alt={`Track frame ${frame.frame_index + 1}`}
          className="anim-fade-in h-full w-full object-cover"
          draggable={false}
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-surface-sunk">
          <p className="max-w-[280px] text-center text-body text-ink-muted">{empty}</p>
        </div>
      )}

      {frame && showZones && hasZones && (
        <ZoneOverlay zones={frame.zones} worst={frame.zone_summary?.worst} />
      )}

      {frame && (
        <>
          <div className="absolute left-4 top-4">
            <ConditionBadge state={frame.state} size="lg" />
          </div>

          {hasZones && (
            <button
              type="button"
              onClick={onToggleZones}
              className="absolute right-4 top-4 rounded-md border border-hairline bg-surface/90 px-2.5 py-1.5 text-[11px] text-ink-muted transition-opacity duration-200 hover:text-ink"
              style={{ fontWeight: 500 }}
              title="Show or hide the per-zone road grid"
            >
              {showZones ? "Hide zones" : "Show zones"}
            </button>
          )}

          <div className="absolute bottom-4 right-4 rounded-md border border-hairline bg-surface/90 px-2.5 py-1.5">
            <span className="num text-[12px] text-ink">
              {String(index + 1).padStart(2, "0")}
              <span className="text-ink-faint"> / {String(total).padStart(2, "0")}</span>
            </span>
          </div>

          <div className="absolute bottom-4 left-4 rounded-md border border-hairline bg-surface/90 px-2.5 py-1.5">
            <span className="num text-[11px] text-ink-muted">
              {frame.latency_ms.toFixed(0)} ms
            </span>
          </div>
        </>
      )}
    </div>
  );
}
