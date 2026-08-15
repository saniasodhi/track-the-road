/**
 * The current frame, with the condition badge in one corner and the frame
 * counter in the other.
 *
 * Keeping the zone overlay honest
 * -------------------------------
 * The zone grid is computed in IMAGE coordinates (0-1 across the frame), but
 * the panel it is displayed in is a different shape to the photograph. Three
 * ways to reconcile that, and only one is acceptable:
 *
 *   stretch  - distorts the road. A circle of water becomes an ellipse.
 *   letterbox- honest, but throws away ~45% of the panel to empty bars.
 *   crop     - fills the panel, but hides part of the frame.
 *
 * We crop, anchored to the BOTTOM. The bottom of a forward-facing frame is the
 * nearest road surface: it is the most informative part, it is where the near
 * zone row sits, and it is the region the optics measure anyway. What gets cut
 * is sky.
 *
 * Cropping then makes the overlay wrong unless it is told about the crop — so
 * `visibleBox` below computes exactly which slice of the image survived, and
 * the overlay uses it as its SVG viewBox. Every marker lands on the pixels it
 * was measured from, whatever shape the panel happens to be.
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { mediaUrl } from "../api/client";
import ConditionBadge from "./ConditionBadge";
import ZoneOverlay from "./ZoneOverlay";

const DEFAULT_ASPECT = 3 / 2;

/**
 * Which normalised slice of the image is actually on screen, given
 * object-fit: cover anchored to the bottom.
 */
function visibleBox(containerW, containerH, aspect) {
  if (!containerW || !containerH || !aspect) return { x: 0, y: 0, w: 1, h: 1 };

  // Treat the image as 1 x (1/aspect) and scale it to cover the container.
  const naturalW = 1;
  const naturalH = 1 / aspect;
  const scale = Math.max(containerW / naturalW, containerH / naturalH);
  const renderedW = naturalW * scale;
  const renderedH = naturalH * scale;

  const offsetX = (containerW - renderedW) / 2;   // centred horizontally
  const offsetY = containerH - renderedH;         // anchored to the bottom

  return {
    x: -offsetX / renderedW,
    y: -offsetY / renderedH,
    w: containerW / renderedW,
    h: containerH / renderedH,
  };
}

export default function FrameViewer({
  frame, index, total, empty, showZones, onToggleZones,
  videoRef, live, liveStarting,
}) {
  const hasZones = Boolean(frame?.zones?.length);
  const [aspect, setAspect] = useState(DEFAULT_ASPECT);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const stageRef = useRef(null);

  // Track the panel size so the overlay's viewBox can follow it as the window
  // resizes. ResizeObserver rather than a window listener, because the panel
  // also changes size when the layout reflows around it.
  useLayoutEffect(() => {
    const node = stageRef.current;
    if (!node) return undefined;
    const measure = () =>
      setBox({ w: node.clientWidth, h: node.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  // Live video has its own aspect; fall back to the default until it reports.
  useEffect(() => {
    if (!live) return undefined;
    const v = videoRef?.current;
    if (!v) return undefined;
    const onMeta = () => {
      if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
    };
    v.addEventListener("loadedmetadata", onMeta);
    onMeta();
    return () => v.removeEventListener("loadedmetadata", onMeta);
  }, [live, videoRef]);

  const view = visibleBox(box.w, box.h, aspect);
  const showEmpty = !live && !liveStarting && !frame;

  return (
    <div
      ref={stageRef}
      className="relative min-h-[240px] flex-1 overflow-hidden rounded-xl border border-hairline bg-surface"
    >
      {/* Live feed sits above the still, so the picture never blanks between
          captures — the camera runs continuously and the analysis catches up
          underneath it. Both are anchored to the bottom so the nearest road
          is always the part that survives the crop. */}
      <video
        ref={videoRef}
        muted
        playsInline
        className="absolute inset-0 h-full w-full object-cover"
        style={{ objectPosition: "center bottom", display: live || liveStarting ? "block" : "none" }}
      />

      {!live && frame && (
        <img
          key={frame.id}
          src={mediaUrl(frame.image_url)}
          alt={`Road frame ${frame.frame_index + 1} — ${frame.state}`}
          className="anim-fade-in absolute inset-0 h-full w-full object-cover"
          style={{ objectPosition: "center bottom" }}
          draggable={false}
          onLoad={(e) => {
            const { naturalWidth: w, naturalHeight: h } = e.currentTarget;
            if (w && h) setAspect(w / h);
          }}
        />
      )}

      {frame && showZones && hasZones && (
        <ZoneOverlay
          zones={frame.zones}
          worst={frame.zone_summary?.worst}
          view={view}
        />
      )}

      {showEmpty && (
        <div className="absolute inset-0 flex items-center justify-center bg-surface-sunk px-8">
          <p className="max-w-[320px] text-center text-body text-ink-muted">{empty}</p>
        </div>
      )}

      {live && (
        <div className="absolute left-4 top-16 z-10 flex items-center gap-2 rounded-md border border-hairline bg-surface/90 px-2.5 py-1.5">
          <span
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: "#E10600" }}
          />
          <span
            className="text-micro uppercase"
            style={{ color: "#E10600", fontWeight: 500, letterSpacing: "0.16em" }}
          >
            Live
          </span>
        </div>
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
