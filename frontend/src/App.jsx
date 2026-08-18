/**
 * TrackSense AI dashboard.
 *
 * One screen, no scrolling at 1440px. Every panel answers exactly one of three
 * questions, and there is no fourth panel:
 *
 *   left   what is it now      the frame, the timeline, the two raw signals
 *   right  where is it going   the wetness readout and the trend chart
 *          what should I do    the tyre call
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import { useLiveCamera } from "./hooks/useLiveCamera";
import { useSession } from "./hooks/useSession";
import HazardWatch from "./components/HazardWatch";
import ConditionBadge, { conditionColour } from "./components/ConditionBadge";
import FrameTimeline from "./components/FrameTimeline";
import FrameViewer from "./components/FrameViewer";
import Landing from "./components/Landing";
import RecommendationCard from "./components/RecommendationCard";
import SignalTiles from "./components/SignalTiles";
import TrendChart from "./components/TrendChart";
import Uploader from "./components/Uploader";

const REDUCED_MOTION =
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/**
 * Short count-up on the big number.
 *
 * The animation is decoration; the value is not. requestAnimationFrame does not
 * run while a tab is hidden or not compositing, so a timer guarantees the
 * readout lands on the true value regardless of whether a single frame was ever
 * painted. The number on screen is never allowed to lag behind the data.
 */
function useCountUp(target, duration = 320) {
  const [value, setValue] = useState(target ?? 0);
  const from = useRef(target ?? 0);

  useEffect(() => {
    if (target == null) return undefined;

    const settle = () => {
      from.current = target;
      setValue(target);
    };

    const canAnimate =
      !REDUCED_MOTION &&
      typeof document !== "undefined" &&
      document.visibilityState === "visible" &&
      from.current !== target;

    if (!canAnimate) {
      settle();
      return undefined;
    }

    const start = performance.now();
    const origin = from.current;
    let raf = requestAnimationFrame(function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - t) * (1 - t);
      setValue(origin + (target - origin) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
      else settle();
    });

    // Safety net: whatever happened to the frame callbacks, show the truth.
    const guard = setTimeout(settle, duration + 80);

    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(guard);
    };
  }, [target, duration]);

  return target == null ? null : value;
}

const HEALTH_DOT = { ok: "#1F7A54", degraded: "#C08A00", error: "#E10600" };

function TopBar({ session, health, modelUsed, children }) {
  const dot = HEALTH_DOT[health?.status] || "#9C968E";
  const model = modelUsed || health?.model?.model_id || "loading…";
  const short = model.includes("/") ? model.split("/").pop() : model;

  return (
    <header className="flex shrink-0 flex-wrap items-center justify-between gap-y-2 border-b border-hairline px-6 py-3.5">
      <div className="flex items-baseline gap-4">
        <span
          className="font-display text-[15px] text-ink"
          style={{ fontWeight: 500, letterSpacing: "-0.01em" }}
        >
          TrackSense AI
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {children}
        <div className="flex items-center gap-2.5">
          {/* The model pill links to the actual model on Hugging Face, so
              anyone can check exactly what is running. */}
          <a
            href={
              model.includes("/")
                ? `https://huggingface.co/${model}`
                : "https://huggingface.co/openai/clip-vit-base-patch32"
            }
            target="_blank"
            rel="noreferrer"
            className="num rounded-full border border-hairline px-2.5 py-1 text-[10px] text-ink-muted transition-colors duration-200 hover:border-ink hover:text-ink"
            title={`${model} — open on Hugging Face`}
          >
            {short}
          </a>
          <span
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: dot }}
            title={
              health
                ? `Backend ${health.status}${
                    health.notes?.length ? ` — ${health.notes.join(" ")}` : ""
                  }`
                : "Checking backend…"
            }
          />
        </div>
      </div>
    </header>
  );
}

export default function App() {
  const s = useSession();
  const [screen, setScreen] = useState("landing");
  const [leaving, setLeaving] = useState(false);
  // The road grid is on by default - it is the difference between "the track is
  // wet" and "the track is drying but the near left is still standing in water".
  const [showZones, setShowZones] = useState(true);

  // ---- zero-shot hazard detectors ------------------------------------------
  const [hazards, setHazards] = useState([]);
  const [hazardBusy, setHazardBusy] = useState(false);

  const refreshHazards = useCallback(async () => {
    try {
      const res = await api.listHazards();
      setHazards(res.hazards || []);
    } catch {
      setHazards([]);
    }
  }, []);

  useEffect(() => { refreshHazards(); }, [refreshHazards]);

  const addHazard = useCallback(async (label) => {
    setHazardBusy(true);
    try {
      const res = await api.addHazard(label);
      setHazards((prev) => [...prev, res.hazard]);
      // Score it against the frames already on screen, so the detector you
      // just created has a reading immediately instead of waiting for the
      // next frame to arrive.
      await s.rescoreHazards();
    } catch (err) {
      s.setError(err.message);
    } finally {
      setHazardBusy(false);
    }
  }, [s]);

  const removeHazard = useCallback(async (id) => {
    setHazardBusy(true);
    try {
      await api.removeHazard(id);
      setHazards((prev) => prev.filter((h) => h.id !== id));
    } catch (err) {
      s.setError(err.message);
    } finally {
      setHazardBusy(false);
    }
  }, [s]);

  // ---- live camera ---------------------------------------------------------
  const cam = useLiveCamera({
    onFrame: s.pushFrame,
    onError: s.setError,
  });

  const toggleLive = useCallback(async () => {
    if (cam.active || cam.starting) {
      cam.stop();
      return;
    }
    try {
      const sessionId = await s.startLiveSession();
      await cam.start(sessionId);
    } catch (err) {
      s.setError(err.message);
    }
  }, [cam, s]);

  const current = s.current;
  const displayed = useCountUp(current ? current.wetness_smoothed : null);

  const startSession = () => {
    setLeaving(true);
    setTimeout(() => setScreen("dashboard"), REDUCED_MOTION ? 0 : 420);
  };

  // The dashboard analyses the bundled sequence the moment it opens, so it
  // fills itself in front of an audience with nothing to click. Runs once;
  // the buttons in the top bar still drive everything else.
  const autoStarted = useRef(false);
  useEffect(() => {
    if (screen !== "dashboard" || autoStarted.current) return;
    if (!s.health || s.busy || cam.active || cam.starting) return;
    autoStarted.current = true;
    s.runDemo("bundled");
  }, [screen, s.health, s.busy, s.runDemo, cam.active, cam.starting]);

  if (screen === "landing") {
    return <Landing health={s.health} onStart={startSession} leaving={leaving} />;
  }

  const slopeLabel =
    current == null
      ? "—"
      : `${current.slope >= 0 ? "+" : ""}${current.slope.toFixed(3)}`;

  // Agreement is only worth colouring when it is genuinely poor - otherwise it
  // reads as just another neutral number, which is what it usually is.
  const agreementColour =
    current?.agreement == null
      ? "#9C968E"
      : current.agreement < 0.5
      ? "#D97534"
      : "#12100E";

  return (
    /* One locked screen on a laptop; below that it reflows and is allowed to
       scroll, because a squeezed-but-unscrollable layout is worse than a tall
       one. */
    <div className="anim-fade-in flex min-h-screen flex-col bg-canvas lg:h-screen lg:overflow-hidden">
      <TopBar session={s.session} health={s.health} modelUsed={current?.model_used}>
        <Uploader
          onDemo={(src) => { cam.stop(); s.runDemo(src); }}
          onImages={(files) => { cam.stop(); s.uploadImages(files); }}
          onVideo={(file) => { cam.stop(); s.uploadVideo(file); }}
          busy={s.busy}
          sources={s.health?.samples?.sources}
          onLive={toggleLive}
          live={cam.active}
          liveStarting={cam.starting}
        />
      </TopBar>

      {(s.error || s.status || s.health?.status === "degraded") && (
        <div className="flex shrink-0 items-center gap-3 border-b border-hairline bg-surface-sunk px-6 py-2">
          {s.error ? (
            <>
              <span
                className="text-micro uppercase"
                style={{ color: "#E10600", fontWeight: 500, letterSpacing: "0.16em" }}
              >
                Error
              </span>
              <span className="text-[12px] text-ink-muted">{s.error}</span>
              <button
                type="button"
                onClick={s.clearError}
                className="ml-auto text-[11px] text-ink-faint hover:text-ink"
              >
                dismiss
              </button>
            </>
          ) : s.status ? (
            <>
              <span className="label-micro">Working</span>
              <span className="num text-[12px] text-ink-muted">{s.status}</span>
            </>
          ) : (
            <>
              <span
                className="text-micro uppercase"
                style={{ color: "#C08A00", fontWeight: 500, letterSpacing: "0.16em" }}
              >
                Degraded
              </span>
              <span className="text-[12px] text-ink-muted">
                {s.health.notes?.[0] || "Running with reduced capability."}
              </span>
            </>
          )}
        </div>
      )}

      <main className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[1fr_minmax(380px,38%)]">
        {/* ---------------------------------------------- what is it now --- */}
        <section className="flex min-h-0 flex-col gap-3.5 border-b border-hairline p-5 lg:border-b-0 lg:border-r">
          <FrameViewer
            frame={current}
            index={s.selected}
            total={s.expectedTotal || 1}
            showZones={showZones}
            onToggleZones={() => setShowZones((v) => !v)}
            videoRef={cam.videoRef}
            live={cam.active}
            liveStarting={cam.starting}
            empty={
              s.busy
                ? "Analysing…"
                : "Press Live camera to point this at a real surface, or Run demo for the bundled sequence."
            }
          />

          <FrameTimeline
            frames={s.frames}
            total={s.expectedTotal}
            selected={s.selected}
            onSelect={s.selectFrame}
          />

          <SignalTiles frame={current} />

          <div className="border-t border-hairline pt-3.5">
            <HazardWatch
              hazards={hazards}
              detections={current?.hazards}
              onAdd={addHazard}
              onRemove={removeHazard}
              busy={hazardBusy}
              disabled={s.health?.model?.loaded === false}
            />
          </div>
        </section>

        {/* ------------------------------- where is it going / what to do --- */}
        <section className="flex min-h-0 flex-col">
          {/* 1. the reading — led by a plain sentence, because that is what a
                 person needs. The numbers underneath are the evidence for it. */}
          <div className="shrink-0 px-6 pb-5 pt-6">
            <div className="flex items-baseline justify-between">
              <span className="label-micro">Right now</span>
              <ConditionBadge state={current?.state} />
            </div>

            {/* When the frame cannot be read, say so above everything else -
                before the number, so nobody acts on it by accident. */}
            {current && !current.light_ok && (
              <div
                key={current.light_level}
                className="anim-condition mt-3 flex items-start gap-2.5 rounded-lg border px-3 py-2.5"
                style={{
                  borderColor: "rgba(225,6,0,0.28)",
                  backgroundColor: "rgba(225,6,0,0.07)",
                }}
              >
                <span
                  className="mt-[6px] inline-block h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ backgroundColor: "#E10600" }}
                />
                <div>
                  <span
                    className="text-micro uppercase"
                    style={{ color: "#E10600", fontWeight: 500, letterSpacing: "0.16em" }}
                  >
                    {current.light_level === "dark" ? "No reading" : "Low light"}
                  </span>
                  <p className="mt-1 text-[12.5px] leading-[1.5] text-ink-muted">
                    {current.light_note}
                  </p>
                </div>
              </div>
            )}

            <p
              key={current?.plain}
              className="anim-condition mt-3 font-display text-[21px] leading-[1.25] text-ink"
              style={{ fontWeight: 500, letterSpacing: "-0.02em" }}
            >
              {current
                ? current.plain
                : "Waiting for the first frame."}
            </p>

            {/* Time-to-dry, in minutes. Only present when the session has a
                real timeline and the curve actually supports a forecast - see
                pipeline/forecast.py. Given its own line because "dry in about
                nine minutes" is the most actionable thing on the screen. */}
            {current?.forecast?.sentence && (
              <p
                key={current.forecast.sentence}
                className="anim-condition mt-2.5 flex items-baseline gap-2 font-display text-[17px] text-ink"
                style={{ fontWeight: 500, letterSpacing: "-0.015em" }}
              >
                <span style={{ color: "#1F7A54" }}>{current.forecast.sentence}</span>
                <span
                  className="num text-[10px] text-ink-faint"
                  title={`Exponential decay fit: time constant ${current.forecast.tau_minutes} min, `
                    + `R² ${current.forecast.r_squared} over ${current.forecast.points} readings`}
                >
                  τ {current.forecast.tau_minutes}m · R² {current.forecast.r_squared}
                </span>
              </p>
            )}

            {/* The road is not uniform. When one part is meaningfully worse
                than the average, say so in words a person can act on. */}
            {current?.zone_summary?.plain && (
              <p
                key={current.zone_summary.plain}
                className="anim-condition mt-2 flex items-start gap-2 text-body"
                style={{ color: conditionColour(current.zone_summary.worst_band) }}
              >
                <span
                  className="mt-[7px] inline-block h-1 w-1 shrink-0 rounded-full"
                  style={{
                    backgroundColor: conditionColour(current.zone_summary.worst_band),
                  }}
                />
                {current.zone_summary.plain}
              </p>
            )}

            <div className="mt-4 flex items-end gap-5">
              <span
                className="num text-readout text-ink transition-opacity duration-200"
                style={{ fontWeight: 500, opacity: current && !current.light_ok ? 0.4 : 1 }}
                title={current && !current.light_ok
                  ? "Shown for reference only - the light was too poor to trust this"
                  : undefined}
              >
                {displayed == null ? "—" : displayed.toFixed(3)}
              </span>

              <div className="pb-1.5">
                <span className="label-micro">Change / frame</span>
                <div
                  className="num mt-1 text-[15px]"
                  style={{
                    color: current ? conditionColour(current.state) : "#6B6660",
                  }}
                >
                  {slopeLabel}
                </div>
              </div>

              {/* How far apart the two independent signals are. This is the
                  width of the shaded band on the chart, not a separate claim. */}
              <div className="pb-1.5">
                <span className="label-micro">AI confidence</span>
                <div
                  className="num mt-1 text-[15px]"
                  title={
                    current?.agreement == null
                      ? "Only one signal available — nothing to compare against"
                      : `CLIP and the optics are ${current.disagreement.toFixed(3)} apart`
                  }
                  style={{ color: agreementColour }}
                >
                  {current?.agreement == null
                    ? "single"
                    : `${Math.round(current.agreement * 100)}%`}
                </div>
              </div>
            </div>

            {/* Say what the scale actually means, so 0.109 is readable by
                someone seeing this for the first time. */}
            <p className="mt-2.5 text-[11px] text-ink-faint">
              0.00 bone dry &nbsp;·&nbsp; 0.25 damp &nbsp;·&nbsp; 0.55 standing water
            </p>
          </div>

          {/* 2. where it is going */}
          <div className="flex min-h-0 flex-1 flex-col border-t border-hairline px-6 py-5">
            <div className="mb-2 flex shrink-0 items-baseline justify-between">
              <span className="label-micro">Where it is heading</span>
              <span className="label-micro">
                {current
                  ? { IMPROVING: "Drying", DETERIORATING: "Getting wetter", STABLE: "Steady" }[
                      current.trend
                    ]
                  : "—"}
              </span>
            </div>
            <div className="min-h-0 flex-1">
              <TrendChart frames={s.present} selected={s.selected} playing={s.busy} />
            </div>
          </div>

          {/* 3. what to do */}
          <div className="shrink-0 border-t border-hairline px-6 pb-6 pt-5">
            <RecommendationCard frame={current} />
          </div>
        </section>
      </main>
    </div>
  );
}
