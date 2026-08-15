/**
 * All dashboard state lives here: the session, its frames, which frame is
 * selected, and backend health.
 *
 * One rule drives the design: selecting a frame is pure local state, so
 * clicking a timeline segment updates every number on the screen in the same
 * render. Animation never sits between the user and the data.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";

const STEP_DELAY_MS = 90; // breathing room between frames so the fill is watchable

export function useSession() {
  const [session, setSession] = useState(null);
  const [frames, setFrames] = useState([]);
  const [selected, setSelected] = useState(0);
  const [followLive, setFollowLive] = useState(true);
  const [expectedTotal, setExpectedTotal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  // Guards the demo loop against running on after the component goes away.
  // The flag is reset on mount as well as set on unmount, because React's
  // StrictMode mounts, unmounts and remounts effects in development - without
  // the reset the very first cleanup would cancel every later run.
  const cancelled = useRef(false);
  useEffect(() => {
    cancelled.current = false;
    return () => { cancelled.current = true; };
  }, []);

  // Mirrors `followLive` so callbacks captured inside a running loop always see
  // the current value rather than the one from when the loop started.
  const followLiveRef = useRef(true);

  // ---- health ------------------------------------------------------------
  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
    } catch (err) {
      setHealth({ status: "error", notes: [err.message], model: {}, samples: {} });
    }
  }, []);

  useEffect(() => { refreshHealth(); }, [refreshHealth]);

  // ---- selection ---------------------------------------------------------
  const selectFrame = useCallback((index) => {
    setSelected(index);
    followLiveRef.current = false;
    setFollowLive(false);
  }, []);

  const pushFrame = useCallback((frame) => {
    setFrames((prev) => {
      const next = [...prev];
      next[frame.frame_index] = frame;
      return next;
    });
    setSelected((prev) => (followLiveRef.current ? frame.frame_index : prev));
  }, []);

  const goLive = useCallback(() => {
    followLiveRef.current = true;
    setFollowLive(true);
  }, []);

  const ensureSession = useCallback(async (name, sourceType) => {
    const created = await api.createSession(name, sourceType);
    setSession(created);
    setFrames([]);
    setSelected(0);
    goLive();
    return created;
  }, [goLive]);

  // ---- demo --------------------------------------------------------------
  /**
   * Runs the bundled sample frames one at a time so the timeline fills with
   * real results. If the stepped route fails for any reason we fall back to the
   * single-request /demo endpoint, which is the one that must never break.
   */
  const runDemo = useCallback(async (source) => {
    setBusy(true);
    setError(null);
    setStatus("Starting session");
    try {
      const info = source ? health?.samples?.sources?.[source] : null;
      const created = await ensureSession(
        info ? `Demo — ${info.label}` : "Demo — bundled track sequence",
        "demo"
      );
      const total = info?.count || health?.samples?.count || 0;
      setExpectedTotal(total);

      try {
        let index = 0;
        let done = false;
        while (!done && !cancelled.current) {
          setStatus(`Analysing frame ${index + 1}${total ? ` of ${total}` : ""}`);
          const step = await api.runDemoStep(created.id, index, source);
          setExpectedTotal(step.total);
          pushFrame(step.frame);
          done = step.done;
          index += 1;
          if (!done) await new Promise((r) => setTimeout(r, STEP_DELAY_MS));
        }
      } catch (stepErr) {
        // Fallback: one request, whole sequence.
        setStatus("Running batch demo");
        const result = await api.runDemo(created.id, source);
        setFrames(result.frames);
        setExpectedTotal(result.frames.length);
        setSelected(result.frames.length - 1);
      }

      setStatus("");
      goLive();
    } catch (err) {
      setError(err.message);
      setStatus("");
    } finally {
      setBusy(false);
    }
  }, [ensureSession, goLive, health, pushFrame]);

  // ---- uploads -----------------------------------------------------------
  const uploadImages = useCallback(async (fileList) => {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    setBusy(true);
    setError(null);
    try {
      let active = session;
      if (!active || active.source_type === "demo") {
        active = await ensureSession(`Upload — ${files.length} frame${files.length > 1 ? "s" : ""}`, "images");
        setExpectedTotal(files.length);
      } else {
        setExpectedTotal((prev) => prev + files.length);
      }
      goLive();
      for (let i = 0; i < files.length; i += 1) {
        setStatus(`Analysing ${files[i].name}`);
        const frame = await api.uploadFrame(active.id, files[i]);
        pushFrame(frame);
      }
      setStatus("");
    } catch (err) {
      setError(err.message);
      setStatus("");
    } finally {
      setBusy(false);
    }
  }, [ensureSession, pushFrame, session]);

  const uploadVideo = useCallback(async (file) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setStatus("Extracting frames — this takes a moment");
    try {
      const created = await ensureSession(`Video — ${file.name}`, "video");
      const result = await api.uploadVideo(created.id, file);
      setFrames(result.frames);
      setExpectedTotal(result.frames.length);
      setSelected(result.frames.length - 1);
      goLive();
      setStatus("");
    } catch (err) {
      setError(err.message);
      setStatus("");
    } finally {
      setBusy(false);
    }
  }, [ensureSession]);

  // ---- live camera ---------------------------------------------------------
  /** Open a session the live camera can push single frames into. */
  const startLiveSession = useCallback(async () => {
    setError(null);
    const created = await ensureSession("Live camera", "images");
    setExpectedTotal(0);
    return created.id;
  }, [ensureSession]);

  const present = frames.filter(Boolean);
  const current = frames[selected] || present[present.length - 1] || null;

  return {
    session,
    frames,
    present,
    current,
    selected,
    selectFrame,
    expectedTotal: Math.max(expectedTotal, present.length),
    busy,
    status,
    error,
    clearError: () => setError(null),
    health,
    refreshHealth,
    runDemo,
    uploadImages,
    uploadVideo,
    startLiveSession,
    pushFrame,
    setError,
  };
}
