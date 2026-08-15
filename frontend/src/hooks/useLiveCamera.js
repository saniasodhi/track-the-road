/**
 * Live camera capture.
 *
 * Opens the webcam, grabs a still every CAPTURE_MS, and posts it to the same
 * /frames endpoint an uploaded photo uses — so a live frame goes through the
 * identical four-step pipeline with no special casing anywhere in the backend.
 *
 * Two things matter for a demo on a borrowed laptop:
 *  - permission denial is a normal outcome, not a crash. It surfaces as a
 *    readable message and the rest of the app keeps working.
 *  - captures never overlap. If the backend takes longer than the interval the
 *    next tick is skipped rather than queued, so a slow machine falls to a
 *    lower frame rate instead of building an unbounded backlog.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";

const CAPTURE_MS = 1400;      // gentle enough for CPU inference to keep up
const CAPTURE_WIDTH = 960;    // matches the bundled frames

export function useLiveCamera({ onFrame, onError }) {
  const [active, setActive] = useState(false);
  const [starting, setStarting] = useState(false);
  const [captured, setCaptured] = useState(0);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const busyRef = useRef(false);
  const sessionRef = useRef(null);
  const indexRef = useRef(0);

  const stop = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    busyRef.current = false;
    // Clearing the session id is what stops a capture that is already in
    // flight from delivering its result. Without this, stopping the camera and
    // immediately starting a demo could drop one stray live frame into the
    // middle of that demo.
    sessionRef.current = null;
    setActive(false);
    setStarting(false);
  }, []);

  // Never leave the camera light on when the component goes away.
  useEffect(() => stop, [stop]);

  const capture = useCallback(async () => {
    const video = videoRef.current;
    if (!video || busyRef.current) return;          // skip, do not queue
    if (!video.videoWidth) return;                  // first frames can be empty

    busyRef.current = true;
    try {
      const scale = CAPTURE_WIDTH / video.videoWidth;
      const canvas = document.createElement("canvas");
      canvas.width = CAPTURE_WIDTH;
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.9)
      );
      if (!blob || !sessionRef.current) return;

      const file = new File([blob], `live_${indexRef.current}.jpg`, { type: "image/jpeg" });
      indexRef.current += 1;
      const target = sessionRef.current;
      const frame = await api.uploadFrame(target, file);

      // The camera may have been stopped, or pointed at a new session, while
      // this request was in the air. Deliver the result only if it still
      // belongs to the session that is on screen.
      if (sessionRef.current !== target) return;
      setCaptured((n) => n + 1);
      onFrame?.(frame);
    } catch (err) {
      onError?.(err.message);
    } finally {
      busyRef.current = false;
    }
  }, [onFrame, onError]);

  const start = useCallback(async (sessionId) => {
    if (active || starting) return;
    setStarting(true);
    sessionRef.current = sessionId;
    indexRef.current = 0;
    setCaptured(0);

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser cannot open a camera. Chrome or Edge will work.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setActive(true);
      setStarting(false);
      timerRef.current = setInterval(capture, CAPTURE_MS);
      capture();                                    // don't wait a full interval
    } catch (err) {
      stop();
      const message =
        err.name === "NotAllowedError"
          ? "Camera permission was denied. Allow it in the address bar, then press Live again."
          : err.name === "NotFoundError"
          ? "No camera found on this machine."
          : err.message || "Could not open the camera.";
      onError?.(message);
    }
  }, [active, starting, capture, stop, onError]);

  return { videoRef, active, starting, captured, start, stop };
}
