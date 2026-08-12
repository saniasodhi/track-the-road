/**
 * The landing screen. The one place in the app allowed to be dramatic.
 *
 * The animation is the actual bundled track frames cross-fading from soaked to
 * dry - not an abstract 3D scene. It shows a judge what the product does before
 * they touch anything, and it needs no extra libraries. Only opacity and
 * transform are animated.
 *
 * The dashboard's calm starts at the door.
 */

import { useEffect, useMemo, useState } from "react";
import { mediaUrl } from "../api/client";

const WORDS = ["drying", "soaking", "clearing"];
const WORD_MS = 2500;
const FRAME_MS = 1000;

/**
 * The wetness figures this pipeline actually produced for the bundled frames
 * (backend/scripts/evaluate_samples.py). Shown only for the synthetic set - if
 * you drop in your own photos the readout is hidden rather than made up.
 */
const BUNDLED_WETNESS = [
  0.66, 0.64, 0.65, 0.66, 0.66, 0.64, 0.61, 0.57,
  0.52, 0.44, 0.38, 0.30, 0.23, 0.18, 0.14, 0.11,
];

export default function Landing({ health, onStart, leaving }) {
  const [wordIndex, setWordIndex] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);

  // Eight evenly-spaced frames across the sequence: an 8 second loop.
  // Deliberately pinned to the bundled synthetic set rather than whichever
  // source happens to be active: this animation exists to show a track going
  // from soaked to dry, and only that sequence tells that story.
  const strip = useMemo(() => {
    const files =
      health?.samples?.sources?.bundled?.files || health?.samples?.files || [];
    if (files.length === 0) return [];
    const wanted = 8;
    const step = Math.max(1, Math.floor(files.length / wanted));
    return files.filter((_, i) => i % step === 0).slice(0, wanted);
  }, [health]);

  const usingBundled = Boolean(health?.samples?.sources?.bundled?.files?.length);
  const showReadout = usingBundled && strip.length > 0;

  useEffect(() => {
    const id = setInterval(() => setWordIndex((i) => (i + 1) % WORDS.length), WORD_MS);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (strip.length === 0) return undefined;
    const id = setInterval(() => setFrameIndex((i) => (i + 1) % strip.length), FRAME_MS);
    return () => clearInterval(id);
  }, [strip.length]);

  const wetness = showReadout
    ? BUNDLED_WETNESS[
        Math.min(
          BUNDLED_WETNESS.length - 1,
          Math.round((frameIndex / Math.max(strip.length - 1, 1)) * (BUNDLED_WETNESS.length - 1))
        )
      ]
    : null;

  return (
    <div
      className="relative h-screen w-screen overflow-hidden bg-ink transition-opacity duration-500"
      style={{ opacity: leaving ? 0 : 1 }}
    >
      {/* The frames themselves */}
      <div className="absolute inset-0">
        {strip.map((src, i) => (
          <img
            key={src}
            src={mediaUrl(src)}
            alt=""
            aria-hidden="true"
            data-active={i === frameIndex}
            className="landing-frame absolute inset-0 h-full w-full object-cover"
            draggable={false}
          />
        ))}
        {/* A flat scrim, not a gradient - it only has to make text legible. */}
        <div className="absolute inset-0 bg-ink/45" />
      </div>

      {/* Top rule */}
      <div className="absolute inset-x-0 top-0 flex items-center justify-between px-10 py-7">
        <span
          className="font-display text-[15px] text-canvas"
          style={{ fontWeight: 500, letterSpacing: "-0.01em" }}
        >
          TrackSense AI
        </span>
        <span
          className="text-micro uppercase text-canvas/60"
          style={{ fontWeight: 500, letterSpacing: "0.16em" }}
        >
          Surface intelligence
        </span>
      </div>

      {/* The wetness readout, counting down as the track dries */}
      {showReadout && (
        <div className="absolute right-10 top-1/2 -translate-y-1/2 text-right">
          <span
            className="text-micro uppercase text-canvas/50"
            style={{ fontWeight: 500, letterSpacing: "0.16em" }}
          >
            Wetness
          </span>
          <div
            className="num mt-2 text-[46px] text-canvas transition-opacity duration-200"
            style={{ fontWeight: 500, letterSpacing: "-0.045em" }}
          >
            {wetness.toFixed(2)}
          </div>
        </div>
      )}

      {/* Headline */}
      <div className="absolute bottom-0 left-0 px-10 pb-16">
        <h1
          className="font-display text-hero text-canvas"
          style={{ fontWeight: 500 }}
        >
          The track is{" "}
          <span
            key={WORDS[wordIndex]}
            className="anim-hero-word inline-block"
            style={{ color: "#FAF9F7" }}
          >
            {WORDS[wordIndex]}.
          </span>
        </h1>

        <p className="mt-5 max-w-[54ch] text-[15px] leading-[1.6] text-canvas/70">
          A vision-language model and classical optics, read together across a sequence of
          frames — because you cannot tell from one photograph whether a wet road is drying
          or getting worse.
        </p>

        <button
          type="button"
          onClick={onStart}
          className="mt-9 rounded-md border border-canvas/25 bg-canvas px-5 py-2.5 text-[13px] text-ink transition-opacity duration-200 hover:opacity-90"
          style={{ fontWeight: 500 }}
        >
          Start session
        </button>
      </div>
    </div>
  );
}
