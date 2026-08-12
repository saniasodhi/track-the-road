/** TrackSense AI design tokens.
 *
 * Clean, light, editorial. Type and whitespace do the work.
 * Rules that are not negotiable:
 *   - font weights 400 and 500 only, never 600 or 700
 *   - every border is 1px hairline, no drop shadows
 *   - no gradients, no blur, no purple, corner radius never above 12px
 */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#FAF9F7",
        surface: "#FFFFFF",
        "surface-sunk": "#F3F1ED",
        hairline: "#E6E3DE",
        ink: "#12100E",
        "ink-muted": "#6B6660",
        "ink-faint": "#9C968E",
        accent: "#E10600",
        dry: "#1F7A54",
        drying: "#C08A00",
        damp: "#D97534",
        wet: "#C0271D",
      },
      fontFamily: {
        display: ['"Instrument Sans"', "Inter", "system-ui", "sans-serif"],
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        micro: ["10px", { lineHeight: "1.2", letterSpacing: "0.16em" }],
        body: ["13px", { lineHeight: "1.55" }],
        readout: ["46px", { lineHeight: "1", letterSpacing: "-0.045em" }],
        section: ["24px", { lineHeight: "1.15", letterSpacing: "-0.03em" }],
        hero: ["72px", { lineHeight: "1.02", letterSpacing: "-0.04em" }],
      },
      borderRadius: {
        DEFAULT: "6px",
        md: "8px",
        lg: "10px",
        xl: "12px",
      },
      transitionDuration: {
        200: "200ms",
      },
    },
  },
  plugins: [],
};
