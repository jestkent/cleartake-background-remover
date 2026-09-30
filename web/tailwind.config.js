/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#ECEEF1",
        surface: "#FFFFFF",
        ink: "#13171C",
        muted: "#667080",
        hairline: "#D6DAE0",
        signal: "#17694A",
        signalsoft: "#DCEAE2",
        original: "#97A1AE",
        alert: "#A63D2B",
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
