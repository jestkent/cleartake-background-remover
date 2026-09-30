import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The dev server proxies the API so the UI runs on one origin in
    // development and in production alike, with no CORS juggling.
    proxy: {
      "/api": { target: "http://127.0.0.1:7788", changeOrigin: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
});
