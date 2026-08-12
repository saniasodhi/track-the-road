import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend talks to the backend over CORS using VITE_API_BASE
// (see .env.example). No dev proxy, so the built app behaves the same way
// as the dev server when it is deployed to Vercel.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
});
