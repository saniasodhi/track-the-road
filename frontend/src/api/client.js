/**
 * Thin wrapper around the backend API.
 *
 * The base URL comes from VITE_API_BASE (see .env.example). It defaults to the
 * local backend, so the app works with no .env file at all.
 */

const BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

export const API_BASE = BASE;

/** Frame images are served by the backend, so relative paths need the base. */
export function mediaUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  return `${BASE}${path}`;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch (err) {
    throw new Error(
      `Cannot reach the backend at ${BASE}. Is it running? (uvicorn app.main:app --reload)`
    );
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  health: () => request("/api/health"),

  createSession: (name, sourceType = "images") =>
    request("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, source_type: sourceType }),
    }),

  getSession: (id) => request(`/api/sessions/${id}`),

  getFrames: (id) => request(`/api/sessions/${id}/frames`),

  /** Whole sample set in one request. The bulletproof path. */
  runDemo: (id, source) =>
    request(`/api/sessions/${id}/demo${source ? `?source=${source}` : ""}`,
            { method: "POST" }),

  /** One sample frame, so the timeline can fill as results land. */
  runDemoStep: (id, index, source) =>
    request(
      `/api/sessions/${id}/demo/step?index=${index}${source ? `&source=${source}` : ""}`,
      { method: "POST" }
    ),

  uploadFrame: (id, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/sessions/${id}/frames`, { method: "POST", body: form });
  },

  listHazards: () => request("/api/hazards"),

  addHazard: (label) =>
    request("/api/hazards", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, prompts: [] }),
    }),

  removeHazard: (id) => request(`/api/hazards/${id}`, { method: "DELETE" }),

  uploadVideo: (id, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/sessions/${id}/video`, { method: "POST", body: form });
  },
};
