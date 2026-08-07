const BASE = "";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  status: () => request("/api/status"),
  startCamera: (source = null, cameraId = null) =>
    request("/api/camera/start", {
      method: "POST",
      body: JSON.stringify(
        cameraId
          ? { camera_id: cameraId }
          : source === null || source === undefined
            ? {}
            : typeof source === "number"
              ? { camera_index: source }
              : { source: String(source) }
      ),
    }),
  cameraConfig: () => request("/api/camera/config"),
  cameras: () => request("/api/cameras"),
  syncCpPlusCameras: (probe = false, channels = null) =>
    request("/api/cameras/sync-cpplus", {
      method: "POST",
      body: JSON.stringify({
        probe: !!probe,
        max_probe: channels != null ? Number(channels) : 8,
        ...(channels != null ? { channels: Number(channels) } : {}),
      }),
    }),
  createCamera: (body) =>
    request("/api/cameras", { method: "POST", body: JSON.stringify(body) }),
  updateCamera: (id, body) =>
    request(`/api/cameras/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  setCameraRole: (id, role) =>
    request(`/api/cameras/${id}/role`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    }),
  deleteCamera: (id) => request(`/api/cameras/${id}`, { method: "DELETE" }),
  discoverNvr: (body) =>
    request("/api/cameras/discover-nvr", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  stopCamera: () => request("/api/camera/stop", { method: "POST" }),
  streamUrl: () => `${BASE}/api/camera/stream`,
  rawStreamUrl: (source) =>
    `${BASE}/api/camera/raw_stream?source=${encodeURIComponent(source)}`,
  rawStreamByCameraId: (cameraId) =>
    `${BASE}/api/camera/raw_stream?camera_id=${encodeURIComponent(cameraId)}`,
  snapshot: () => request("/api/camera/snapshot"),
  persons: () => request("/api/persons"),
  unknownPersons: () => request("/api/unknown-persons"),
  deletePerson: (id) => request(`/api/persons/${id}`, { method: "DELETE" }),
  deleteUnknownPerson: (id) =>
    request(`/api/unknown-persons/${id}`, { method: "DELETE" }),
  registerPerson: async (name, file, faceIndex = 0) => {
    const fd = new FormData();
    fd.append("name", name);
    fd.append("file", file);
    fd.append("face_index", String(faceIndex));
    const res = await fetch(`${BASE}/api/persons`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    return res.json();
  },
  registerFromCamera: (name, faceIndex = 0) =>
    request("/api/persons/from-camera", {
      method: "POST",
      body: JSON.stringify({ name, face_index: faceIndex }),
    }),
  entries: (limit = 100) => request(`/api/entries?limit=${limit}`),
  activity: (limit = 50) => request(`/api/activity?limit=${limit}`),
  incidents: (limit = 50) => request(`/api/incidents?limit=${limit}`),
  resetIncidents: () => request("/api/incidents/reset", { method: "POST" }),
  zones: () => request("/api/zones"),
  updateZone: (id, body) =>
    request(`/api/zones/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  liveObjects: () => request("/api/objects/live"),
  liveHazards: () => request("/api/hazards/live"),
  features: () => request("/api/features"),
  updateFeatures: (body) =>
    request("/api/features", { method: "PATCH", body: JSON.stringify(body) }),
  updateFeature: (id, enabled) =>
    request(`/api/features/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled }),
    }),
};
