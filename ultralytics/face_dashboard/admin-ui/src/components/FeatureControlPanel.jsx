import { useCallback, useEffect, useState } from "react";
import { Video, VideoOff, Zap } from "lucide-react";
import { api } from "../api/client";

const LOAD_CLASS = {
  Low: "bg-slate-700 text-slate-300",
  Medium: "bg-campus-accent/20 text-campus-accent",
  High: "bg-campus-danger/20 text-campus-danger",
};

function statusPill(feature, cameraRunning) {
  if (!feature.enabled) return { label: "Off", cls: "bg-slate-700 text-slate-400" };
  if (feature.active) return { label: "Active", cls: "bg-campus-ok/20 text-campus-ok" };
  if (cameraRunning) return { label: "Ready", cls: "bg-amber-500/20 text-amber-300" };
  return { label: "Ready", cls: "bg-slate-600 text-slate-300" };
}

export default function FeatureControlPanel() {
  const [features, setFeatures] = useState([]);
  const [cameraRunning, setCameraRunning] = useState(false);
  const [cameraMode, setCameraMode] = useState("cpplus");
  const [cameraIndex, setCameraIndex] = useState(0);
  const [defaultSource, setDefaultSource] = useState("CP Plus RTSP");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const st = await api.status();
      setCameraRunning(!!st.running);
      setFeatures(st.features || []);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
    api.cameraConfig().then((cfg) => {
      if (cfg?.default_source) setDefaultSource(cfg.default_source);
      if (cfg?.is_rtsp_default) setCameraMode("cpplus");
    }).catch(() => {});
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [load]);

  const toggle = async (id, enabled) => {
    setLoading(true);
    try {
      const updated = await api.updateFeature(id, enabled);
      setFeatures(updated);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    setLoading(true);
    try {
      const source = cameraMode === "webcam" ? cameraIndex : null;
      await api.startCamera(source);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const stopCamera = async () => {
    setLoading(true);
    try {
      await api.stopCamera();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card mb-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="flex items-center gap-2 text-lg font-semibold text-white">
            <Zap className="h-5 w-5 text-campus-accent" />
            Feature Control
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            Enable only what you need — reduces CPU load. Camera runs enabled features only.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm text-slate-400">
            Source{" "}
            <select
              value={cameraMode}
              onChange={(e) => setCameraMode(e.target.value)}
              className="ml-1 rounded border border-campus-700 bg-campus-800 px-2 py-1 text-white"
            >
              <option value="cpplus">CP Plus ({defaultSource})</option>
              <option value="webcam">Webcam</option>
            </select>
          </label>
          {cameraMode === "webcam" && (
            <label className="text-sm text-slate-400">
              Index{" "}
              <input
                type="number"
                min={0}
                max={5}
                value={cameraIndex}
                onChange={(e) => setCameraIndex(Number(e.target.value))}
                className="ml-1 w-14 rounded border border-campus-700 bg-campus-800 px-2 py-1 text-white"
              />
            </label>
          )}
          <button
            type="button"
            className="btn-primary flex items-center gap-2"
            onClick={startCamera}
            disabled={loading || cameraRunning}
          >
            <Video className="h-4 w-4" />
            Start camera
          </button>
          <button
            type="button"
            className="btn-ghost flex items-center gap-2"
            onClick={stopCamera}
            disabled={loading || !cameraRunning}
          >
            <VideoOff className="h-4 w-4" />
            Stop
          </button>
          <span
            className={`badge ${cameraRunning ? "bg-campus-ok/20 text-campus-ok" : "bg-slate-700 text-slate-400"}`}
          >
            {cameraRunning ? "Camera ON" : "Camera OFF"}
          </span>
        </div>
      </div>

      {error && <p className="mb-4 text-sm text-campus-danger">{error}</p>}

      <div className="divide-y divide-campus-700 rounded-lg border border-campus-700">
        {features.map((f) => {
          const pill = statusPill(f, cameraRunning);
          return (
            <div
              key={f.id}
              className="flex flex-wrap items-center justify-between gap-4 px-4 py-3"
            >
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-white">{f.name}</span>
                  <span className={`rounded px-2 py-0.5 text-xs ${LOAD_CLASS[f.load] || LOAD_CLASS.Low}`}>
                    {f.load}
                  </span>
                  <span className={`rounded px-2 py-0.5 text-xs ${pill.cls}`}>{pill.label}</span>
                </div>
                <p className="mt-0.5 text-sm text-slate-500">{f.description}</p>
                {f.dependencies?.length > 0 && (
                  <p className="mt-1 text-xs text-slate-600">
                    Requires: {f.dependencies.join(", ")}
                  </p>
                )}
              </div>
              <label className="relative inline-flex cursor-pointer items-center">
                <input
                  type="checkbox"
                  className="peer sr-only"
                  checked={f.enabled}
                  disabled={loading}
                  onChange={(e) => toggle(f.id, e.target.checked)}
                />
                <div className="peer h-6 w-11 rounded-full bg-campus-700 after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5 after:rounded-full after:bg-white after:transition-all peer-checked:bg-campus-accent peer-checked:after:translate-x-full peer-disabled:opacity-50" />
              </label>
            </div>
          );
        })}
      </div>
    </div>
  );
}
