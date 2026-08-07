import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { Video, LayoutGrid, MonitorPlay, Settings2 } from "lucide-react";

const ROLE_BADGE = {
  none: "bg-slate-700 text-slate-300",
  entry: "bg-emerald-500/20 text-emerald-300",
  exit: "bg-sky-500/20 text-sky-300",
  restricted: "bg-orange-500/20 text-orange-300",
};

export default function LiveCamera() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [frameSrc, setFrameSrc] = useState(null);
  const [isGridView, setIsGridView] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");

  const loadCameras = useCallback(async () => {
    try {
      await api.syncCpPlusCameras(false);
      const data = await api.cameras();
      const list = (data.cameras || []).filter((c) => c.enabled !== false);
      setCameras(list);
      setSelectedCameraId((prev) => {
        if (prev && list.some((c) => c.id === prev)) return prev;
        return list[0]?.id || "";
      });
    } catch {
      /* ignore */
    }
  }, []);

  const refresh = async () => {
    try {
      setStatus(await api.status());
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    loadCameras();
  }, [loadCameras]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, []);

  const running = status?.running;

  useEffect(() => {
    if (!running) {
      setFrameSrc(null);
      return undefined;
    }
    let alive = true;
    const poll = async () => {
      try {
        const data = await api.snapshot();
        if (alive && data?.image_base64) {
          setFrameSrc(`data:image/jpeg;base64,${data.image_base64}`);
        }
      } catch {
        /* warming up */
      }
    };
    poll();
    const id = setInterval(poll, 120);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [running, streamKey]);

  const start = async () => {
    if (!selectedCameraId) return;
    setLoading(true);
    try {
      await api.startCamera(null, selectedCameraId);
      setStreamKey((k) => k + 1);
      await refresh();
    } finally {
      setLoading(false);
    }
  };

  const stop = async () => {
    setLoading(true);
    try {
      await api.stopCamera();
      setStreamKey((k) => k + 1);
      setFrameSrc(null);
      await refresh();
    } finally {
      setLoading(false);
    }
  };

  const selected = cameras.find((c) => c.id === selectedCameraId);

  return (
    <div className="flex flex-col gap-6 lg:flex-row">
      <div className="w-full shrink-0 space-y-4 lg:w-72">
        <div className="card flex h-full flex-col">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Cameras</h3>
            <Link
              to="/cameras"
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-campus-accent hover:bg-campus-800"
              title="Assign Entry / Exit / Restricted"
            >
              <Settings2 size={14} /> Roles
            </Link>
          </div>

          <div className="max-h-[60vh] space-y-2 overflow-y-auto pr-1">
            {cameras.length === 0 ? (
              <p className="text-sm text-slate-500">
                No cameras. Open{" "}
                <Link to="/cameras" className="text-campus-accent">
                  Camera Roles
                </Link>{" "}
                to discover NVR channels.
              </p>
            ) : (
              cameras.map((cam) => (
                <div
                  key={cam.id}
                  onClick={() => setSelectedCameraId(cam.id)}
                  className={`flex cursor-pointer items-center justify-between rounded-lg p-3 transition ${
                    selectedCameraId === cam.id
                      ? "border border-campus-accent/50 bg-campus-accent/20"
                      : "border border-transparent bg-campus-800 hover:border-campus-700"
                  }`}
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <Video
                      className={`shrink-0 ${
                        selectedCameraId === cam.id ? "text-campus-accent" : "text-slate-500"
                      }`}
                      size={18}
                    />
                    <div className="truncate">
                      <p
                        className={`truncate text-sm font-medium ${
                          selectedCameraId === cam.id ? "text-white" : "text-slate-300"
                        }`}
                      >
                        {cam.name}
                      </p>
                      <p className="truncate text-xs text-slate-500">
                        {cam.channel != null ? `CH ${cam.channel}` : cam.source_label || cam.type}
                      </p>
                    </div>
                  </div>
                  <span
                    className={`ml-2 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${
                      ROLE_BADGE[cam.role] || ROLE_BADGE.none
                    }`}
                  >
                    {cam.role_label || cam.role}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="min-w-0 flex-1">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white">Live Camera</h2>
            <p className="text-slate-400">
              {isGridView
                ? "Grid view — raw streams of all cameras"
                : selected
                  ? `AI feed · ${selected.name} (${selected.role_label || selected.role})`
                  : "Select a camera and start AI feed"}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="mr-2 flex rounded-lg border border-campus-700 bg-campus-800 p-1">
              <button
                type="button"
                onClick={() => setIsGridView(false)}
                className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  !isGridView ? "bg-campus-700 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                <MonitorPlay size={16} /> Single
              </button>
              <button
                type="button"
                onClick={() => setIsGridView(true)}
                className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  isGridView ? "bg-campus-700 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                <LayoutGrid size={16} /> Grid
              </button>
            </div>

            {!isGridView && (
              <>
                <button
                  className="btn-primary"
                  onClick={start}
                  disabled={loading || running || !selectedCameraId}
                >
                  Start Feed
                </button>
                <button className="btn-ghost" onClick={stop} disabled={loading || !running}>
                  Stop
                </button>
              </>
            )}
          </div>
        </header>

        {isGridView ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {cameras.map((cam) => {
              const streamUrl = api.rawStreamByCameraId(cam.id);
              return (
                <div key={cam.id} className="card flex flex-col gap-2 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-semibold text-white">{cam.name}</span>
                    <span
                      className={`rounded px-2 py-0.5 text-[10px] uppercase ${
                        ROLE_BADGE[cam.role] || ROLE_BADGE.none
                      }`}
                    >
                      {cam.role_label || cam.role}
                    </span>
                  </div>
                  <div className="aspect-video overflow-hidden rounded bg-black">
                    {cam.source ? (
                      <img
                        src={streamUrl}
                        alt={cam.name}
                        className="h-full w-full object-contain"
                        onError={(e) => {
                          e.target.style.display = "none";
                          if (e.target.nextSibling) e.target.nextSibling.style.display = "flex";
                        }}
                      />
                    ) : null}
                    <div className="hidden h-full w-full items-center justify-center bg-campus-900 text-xs text-slate-600">
                      Stream failed or loading…
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <>
            <div className="grid gap-6 xl:grid-cols-3">
              <div className="card xl:col-span-2">
                <div className="aspect-video overflow-hidden rounded-lg bg-black">
                  {running ? (
                    frameSrc ? (
                      <img
                        src={frameSrc}
                        alt="Live feed"
                        className="h-full w-full object-contain"
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center text-slate-500">
                        Loading camera feed…
                      </div>
                    )
                  ) : (
                    <div className="flex h-full items-center justify-center text-slate-500">
                      Select a camera and click Start
                    </div>
                  )}
                </div>
                <p className="mt-2 text-sm text-slate-500">
                  {status?.camera_name && <span>{status.camera_name}</span>}
                  {status?.camera_role && status.camera_role !== "none" && (
                    <span> · {status.camera_role}</span>
                  )}
                  {status?.fps > 0 && (
                    <span>
                      {" "}
                      · {status.fps} FPS · {status.face_count} faces
                    </span>
                  )}
                </p>
              </div>

              <div className="space-y-4">
                <div className="card">
                  <h3 className="mb-2 font-semibold">In frame now</h3>
                  <ul className="space-y-1 text-sm text-slate-300">
                    <li>Persons: {status?.person_count ?? 0}</li>
                    <li>Known: {status?.known_count ?? 0}</li>
                    <li>Unknown: {status?.unknown_count ?? 0}</li>
                    <li>Objects: {status?.object_count ?? 0}</li>
                  </ul>
                </div>
                {status?.alerts?.length > 0 && (
                  <div className="card border-campus-danger/40">
                    <h3 className="mb-2 font-semibold text-campus-danger">Alerts</h3>
                    {status.alerts.map((a, i) => (
                      <p key={i} className="text-sm text-red-200">
                        {a.message}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="mt-8 grid gap-6 xl:grid-cols-2">
              <div className="card">
                <h3 className="mb-4 font-semibold">Faces detected</h3>
                <DataTable
                  columns={[
                    { key: "name", label: "Name" },
                    {
                      key: "known",
                      label: "Status",
                      render: (r) => (
                        <span className={r.known ? "text-campus-ok" : "text-campus-warn"}>
                          {r.known ? "Known" : "Unknown"}
                        </span>
                      ),
                    },
                    { key: "emotion", label: "Emotion" },
                    { key: "behavior", label: "Behavior" },
                  ]}
                  rows={status?.faces || []}
                  empty="No faces in frame"
                />
              </div>
              <div className="card">
                <h3 className="mb-4 font-semibold">Objects detected</h3>
                <DataTable
                  columns={[
                    { key: "label", label: "Object" },
                    {
                      key: "confidence",
                      label: "Confidence",
                      render: (r) => `${(r.confidence * 100).toFixed(0)}%`,
                    },
                  ]}
                  rows={status?.objects || []}
                  empty="No objects in frame"
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
