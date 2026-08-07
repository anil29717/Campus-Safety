import { useCallback, useEffect, useState } from "react";
import { DoorOpen, LogOut, ShieldAlert, Video, Server, Plus, Trash2, RefreshCw } from "lucide-react";
import { api } from "../api/client";

const ROLE_META = {
  none: { label: "General", color: "bg-slate-700 text-slate-300", hint: "Monitor only" },
  entry: { label: "Entry", color: "bg-emerald-500/20 text-emerald-300", hint: "Clock-in when face seen" },
  exit: { label: "Exit", color: "bg-sky-500/20 text-sky-300", hint: "Clock-out when face seen" },
  restricted: { label: "Restricted", color: "bg-orange-500/20 text-orange-300", hint: "Safety zones + alerts" },
};

export default function Cameras() {
  const [cameras, setCameras] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);
  const [showNvr, setShowNvr] = useState(false);
  const [nvr, setNvr] = useState({
    ip: "172.16.40.2",
    username: "admin",
    password: "",
    channels: 8,
  });

  const applyList = (data) => {
    setCameras(data.cameras || []);
    setSummary(data.summary || null);
  };

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Fast path: list endpoint already syncs channels from .env (no RTSP probe)
      const data = await api.cameras();
      applyList(data);
      setInfo(
        data.cameras?.length
          ? `Loaded ${data.cameras.length} camera(s) from CP Plus / .env`
          : "No cameras in registry yet — click Fetch all cameras"
      );
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchAll = async () => {
    setLoading(true);
    setError(null);
    setInfo("Fetching all CP Plus channels from .env…");
    try {
      const result = await api.syncCpPlusCameras(false, Number(nvr.channels) || 8);
      applyList(result);
      if (!result.ok) {
        setError(result.message || "Sync failed — check CAMERA_IP in .env");
      } else {
        setInfo(result.message || `Fetched ${(result.cameras || []).length} camera(s)`);
      }
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const probeAndSync = async () => {
    setLoading(true);
    setError(null);
    setInfo("Fetching channels, then probing live RTSP (may take ~10–20s)…");
    try {
      // Always fetch first without probe so the list is never empty
      const fast = await api.syncCpPlusCameras(false, Number(nvr.channels) || 8);
      applyList(fast);
      setInfo(`${fast.message || "Synced"} — now probing live streams…`);

      const result = await api.syncCpPlusCameras(true, Number(nvr.channels) || 8);
      applyList(result);
      const live = result.probed?.length
        ? `Live channels: ${result.probed.join(", ")}`
        : "Probe finished (no live frames detected — list still kept from .env)";
      setInfo(`${result.message}. ${live}`);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    api.cameraConfig()
      .then((cfg) => {
        if (cfg?.cpplus) {
          setNvr((prev) => ({
            ...prev,
            ip: cfg.cpplus.ip || prev.ip,
            username: cfg.cpplus.username || prev.username,
            channels: cfg.cpplus.channels || prev.channels,
          }));
        }
      })
      .catch(() => {});
  }, [load]);

  const setRole = async (id, role) => {
    setLoading(true);
    try {
      await api.setCameraRole(id, role);
      await load();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const rename = async (id, name) => {
    const next = window.prompt("Camera name", name);
    if (!next || !next.trim()) return;
    setLoading(true);
    try {
      await api.updateCamera(id, { name: next.trim() });
      await load();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const remove = async (id) => {
    if (!window.confirm("Remove this camera from the registry?")) return;
    setLoading(true);
    try {
      await api.deleteCamera(id);
      await load();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const discover = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await api.discoverNvr({
        ip: nvr.ip,
        username: nvr.username,
        password: nvr.password,
        channels: Number(nvr.channels) || 8,
      });
      applyList({ cameras: result.cameras, summary });
      setShowNvr(false);
      setInfo(result.message || `Added ${result.added?.length || 0} channel(s)`);
      await load();
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  };

  const byRole = (role) => cameras.filter((c) => c.role === role);

  return (
    <div>
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Cameras & Roles</h2>
          <p className="text-slate-400">
            List all CP Plus channels and assign Entry, Exit, or Restricted roles
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className="btn-ghost flex items-center gap-2"
            onClick={load}
            disabled={loading}
          >
            <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
          <button
            type="button"
            className="btn-primary flex items-center gap-2"
            onClick={fetchAll}
            disabled={loading}
            title="Import all channels from CAMERA_IP / CAMERA_CHANNELS_COUNT in .env"
          >
            <Server size={16} />
            Fetch all cameras
          </button>
          <button
            type="button"
            className="btn-ghost flex items-center gap-2"
            onClick={probeAndSync}
            disabled={loading}
            title="Fetch from .env, then test which RTSP channels are live"
          >
            Probe live
          </button>
          <button
            type="button"
            className="btn-ghost flex items-center gap-2"
            onClick={() => setShowNvr((v) => !v)}
          >
            Manual NVR
          </button>
        </div>
      </header>

      {loading && (
        <p className="mb-4 text-sm text-campus-accent">Working… please wait</p>
      )}
      {info && !error && <p className="mb-4 text-sm text-slate-400">{info}</p>}
      {error && <p className="mb-4 text-sm text-campus-danger">{error}</p>}

      {showNvr && (
        <form onSubmit={discover} className="card mb-6 grid gap-3 md:grid-cols-4">
          <div>
            <label className="mb-1 block text-xs text-slate-400">NVR IP</label>
            <input
              className="w-full rounded border border-campus-700 bg-campus-900 px-2 py-1.5 text-sm text-white"
              value={nvr.ip}
              onChange={(e) => setNvr({ ...nvr, ip: e.target.value })}
              required
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">Username</label>
            <input
              className="w-full rounded border border-campus-700 bg-campus-900 px-2 py-1.5 text-sm text-white"
              value={nvr.username}
              onChange={(e) => setNvr({ ...nvr, username: e.target.value })}
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">Password</label>
            <input
              type="password"
              className="w-full rounded border border-campus-700 bg-campus-900 px-2 py-1.5 text-sm text-white"
              value={nvr.password}
              onChange={(e) => setNvr({ ...nvr, password: e.target.value })}
              placeholder="Leave blank only if NVR has no password"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">Channels</label>
            <div className="flex gap-2">
              <input
                type="number"
                min={1}
                max={128}
                className="w-full rounded border border-campus-700 bg-campus-900 px-2 py-1.5 text-sm text-white"
                value={nvr.channels}
                onChange={(e) => setNvr({ ...nvr, channels: e.target.value })}
              />
              <button type="submit" className="btn-primary whitespace-nowrap" disabled={loading}>
                <Plus size={16} /> Add
              </button>
            </div>
          </div>
        </form>
      )}

      <div className="mb-8 grid gap-4 md:grid-cols-3">
        <RoleCard
          title="Entry gate"
          icon={DoorOpen}
          role="entry"
          cameras={byRole("entry")}
          empty="Assign a camera as Entry — people clock in when seen here"
        />
        <RoleCard
          title="Exit gate"
          icon={LogOut}
          role="exit"
          cameras={byRole("exit")}
          empty="Assign a camera as Exit — people clock out when seen here"
        />
        <RoleCard
          title="Restricted area"
          icon={ShieldAlert}
          role="restricted"
          cameras={byRole("restricted")}
          empty="Assign cameras covering restricted / lab zones"
        />
      </div>

      <div className="card">
        <h3 className="mb-4 flex items-center gap-2 font-semibold text-white">
          <Video size={18} className="text-campus-accent" />
          All cameras ({cameras.length})
        </h3>
        {cameras.length === 0 ? (
          <p className="text-sm text-slate-500">
            No cameras yet. Click <strong className="text-slate-300">Fetch all cameras</strong>{" "}
            (uses CAMERA_IP in .env) or fill Manual NVR and click Add.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-campus-700 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2">Name</th>
                  <th className="px-3 py-2">Channel</th>
                  <th className="px-3 py-2">Source</th>
                  <th className="px-3 py-2">Role</th>
                  <th className="px-3 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((cam) => {
                  const meta = ROLE_META[cam.role] || ROLE_META.none;
                  return (
                    <tr key={cam.id} className="border-b border-campus-800/80">
                      <td className="px-3 py-3">
                        <button
                          type="button"
                          className="font-medium text-white hover:text-campus-accent"
                          onClick={() => rename(cam.id, cam.name)}
                          title="Rename"
                        >
                          {cam.name}
                        </button>
                      </td>
                      <td className="px-3 py-3 text-slate-400">
                        {cam.channel != null ? `CH ${cam.channel}` : "—"}
                      </td>
                      <td className="max-w-xs truncate px-3 py-3 text-slate-500" title={cam.source}>
                        {cam.source_label || cam.type}
                      </td>
                      <td className="px-3 py-3">
                        <select
                          value={cam.role}
                          disabled={loading}
                          onChange={(e) => setRole(cam.id, e.target.value)}
                          className={`rounded border border-campus-700 bg-campus-900 px-2 py-1 text-sm ${meta.color}`}
                        >
                          {Object.entries(ROLE_META).map(([id, m]) => (
                            <option key={id} value={id}>
                              {m.label}
                            </option>
                          ))}
                        </select>
                        <p className="mt-1 text-xs text-slate-600">{meta.hint}</p>
                      </td>
                      <td className="px-3 py-3">
                        <button
                          type="button"
                          className="rounded p-1.5 text-slate-500 hover:bg-red-400/10 hover:text-red-400"
                          onClick={() => remove(cam.id)}
                          title="Remove"
                        >
                          <Trash2 size={16} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {summary && (
        <p className="mt-4 text-xs text-slate-600">
          Tip: Start AI on Live Camera using an Entry/Exit camera to clock people in or out.
          Restricted cameras evaluate safety zones more aggressively.
        </p>
      )}
    </div>
  );
}

function RoleCard({ title, icon: Icon, cameras, empty }) {
  return (
    <div className="card">
      <h3 className="mb-3 flex items-center gap-2 font-semibold text-white">
        <Icon size={18} className="text-campus-accent" />
        {title}
      </h3>
      {cameras.length === 0 ? (
        <p className="text-sm text-slate-500">{empty}</p>
      ) : (
        <ul className="space-y-2">
          {cameras.map((c) => (
            <li
              key={c.id}
              className="flex items-center justify-between rounded-lg bg-campus-800 px-3 py-2 text-sm"
            >
              <span className="text-slate-200">{c.name}</span>
              {c.channel != null && (
                <span className="text-xs text-slate-500">CH {c.channel}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
