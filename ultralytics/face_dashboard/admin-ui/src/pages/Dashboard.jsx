import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { StatCard } from "../components/StatCard";
import FeatureControlPanel from "../components/FeatureControlPanel";

export default function Dashboard() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        setStatus(await api.status());
        setError(null);
      } catch (e) {
        setError(e.message);
      }
    };
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, []);

  if (error && !status) {
    return (
      <div className="card border-campus-danger/50">
        <p className="text-campus-danger">API offline — start backend:</p>
        <code className="mt-2 block text-sm text-slate-400">
          python -m face_dashboard.api.server
        </code>
      </div>
    );
  }

  const s = status || {};

  return (
    <div>
      <header className="mb-8">
        <h2 className="text-2xl font-bold text-white">Dashboard</h2>
        <p className="text-slate-400">Campus safety overview — live metrics & alerts</p>
      </header>

      <div className="mb-4 flex items-center gap-3">
        <span
          className={`badge ${s.running ? "bg-campus-ok/20 text-campus-ok" : "bg-slate-700 text-slate-400"}`}
        >
          {s.running ? "Camera ON" : "Camera OFF"}
        </span>
        {s.fps > 0 && <span className="text-sm text-slate-500">{s.fps} FPS</span>}
        {s.error && <span className="text-sm text-campus-danger">{s.error}</span>}
      </div>

      <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5">
        <StatCard label="Persons in frame" value={s.person_count ?? "—"} tone="accent" />
        <StatCard label="Identified faces" value={s.known_count ?? "—"} tone="ok" />
        <StatCard label="Unknown faces" value={s.unknown_count ?? "—"} tone="warn" />
        <StatCard label="Objects" value={s.object_count ?? "—"} />
        <StatCard label="Registered DB" value={s.registered_persons ?? "—"} sub="known persons" />
      </div>

      <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Fire"
          value={s.fire_detected ? "Detected" : "Clear"}
          tone={s.fire_detected ? "danger" : "ok"}
        />
        <StatCard
          label="Smoke"
          value={s.smoke_detected ? "Detected" : "Clear"}
          tone={s.smoke_detected ? "warn" : "ok"}
        />
        <StatCard
          label="Fighting"
          value={s.fighting_detected ? "Alert" : "Clear"}
          tone={s.fighting_detected ? "danger" : "ok"}
        />
        <StatCard label="Bags in frame" value={s.bags_tracked ?? "—"} tone="warn" />
      </div>

      <div className="mb-8 grid gap-4 lg:grid-cols-3">
        <StatCard label="Total entries" value={s.entries_total ?? "—"} />
        <StatCard label="Incidents" value={s.incidents_total ?? "—"} tone="danger" />
        <StatCard label="Active alerts" value={s.alerts?.length ?? 0} tone="warn" />
      </div>

      <FeatureControlPanel />


      {s.alerts?.length > 0 && (
        <div className="card mb-8 border-campus-danger/40">
          <h3 className="mb-3 font-semibold text-campus-danger">Live safety alerts</h3>
          <ul className="space-y-2">
            {s.alerts.map((a, i) => (
              <li key={i} className="rounded-lg bg-campus-danger/10 px-3 py-2 text-sm text-red-200">
                {a.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {s.live_activity?.length > 0 && (
        <div className="card mb-8">
          <h3 className="mb-3 font-semibold">Live emotion & behavior</h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {s.live_activity.map((a, i) => (
              <div key={i} className="rounded-lg border border-campus-700 bg-campus-800/50 p-3">
                <p className="font-medium text-white">{a.name}</p>
                <p className="text-sm text-slate-400">
                  {a.emotion} · {a.behavior}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <Link to="/camera" className="btn-primary">
          Open live camera
        </Link>
        <Link to="/incidents" className="btn-ghost">
          View incidents
        </Link>
      </div>
    </div>
  );
}
