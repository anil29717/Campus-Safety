import { useEffect, useState } from "react";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { StatCard } from "../components/StatCard";

export default function Incidents() {
  const [data, setData] = useState(null);

  const load = () => api.incidents(100).then(setData).catch(() => {});

  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const reset = async () => {
    if (!confirm("Clear all incidents?")) return;
    await api.resetIncidents();
    load();
  };

  const countEntries = data?.counts
    ? Object.entries(data.counts).map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(" · ")
    : "";

  return (
    <div>
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Safety Incidents</h2>
          <p className="text-slate-400">Unattended bags, fire, smoke, fighting, zones, overcrowding</p>
        </div>
        <button className="btn-ghost text-campus-danger" onClick={reset}>
          Reset incidents
        </button>
      </header>

      <div className="mb-8 grid gap-4 sm:grid-cols-2">
        <StatCard label="Total incidents" value={data?.total ?? "—"} tone="danger" />
        <StatCard label="By type" value={Object.keys(data?.counts || {}).length || "—"} sub={countEntries} />
      </div>

      <div className="card">
        <h3 className="mb-4 font-semibold">Incident log</h3>
        <DataTable
          columns={[
            { key: "Time", label: "Time" },
            { key: "Type", label: "Type" },
            {
              key: "Severity",
              label: "Severity",
              render: (r) => (
                <span
                  className={
                    r.Severity === "high" ? "text-campus-danger" : "text-campus-warn"
                  }
                >
                  {r.Severity}
                </span>
              ),
            },
            { key: "Message", label: "Message" },
          ]}
          rows={data?.recent || []}
          empty="No incidents logged"
        />
      </div>
    </div>
  );
}
