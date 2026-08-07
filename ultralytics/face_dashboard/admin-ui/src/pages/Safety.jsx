import { useEffect, useState } from "react";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";

export default function Safety() {
  const [zones, setZones] = useState([]);

  const load = () => api.zones().then(setZones).catch(() => {});

  useEffect(() => {
    load();
  }, []);

  const toggle = async (zone) => {
    await api.updateZone(zone.id, { enabled: !zone.enabled });
    load();
  };

  const setMax = async (zone, val) => {
    await api.updateZone(zone.id, { max_persons: val });
    load();
  };

  return (
    <div>
      <header className="mb-8">
        <h2 className="text-2xl font-bold text-white">Safety Zones</h2>
        <p className="text-slate-400">
          Restricted areas (unknown faces) and crowd limits — orange & yellow overlays on camera
        </p>
      </header>

      <div className="mb-8 grid gap-4 lg:grid-cols-2">
        {zones.map((z) => (
          <div key={z.id} className="card">
            <div className="mb-3 flex items-start justify-between">
              <div>
                <h3 className="font-semibold text-white">{z.name}</h3>
                <p className="text-sm text-slate-500 capitalize">{z.zone_type} zone</p>
              </div>
              <button
                className={`badge ${z.enabled ? "bg-campus-ok/20 text-campus-ok" : "bg-slate-700 text-slate-400"}`}
                onClick={() => toggle(z)}
              >
                {z.enabled ? "Enabled" : "Disabled"}
              </button>
            </div>
            <p className="mb-3 text-xs text-slate-500">
              Area: ({Math.round(z.x1 * 100)}%, {Math.round(z.y1 * 100)}%) → (
              {Math.round(z.x2 * 100)}%, {Math.round(z.y2 * 100)}%)
            </p>
            {z.zone_type === "crowd" && (
              <label className="flex items-center gap-3 text-sm text-slate-300">
                Max persons
                <input
                  type="range"
                  min={2}
                  max={20}
                  value={z.max_persons}
                  onChange={(e) => setMax(z, Number(e.target.value))}
                  className="flex-1"
                />
                <span className="w-8 text-right font-mono">{z.max_persons}</span>
              </label>
            )}
            {z.zone_type === "restricted" && (
              <p className="text-sm text-slate-400">
                Unknown faces trigger security alert when enabled.
              </p>
            )}
          </div>
        ))}
      </div>

      <div className="card">
        <h3 className="mb-4 font-semibold">Zone configuration</h3>
        <DataTable
          columns={[
            { key: "name", label: "Zone" },
            { key: "zone_type", label: "Type" },
            {
              key: "enabled",
              label: "Enabled",
              render: (r) => (r.enabled ? "Yes" : "No"),
            },
            { key: "max_persons", label: "Max persons" },
            {
              key: "allow_unknown",
              label: "Allow unknown",
              render: (r) => (r.allow_unknown ? "Yes" : "No"),
            },
          ]}
          rows={zones}
        />
      </div>
    </div>
  );
}
