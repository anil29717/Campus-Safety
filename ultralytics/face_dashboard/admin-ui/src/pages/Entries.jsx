import { useEffect, useState } from "react";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { StatCard } from "../components/StatCard";

const BOX_HEIGHT = "h-[420px]";

export default function Entries() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const load = () => api.entries(200).then(setData).catch(() => {});
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      <header className="mb-8">
        <h2 className="text-2xl font-bold text-white">Entry History</h2>
        <p className="text-slate-400">
          In / out times from Entry and Exit gate cameras
        </p>
      </header>

      <div className="mb-8 grid gap-4 sm:grid-cols-4">
        <StatCard label="Total visits" value={data?.total ?? "—"} />
        <StatCard label="Inside now" value={data?.active ?? "—"} tone="accent" />
        <StatCard label="Identified visits" value={data?.identified ?? "—"} tone="ok" />
        <StatCard label="Unknown visits" value={data?.unknown ?? "—"} tone="warn" />
      </div>

      {data?.currently_in?.length > 0 && (
        <div className="card mb-8">
          <h3 className="mb-4 font-semibold text-campus-ok">Currently inside</h3>
          <DataTable
            columns={[
              { key: "Person ID", label: "Person ID" },
              { key: "Person", label: "Person" },
              { key: "Type", label: "Type" },
              { key: "In time", label: "In time" },
              { key: "Camera", label: "Camera" },
              { key: "Gate", label: "Gate" },
              {
                key: "Status",
                label: "Status",
                render: (r) => <span className="text-campus-ok">{r.Status}</span>,
              },
            ]}
            rows={data.currently_in}
          />
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className={`card flex ${BOX_HEIGHT} flex-col`}>
          <h3 className="mb-4 shrink-0 font-semibold">Visits per person</h3>
          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            <DataTable
              columns={[
                { key: "Person ID", label: "Person ID" },
                { key: "Person", label: "Person" },
                { key: "Visits", label: "Visits" },
              ]}
              rows={data?.summary || []}
            />
          </div>
        </div>

        <div className={`card flex ${BOX_HEIGHT} flex-col`}>
          <h3 className="mb-4 shrink-0 font-semibold">In / out log</h3>
          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            <DataTable
              columns={[
                { key: "Person ID", label: "Person ID" },
                { key: "Person", label: "Person" },
                { key: "In time", label: "In" },
                { key: "Out time", label: "Out" },
                { key: "Duration", label: "Duration" },
                { key: "Camera", label: "Camera" },
                { key: "Gate", label: "Gate" },
                {
                  key: "Status",
                  label: "Status",
                  render: (r) => (
                    <span
                      className={
                        r.Status === "Inside" || r.Status === "In camera"
                          ? "text-campus-ok"
                          : "text-slate-400"
                      }
                    >
                      {r.Status}
                    </span>
                  ),
                },
              ]}
              rows={data?.recent || []}
              empty="No visits yet — start an Entry camera"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
