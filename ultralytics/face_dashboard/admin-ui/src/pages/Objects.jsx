import { useEffect, useState } from "react";
import { api } from "../api/client";
import { DataTable } from "../components/DataTable";
import { StatCard } from "../components/StatCard";

export default function Objects() {
  const [objects, setObjects] = useState([]);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [objs, st] = await Promise.all([api.liveObjects(), api.status()]);
        setObjects(objs);
        setStatus(st);
      } catch {
        /* ignore */
      }
    };
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      <header className="mb-8">
        <h2 className="text-2xl font-bold text-white">Object Detection</h2>
        <p className="text-slate-400">
          YOLO detects bags (backpack, handbag, suitcase) plus fire, smoke, and fighting hazards
        </p>
      </header>

      <div className="mb-8 grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Objects in frame" value={objects.length} tone="accent" />
        <StatCard label="Bags tracked" value={status?.bags_tracked ?? "—"} tone="warn" />
        <StatCard
          label="Fire"
          value={status?.fire_detected ? "YES" : "—"}
          tone={status?.fire_detected ? "danger" : "default"}
        />
        <StatCard
          label="Smoke"
          value={status?.smoke_detected ? "YES" : "—"}
          tone={status?.smoke_detected ? "warn" : "default"}
        />
        <StatCard
          label="Fighting"
          value={status?.fighting_detected ? "YES" : "—"}
          tone={status?.fighting_detected ? "danger" : "default"}
        />
        <StatCard
          label="Camera"
          value={status?.running ? "Live" : "Off"}
          tone={status?.running ? "ok" : "default"}
        />
      </div>

      {(status?.hazards?.length ?? 0) > 0 && (
        <div className="card mb-8 border-campus-danger/40">
          <h3 className="mb-4 font-semibold text-campus-danger">Live hazards</h3>
          <DataTable
            columns={[
              { key: "type", label: "Type" },
              { key: "message", label: "Message" },
              {
                key: "confidence",
                label: "Confidence",
                render: (r) => `${(r.confidence * 100).toFixed(0)}%`,
              },
            ]}
            rows={status.hazards}
            empty=""
          />
        </div>
      )}

      <div className="card">
        <h3 className="mb-4 font-semibold">Live objects</h3>
        <DataTable
          columns={[
            { key: "label", label: "Label" },
            {
              key: "confidence",
              label: "Confidence",
              render: (r) => `${(r.confidence * 100).toFixed(0)}%`,
            },
            {
              key: "bbox",
              label: "Bounding box",
              render: (r) => r.bbox?.join(", "),
            },
          ]}
          rows={objects}
          empty="No objects — start camera on Live Camera page"
        />
      </div>
    </div>
  );
}
