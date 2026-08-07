import { useEffect, useState } from "react";
import { api } from "../api/client";
import ActivityChart from "../components/ActivityChart";
import { DataTable } from "../components/DataTable";

const BOX_HEIGHT = "h-[420px]";

export default function Activity() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const load = () => api.activity(100).then(setData).catch(() => {});
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      <header className="mb-8">
        <h2 className="text-2xl font-bold text-white">Emotion & Behavior</h2>
        <p className="text-slate-400">Logged activity for registered campus members only</p>
      </header>

      <div className={`card mb-8 flex ${BOX_HEIGHT} flex-col`}>
        <h3 className="mb-2 shrink-0 font-semibold">Emotion confidence over time</h3>
        <p className="mb-3 shrink-0 text-xs text-slate-500">
          One line per registered user — Y-axis is emotion model confidence (%)
        </p>
        <div className="min-h-0 flex-1">
          <ActivityChart chartSeries={data?.chart_series} />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className={`card flex ${BOX_HEIGHT} flex-col`}>
          <h3 className="mb-4 shrink-0 font-semibold">Latest per person</h3>
          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            <DataTable
              columns={[
                { key: "Person", label: "Person" },
                { key: "Last emotion", label: "Emotion" },
                { key: "Last behavior", label: "Behavior" },
                { key: "Time", label: "Time" },
              ]}
              rows={data?.summary || []}
            />
          </div>
        </div>

        <div className={`card flex ${BOX_HEIGHT} flex-col`}>
          <h3 className="mb-4 shrink-0 font-semibold">Recent events</h3>
          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            <DataTable
              columns={[
                { key: "Time", label: "Time" },
                { key: "Person", label: "Person" },
                { key: "Emotion", label: "Emotion" },
                { key: "Behavior", label: "Behavior" },
              ]}
              rows={data?.recent || []}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
