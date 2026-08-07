import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4", "#ec4899", "#84cc16"];

/** Merge per-person series into rows for Recharts multi-line chart. */
function mergeSeries(chartSeries) {
  if (!chartSeries?.length) return { rows: [], people: [] };

  const people = chartSeries.map((s) => s.person);
  const timeMap = new Map();

  for (const { person, points } of chartSeries) {
    for (const pt of points) {
      const key = pt.time;
      if (!timeMap.has(key)) {
        timeMap.set(key, { time: key, _sort: key });
      }
      timeMap.get(key)[person] = pt.value;
      timeMap.get(key)[`${person}_emotion`] = pt.emotion;
      timeMap.get(key)[`${person}_behavior`] = pt.behavior;
    }
  }

  const rows = Array.from(timeMap.values()).sort((a, b) => a._sort.localeCompare(b._sort));
  return { rows, people };
}

function formatTime(t) {
  if (!t) return "";
  const part = t.includes(" ") ? t.split(" ")[1] : t;
  return part?.slice(0, 8) || t;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-600 bg-campus-900 px-3 py-2 text-xs shadow-xl">
      <p className="mb-1 font-medium text-slate-300">{label}</p>
      {payload.map((p) => {
        const emotion = p.payload[`${p.name}_emotion`];
        const behavior = p.payload[`${p.name}_behavior`];
        return (
          <p key={p.name} style={{ color: p.color }} className="leading-relaxed">
            {p.name}: {p.value}% — {emotion}
            {behavior && behavior !== "—" ? ` · ${behavior}` : ""}
          </p>
        );
      })}
    </div>
  );
}

export default function ActivityChart({ chartSeries }) {
  const { rows, people } = mergeSeries(chartSeries);

  if (!rows.length) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        No chart data yet — activity appears when registered users are on camera.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#243044" />
        <XAxis
          dataKey="time"
          tickFormatter={formatTime}
          stroke="#64748b"
          tick={{ fontSize: 11 }}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[0, 100]}
          stroke="#64748b"
          tick={{ fontSize: 11 }}
          label={{ value: "Emotion conf %", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
        {people.map((name, i) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            name={name}
            stroke={COLORS[i % COLORS.length]}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
