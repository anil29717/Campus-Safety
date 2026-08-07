export function StatCard({ label, value, sub, tone = "default" }) {
  const tones = {
    default: "text-white",
    ok: "text-campus-ok",
    warn: "text-campus-warn",
    danger: "text-campus-danger",
    accent: "text-campus-accent",
  };
  return (
    <div className="card">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-2 text-3xl font-bold ${tones[tone] || tones.default}`}>{value}</p>
      {sub && <p className="mt-1 text-sm text-slate-400">{sub}</p>}
    </div>
  );
}
