export function DataTable({ columns, rows, empty = "No data" }) {
  if (!rows?.length) {
    return <p className="py-8 text-center text-slate-500">{empty}</p>;
  }
  return (
    <div className="overflow-x-auto rounded-lg border border-campus-700">
      <table className="w-full text-left text-sm">
        <thead className="bg-campus-800 text-xs uppercase text-slate-400">
          <tr>
            {columns.map((c) => (
              <th key={c.key} className="px-4 py-3 font-semibold">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-campus-700">
          {rows.map((row, i) => (
            <tr key={row.id || i} className="hover:bg-campus-800/50">
              {columns.map((c) => (
                <td key={c.key} className="px-4 py-3 text-slate-300">
                  {c.render ? c.render(row) : row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
