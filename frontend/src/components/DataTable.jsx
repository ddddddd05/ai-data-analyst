// 通用表格：columns 是列名，rows 是二维数组；headers 可以替换显示的表头
function show(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? v : Math.round(v * 10000) / 10000;
  return String(v);
}

export default function DataTable({ columns, rows, headers }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{(headers || columns).map((c, i) => <th key={i}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j}>{show(v)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
