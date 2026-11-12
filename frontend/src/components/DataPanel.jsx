// 数据区：上传文件 / 使用示例数据；显示数据概况、各列情况和前 50 行
import DataTable from "./DataTable.jsx";

export default function DataPanel({ info, busy, onSample, onUpload }) {
  const d = info?.data;
  const missingCols = d ? d.columns.filter((c) => c.missing > 0).length : 0;
  return (
    <section className="card">
      <div className="data-actions">
        <label className="upload">
          上传 CSV / Excel
          <input type="file" accept=".csv,.xlsx,.xls" disabled={busy}
            onChange={(e) => e.target.files[0] && onUpload(e.target.files[0])} />
        </label>
        <button onClick={onSample} disabled={busy}
          title="模拟的跨境电商渔具订单，含故意制造的数据问题">使用示例数据</button>
      </div>
      {!d && <p className="muted">请上传文件，或点击“使用示例数据”。</p>}
      {d && (
        <details>
          <summary>
            数据：{info.data_name}　|　{d.rows} 行 × {d.cols} 列，重复 {d.duplicate_rows} 行，
            {missingCols} 列有缺失（点击查看）
          </summary>
          <h3>前 50 行</h3>
          <DataTable columns={d.preview.columns} rows={d.preview.rows} />
          <h3>各列情况</h3>
          <DataTable
            columns={["name", "kind", "missing", "unique", "negative", "outliers"]}
            headers={["列名", "类型", "缺失", "不同取值", "负数", "极端值"]}
            rows={d.columns.map((c) => [c.name, c.kind ?? c.dtype, c.missing, c.unique, c.negative ?? "", c.outliers ?? ""])}
          />
        </details>
      )}
    </section>
  );
}
