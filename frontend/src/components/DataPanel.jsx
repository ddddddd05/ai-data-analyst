// 数据区：上传文件 / 使用示例数据（表格或数据库）；显示数据概况、各列情况和前 50 行，或数据库的表结构
import DataTable from "./DataTable.jsx";

function Database({ name, db }) {
  return (
    <details>
      <summary>
        数据库：{name}　|　{db.tables.length} 张表（{db.tables.map((t) => `${t.name} ${t.rows} 行`).join("、")}），
        {db.metrics.length} 个预定义指标（点击查看）
      </summary>
      {db.tables.map((t) => (
        <div key={t.name}>
          <h3>{t.name}（{t.rows} 行）</h3>
          <p className="muted small">{t.desc}</p>
          <DataTable columns={["列名", "类型", "说明"]} rows={t.columns.map((c) => [c.name, c.type, c.desc])} />
          {t.foreign_keys.length > 0 && <p className="muted small">外键：{t.foreign_keys.join("；")}</p>}
        </div>
      ))}
      <h3>指标口径（语义层）</h3>
      <DataTable columns={["指标", "口径"]} rows={db.metrics.map((m) => [m.name, m.formula])} />
    </details>
  );
}

export default function DataPanel({ info, busy, onSample, onSampleDb, onUpload }) {
  const d = info?.data;
  const missingCols = d ? d.columns.filter((c) => c.missing > 0).length : 0;
  return (
    <section className="card">
      <div className="data-actions">
        <label className="upload">
          上传 CSV / Excel / SQLite
          <input type="file" accept=".csv,.xlsx,.xls,.db,.sqlite,.sqlite3" disabled={busy}
            onChange={(e) => e.target.files[0] && onUpload(e.target.files[0])} />
        </label>
        <button onClick={onSample} disabled={busy}
          title="模拟的跨境电商渔具订单，含故意制造的数据问题">使用示例数据</button>
        <button onClick={onSampleDb} disabled={busy}
          title="5 张表的 SQLite 数据库（订单、明细、商品、店铺、广告），Agent 自己写 SQL">使用示例数据库（多表）</button>
      </div>
      {!d && !info?.database && <p className="muted">请上传文件，或点击“使用示例数据”/“使用示例数据库”。</p>}
      {info?.database && <Database name={info.data_name} db={info.database} />}
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
