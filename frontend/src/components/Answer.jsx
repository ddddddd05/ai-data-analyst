// 一条回答：加载的 Skill、数字核对、独立审查、报告正文、工具调用记录、分析过程、下载
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DataTable from "./DataTable.jsx";

function Review({ review }) {
  if (!review) return null;
  if (review.error) return <p className="muted small">🧑‍⚖️ 独立审查未完成（不影响报告）：{review.error}</p>;
  const serious = review.issues.filter((i) => i.severity === "严重");
  if (review.revised)
    return (
      <div className="banner warn">
        🧑‍⚖️ 独立审查发现结论与计算结果矛盾，报告已按审查意见修改一次：
        <ul>{serious.map((i, k) => <li key={k}>{i.claim}（结果显示：{i.evidence}）</li>)}</ul>
      </div>
    );
  if (review.issues.length)
    return (
      <div className="banner info">
        🧑‍⚖️ 独立审查：没有严重矛盾，有几点表述提醒：
        <ul>{review.issues.map((i, k) => <li key={k}>{i.claim}（{i.evidence}）</li>)}</ul>
      </div>
    );
  return <div className="banner ok">🧑‍⚖️ 独立审查通过：结论与计算结果没有发现矛盾。</div>;
}

function Step({ step }) {
  return (
    <details className="step">
      <summary>
        {step.ok ? "✅" : "❌"} 第{step.id}步：{step.goal}（使用 {step.data_rows} 行，尝试 {step.attempts} 次）
      </summary>
      {step.errors.map((e, i) => (
        <div key={i} className="banner warn small">第 {i + 1} 次失败，已把错误交给 AI 修改：<pre>{e}</pre></div>
      ))}
      {step.notes && <p>{step.notes}</p>}
      {Object.keys(step.metrics).length > 0 && (
        <DataTable columns={["指标", "数值"]} rows={Object.entries(step.metrics)} />
      )}
      {step.tables.map((t) => (
        <div key={t.name}>
          <h4>{t.name}</h4>
          <DataTable columns={t.columns} rows={t.rows} />
        </div>
      ))}
      {step.figures.map((f, i) => <img key={i} src={`data:image/png;base64,${f}`} alt={`图表 ${i + 1}`} />)}
      {step.warnings.map((w, i) => <p key={i} className="banner info small">{w}</p>)}
      <details>
        <summary className="small">查看代码</summary>
        <pre className="code">{step.code}</pre>
      </details>
    </details>
  );
}

export default function Answer({ result }) {
  return (
    <div className="answer">
      {result.skills.length > 0 && (
        <p className="muted small">🧩 加载的 Skill：{result.skills.map((s) => `${s.name}（${s.reason}）`).join("；")}</p>
      )}
      {result.resumed_steps > 0 && <p className="muted small">💾 从存档恢复了 {result.resumed_steps} 步，没有重做</p>}
      {result.unverified.length > 0 ? (
        <div className="banner warn">⚠ 自动校验：报告中这些数字没有在计算结果里找到出处，请人工核对：{result.unverified.join("、")}</div>
      ) : (
        <div className="banner ok">自动校验通过：报告中的数字都能在计算结果中找到出处。</div>
      )}
      <Review review={result.review} />

      <div className="report">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.report}</ReactMarkdown>
      </div>

      {result.tool_calls.length > 0 && (
        <details>
          <summary>查看工具调用记录（{result.tool_calls.length} 次{result.mode === "mcp" ? "，经 MCP" : ""}）</summary>
          <ol className="tools">
            {result.tool_calls.map((t) => {
              const a = t.args || {};
              return (
                <li key={t.n}>
                  {t.ok ? "✅" : "❌"} <code>{t.name}</code>（{t.level}）{a.purpose || a.name || a.column || ""}
                  {!t.ok && <div className="muted small">{t.result}</div>}
                </li>
              );
            })}
          </ol>
        </details>
      )}

      <details>
        <summary>查看分析过程（{result.steps.length} 步，用时 {result.seconds} 秒）</summary>
        {result.steps.map((s) => <Step key={s.id} step={s} />)}
      </details>

      <div className="downloads">
        <a href={`/api/runs/${result.run_id}/report.html`}>下载 HTML 报告</a>
        <a href={`/api/runs/${result.run_id}/log.json`}>下载运行日志（JSON）</a>
      </div>
    </div>
  );
}
