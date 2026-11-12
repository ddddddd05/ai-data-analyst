"""把一次运行的结果导出成 HTML 报告和 JSON 日志。"""
from __future__ import annotations

import base64
import html
import json

from .agent import RunResult


def _md_to_html(md: str) -> str:
    """极简 Markdown 转换（标题、列表、加粗、段落），够报告用即可。"""
    out, in_list = [], False
    for line in md.splitlines():
        s = html.escape(line.strip())
        s = _bold(s)
        if s.startswith("### "):
            tag = f"<h4>{s[4:]}</h4>"
        elif s.startswith("## "):
            tag = f"<h3>{s[3:]}</h3>"
        elif s.startswith("# "):
            tag = f"<h2>{s[2:]}</h2>"
        elif s.startswith(("- ", "* ")) or (s[:2].rstrip(".").isdigit() and ". " in s[:4]):
            item = s[2:] if s.startswith(("- ", "* ")) else s.split(". ", 1)[1]
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{item}</li>")
            continue
        elif s:
            tag = f"<p>{s}</p>"
        else:
            tag = ""
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(tag)
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _bold(s: str) -> str:
    parts = s.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def to_html(run: RunResult) -> str:
    warn = ""
    if run.unverified:
        items = "、".join(html.escape(x) for x in run.unverified)
        warn = (f'<div class="warn"><b>⚠ 自动校验提醒：</b>报告中这些数字没有在计算结果里找到出处，'
                f'请人工核对：{items}</div>')
    steps_html = []
    for s in run.steps:
        status = "成功" if s.ok else "失败"
        body = [f"<h4>第{s.id}步：{html.escape(s.goal)}（{status}，尝试 {s.attempts} 次）</h4>"]
        if s.result and s.result.metrics:
            rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:g}</td></tr>" if isinstance(v, float)
                           else f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                           for k, v in s.result.metrics.items())
            body.append(f"<table>{rows}</table>")
        if s.result:
            for name, t in s.result.tables.items():
                body.append(f"<p><b>{html.escape(name)}</b></p>{t.to_html(max_rows=20)}")
            for img in s.result.figures:
                b64 = base64.b64encode(img).decode()
                body.append(f'<img src="data:image/png;base64,{b64}">')
        body.append(f"<details><summary>查看代码</summary><pre>{html.escape(s.code)}</pre></details>")
        steps_html.append("\n".join(body))
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>电商经营分析报告</title>
<style>
body{{font-family:"Microsoft YaHei",sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;line-height:1.7}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #ccc;padding:4px 8px}}
img{{max-width:100%}} pre{{background:#f5f5f5;padding:8px;overflow:auto;font-size:13px}}
.warn{{background:#fff4e5;border:1px solid #f0a500;padding:8px 12px;border-radius:6px}}
.q{{background:#eef4ff;padding:8px 12px;border-radius:6px}}
</style></head><body>
<h2>电商经营分析报告</h2>
<p class="q"><b>问题：</b>{html.escape(run.question)}</p>
<p>数据：{run.profile['rows']} 行 × {run.profile['cols']} 列；用时 {run.seconds} 秒。
以下内容由 AI 生成，数字经程序核对，结论仍需人工判断。</p>
{warn}
{_md_to_html(run.report)}
<h3>分析过程</h3>
{''.join(steps_html)}
</body></html>"""


def to_log_json(run: RunResult) -> str:
    return json.dumps({"question": run.question, "plan": run.plan, "unverified": run.unverified,
                       "seconds": run.seconds, "log": run.log}, ensure_ascii=False, indent=2, default=str)
