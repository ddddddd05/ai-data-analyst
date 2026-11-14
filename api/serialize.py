"""把 Python 对象（RunResult、DataFrame、numpy 数字）转成前端能直接用的 JSON。"""
from __future__ import annotations

import base64
import json
import math

import numpy as np
import pandas as pd

from analyst.agent import RunResult, StepRecord


def plain(obj):
    """递归地把 numpy 数字、NaN 等转成普通 Python 值（JSON 不认识 numpy 和 NaN）。"""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def table(df: pd.DataFrame, max_rows: int = 50) -> dict:
    """表格 → {columns, rows}；日期转成文字，缺失值变 null。"""
    data = json.loads(df.head(max_rows).reset_index(drop=df.index.name is None and
                                                    isinstance(df.index, pd.RangeIndex))
                      .to_json(orient="split", date_format="iso", force_ascii=False))
    return {"columns": [str(c) for c in data["columns"]], "rows": data["data"]}


def step(s: StepRecord) -> dict:
    r = s.result
    return {
        "id": s.id, "goal": s.goal, "method": s.method, "ok": s.ok, "attempts": s.attempts,
        "data_rows": s.data_rows, "code": s.code, "errors": [e[-800:] for e in s.errors],
        "warnings": s.warnings,
        "notes": r.notes if r and r.ok else "",
        "metrics": plain(r.metrics) if r and r.ok else {},
        "tables": [{"name": k, **table(v)} for k, v in r.tables.items()] if r and r.ok else [],
        "figures": [base64.b64encode(b).decode() for b in r.figures] if r and r.ok else [],
    }


def run(run_id: str, r: RunResult) -> dict:
    return {
        "run_id": run_id, "question": r.question, "mode": r.mode, "report": r.report,
        "unverified": r.unverified, "seconds": r.seconds,
        "skills": [{"name": n, "reason": w} for n, w in r.skills],
        "review": r.review.to_dict() if r.review else None,
        "tool_calls": plain(r.tool_calls), "resumed_steps": r.resumed_steps,
        "knowledge": plain(r.knowledge),
        "steps": [step(s) for s in r.steps],
    }


def event_text(kind: str, payload) -> str | None:
    """把 Agent 的进度事件翻译成给人看的一句话（前端直接显示）。返回 None 表示不显示。"""
    if kind == "profile":
        return "🔍 已读取数据概况"
    if kind == "skills":
        return "；".join(f"🧩 加载 Skill：{n}（{w}）" for n, w in payload) if payload else None
    if kind == "knowledge":
        if not payload:
            return "📚 业务知识库：没有找到和问题相关的资料"
        return "📚 检索业务知识：" + "；".join(h["title"] for h in payload)
    if kind == "preferences":
        return "📌 遵守长期偏好：" + "；".join(payload)
    if kind == "resume":
        return f"💾 找到上次的存档：已完成 {payload['done']} 步，从断点继续"
    if kind == "mcp_connected":
        text = f"🔌 已连接 MCP 服务器，获得 {len(payload['tools'])} 个工具：" + "、".join(payload["tools"])
        if payload["hidden"]:
            text += "；未开放：" + "、".join(payload["hidden"])
        return text
    if kind == "plan_trimmed":
        return f"✂️ 去掉汇总类步骤“{payload}”，改由最后的报告完成"
    if kind == "plan":
        return "🗺️ 分析计划：" + "；".join(f"{s.get('id')}. {s.get('goal')}" for s in payload)
    if kind == "step_start":
        return f"⚙️ 正在执行第{payload.id}步：{payload.goal}"
    if kind == "step_error":
        return f"↻ 第{payload['step']}步第{payload['attempt']}次出错，正在修正"
    if kind == "cleaned":
        return f"🧹 第{payload['step']}步清洗完成，后续统一使用 {payload['rows']} 行数据"
    if kind == "tool_call":
        a = payload["args"]
        detail = a.get("purpose") or a.get("name") or a.get("column") or a.get("query") or ""
        return f"🛠️ 第{payload['n']}次调用工具 {payload['name']}（{payload['level']}）{detail}"
    if kind == "tool_result" and not payload["ok"]:
        return f"↻ {payload['name']} 没成功：{payload['text'][:120]}"
    if kind == "tool_limit":
        return f"⛔ 工具调用达到上限 {payload} 次，强制写报告"
    if kind == "review_start":
        return "🧑‍⚖️ 独立审查员正在检查结论是否与计算结果矛盾"
    if kind == "review" and payload.revised:
        return "↻ 审查发现严重矛盾，已让 AI 修改报告"
    if kind == "verify":
        return "🔎 正在核对报告中的数字"
    return None
