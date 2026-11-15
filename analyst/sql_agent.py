"""数据库模式的规划-执行 Agent（Text-to-SQL）：数据在多张表里，每一步写一条 SQL 查询。

流程和表格模式（agent.py）一样，只是“写 Python → 子进程执行”换成了“写 SQL → 只读执行”：
    schema linking（挑出相关的表 + 指标口径）→ 规划 → 对每一步：写 SQL → 安全检查 → 只读执行
    → 出错或结果为空就把问题交回模型修改（最多重试 max_retries 次）→ 写报告 → 核对数字 → 独立审查

和表格模式的区别：
| | 表格模式 | 数据库模式 |
|---|---|---|
| 交给模型看什么 | 数据概况 + 前 5 行 | schema linking 选出的表结构 + 语义层的指标口径（不是全部表） |
| 每一步写什么 | Python（pandas） | 一条 SQLite 查询 |
| 护栏 | 静态检查 + 子进程 + 超时 | 只读连接 + 授权回调 + 超时 + 行数上限 |
| 指标口径从哪来 | 电商指标 Skill | 语义层（<库名>.schema.json 的 metrics） |
| 数据清洗 | 第一步清洗，后面共用 | 数据库有约束，不做清洗 |

schema linking 漏掉了表、模型写了不存在的列时，报错信息里会附上全部表的结构，让模型在修正时补上。
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import prompts, usage
from .agent import (DataAnalystAgent, RunResult, StepRecord, _brief, _format_history, check_and_review,
                    format_results, review_context, run_tracked)
from .checkpoint import task_id
from .database import Database, SQLResult
from .knowledge import knowledge_numbers, render_knowledge
from .llm import extract_sql
from .memory import render_preferences
from .profiler import profile_numbers
from .sandbox import ExecResult
from .validator import check_step, numbers_from_results

EMPTY_RESULT = ("查询成功但结果为空。检查 WHERE 条件里的取值是否和表里的取值完全一致（取值样例见表结构），"
                "以及 JOIN 条件是否正确。")


def sql_to_exec(res: SQLResult, max_rows: int) -> ExecResult:
    """把查询结果整理成和 Python 步骤一样的格式，报告、数字核对、审查、评测都不用改。

    只有一行结果时，数字列同时作为“指标”（如 钓竿退货率 = 0.0712）。
    """
    df = res.df
    metrics = {}
    if len(df) == 1:
        for col in df.columns:
            v = df[col].iloc[0]
            if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)):
                metrics[str(col)] = float(v)
    notes = f"查询返回 {len(df)} 行"
    if res.truncated:
        notes += f"（超过 {max_rows} 行，只保留前 {max_rows} 行；应该在 SQL 里汇总）"
    return ExecResult(ok=True, metrics=metrics, tables={"查询结果": df}, notes=notes)


def schema_prompt(db: Database, question: str, names: list[str]) -> str:
    """交给模型的“数据说明”：全部表的一行简介 + 相关表的完整结构 + 指标口径。"""
    return (f"数据库 {db.name}，共 {len(db.tables)} 张表：\n{db.overview()}\n\n"
            f"与问题相关的表（schema linking 选出，写 SQL 只用这些表）：\n{db.schema_text(names)}\n\n"
            f"指标口径（语义层，必须按此计算）：\n{db.metrics_text(question)}")


def last_turn(history: list[tuple[str, str]] | str | None) -> str:
    """上一轮的问题和结论（追问时 schema linking 用来理解“它”指什么）。"""
    if not history:
        return ""
    if isinstance(history, str):          # 记忆模块整理好的文字：取最后一个“第N轮问题……”段落
        blocks = [b for b in history.split("\n\n") if b.startswith("第") and "轮问题" in b]
        return blocks[-1] if blocks else ""
    q, c = history[-1]
    return f"{q}\n{c}"


class SQLAnalystAgent(DataAnalystAgent):
    def __init__(self, llm, db: Database, max_steps: int = 4, **kwargs):
        super().__init__(llm, max_steps=max_steps, skills=[], **kwargs)
        self.db = db

    def run(self, question: str, on_event: Callable[[str, object], None] | None = None,
            history: list[tuple[str, str]] | str | None = None,
            preferences: list[str] | None = None) -> RunResult:
        return run_tracked(self._run_db, question, on_event, history, preferences)

    def _run_db(self, question, on_event, history, preferences) -> RunResult:
        t0 = time.time()
        log: list[dict] = []

        def emit(kind: str, payload: object = None) -> None:
            log.append({"time": round(time.time() - t0, 2), "event": kind, "detail": _brief(payload)})
            if on_event:
                on_event(kind, payload)

        db = self.db
        prof = db.profile()
        # ① schema linking：只把相关的表交给模型（表多时省 token，也减少用错表）
        linked = db.link(question, context=last_turn(history))
        emit("schema_link", linked)
        ptext = schema_prompt(db, question, [n for n, _ in linked])
        extra = render_preferences(preferences)
        if preferences:
            emit("preferences", preferences)
        hits = self.knowledge.search(question) if self.knowledge else []
        kb_text = render_knowledge(hits)
        extra += kb_text
        if self.knowledge:
            emit("knowledge", [h.to_dict() for h in hits])
        history_text = _format_history(history)

        store = self.checkpoints
        tid = task_id("sql", question, db, history, preferences) if store else ""
        saved = store.load(tid) if store else None
        if saved:
            plan, steps = saved["plan"], saved["steps"]
            emit("resume", {"done": len(steps), "total": len(plan)})
        else:
            plan, steps = self._plan(question, ptext, extra or "（无）", history_text, emit,
                                     system=prompts.SQL_PLANNER_SYSTEM), []
            if store:
                store.save(tid, {"plan": plan, "steps": steps})
        emit("plan", plan)
        resumed = len(steps)

        for item in plan[len(steps):]:
            rec = StepRecord(id=int(item.get("id", len(steps) + 1)), goal=str(item.get("goal", "")),
                             method=str(item.get("method", "")))
            emit("step_start", rec)
            self._run_sql_step(rec, question, ptext, extra or "（无）", steps, emit)
            steps.append(rec)
            emit("step_done", rec)
            if store:
                store.save(tid, {"plan": plan, "steps": steps})

        results_text = format_results(steps)
        report = self._report(question, ptext, results_text, extra or "（无）", history_text, emit)
        known = profile_numbers(prof) + db.numbers() + numbers_from_results([s.result for s in steps if s.result])
        known += knowledge_numbers(hits)
        report, unverified, review = check_and_review(
            report, known, question, review_context(ptext, results_text, knowledge_text=kb_text), self.llm,
            self.reviewer if self.review else None, emit)
        if store:
            store.clear(tid)
        return RunResult(question, prof, plan, steps, report, unverified, log, round(time.time() - t0, 1),
                         [], mode="sql", review=review, resumed_steps=resumed,
                         knowledge=[h.to_dict() for h in hits],
                         sql=[{"table": n, "reason": r} for n, r in linked])

    # ---------- 单步：写 SQL → 只读执行 → 修正 ----------
    def _run_sql_step(self, rec: StepRecord, question, ptext, extra, done: list[StepRecord], emit) -> None:
        previous = "\n\n".join(
            f"第{s.id}步（{s.goal}）：\n```sql\n{s.code}\n```\n"
            + (s.result.tables["查询结果"].head(10).to_string(index=False) if s.ok else "失败")
            for s in done) or "（无）"
        with usage.stage("写 SQL"):
            reply = self.llm.chat(prompts.SQL_CODER_SYSTEM, prompts.SQL_CODER_USER.format(
                question=question, step_id=rec.id, goal=rec.goal, method=rec.method,
                schema=ptext, previous=previous, skills=extra))
        sql = extract_sql(reply)
        widened = False
        for attempt in range(1, self.max_retries + 2):
            rec.attempts, rec.code = attempt, sql
            res = self.db.run_sql(sql)
            if res.ok and len(res.df):
                rec.result = sql_to_exec(res, self.db.max_rows)
                rec.data_rows = len(res.df)
                rec.warnings = check_step(rec.result)
                return
            error = EMPTY_RESULT if res.ok else res.error
            rec.result = ExecResult(ok=False, error=error)
            rec.errors.append(error)
            emit("step_error", {"step": rec.id, "attempt": attempt, "error": error})
            if attempt > self.max_retries:
                return
            if not widened and ("no such column" in error or "no such table" in error):
                # schema linking 可能漏了表：附上全部表结构，让模型自己补
                error += "\n\n数据库全部表的结构：\n" + self.db.schema_text()
                widened = True
                emit("schema_widen", rec.id)
            with usage.stage("修 SQL"):
                reply = self.llm.chat(prompts.SQL_CODER_SYSTEM,
                                      prompts.SQL_FIXER_USER.format(code=sql, error=error[-3000:]))
            sql = extract_sql(reply)
