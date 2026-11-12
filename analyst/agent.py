"""电商经营分析 Agent 主流程。

感知（数据概况）→ 选择 Skill → 规划（拆步骤）→ 对每一步：写代码 → 安全检查 → 执行 →
出错就把错误交回模型修改（最多重试 max_retries 次）→ 汇总结果写报告 →
校验报告里的数字 → 返回完整记录（可追溯）。

清洗步骤产出的数据（result["cleaned_df"]）会交给后续所有步骤共用，保证口径一致。
支持多轮对话：把之前的问答摘要作为“对话历史”交给模型，可以连续追问。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import prompts
from .checkpoint import CheckpointStore, task_id
from .llm import LLM, extract_code, extract_json
from .memory import render_preferences
from .profiler import profile, profile_numbers, profile_to_text
from .reviewer import ReviewResult, review_report, revise_report
from .sandbox import ExecResult, run_code, static_check
from .skills import Skill, load_skills, render_skills, select_skills
from .validator import check_step, numbers_from_results, verify_report_numbers


@dataclass
class StepRecord:
    id: int
    goal: str
    method: str
    code: str = ""
    attempts: int = 0
    result: ExecResult | None = None
    errors: list[str] = field(default_factory=list)    # 每次失败的原因
    warnings: list[str] = field(default_factory=list)
    data_rows: int = 0                                  # 这一步用的数据有多少行

    @property
    def ok(self) -> bool:
        return bool(self.result and self.result.ok and self.result.has_output)


@dataclass
class RunResult:
    question: str
    profile: dict
    plan: list[dict]
    steps: list[StepRecord]
    report: str = ""
    unverified: list[str] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    skills: list[tuple[str, str]] = field(default_factory=list)  # [(Skill 名称, 加载原因)]
    mode: str = "plan"                                            # plan：规划-执行；tools：工具调用；mcp：工具经 MCP 调用
    tool_calls: list[dict] = field(default_factory=list)          # 工具模式下每次工具调用的记录
    review: ReviewResult | None = None                            # 独立审查 Agent 的意见
    resumed_steps: int = 0                                        # 从存档恢复、没有重新执行的步骤数


class DataAnalystAgent:
    def __init__(self, llm: LLM, max_retries: int = 2, timeout: int = 60, max_steps: int = 5,
                 skills: list[Skill] | None = None, reviewer: LLM | None = None,
                 review: bool = True, checkpoints: CheckpointStore | None = None):
        self.llm = llm
        self.reviewer = reviewer or llm   # 审查员可以换成另一个模型
        self.review = review
        self.checkpoints = checkpoints    # 传入后：每步存盘，失败后同一任务重跑时从断点继续
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_steps = max_steps
        self.skills = load_skills() if skills is None else skills

    # ---------- 对外入口 ----------
    def run(self, df: pd.DataFrame, question: str,
            on_event: Callable[[str, object], None] | None = None,
            history: list[tuple[str, str]] | str | None = None,
            preferences: list[str] | None = None) -> RunResult:
        """history：之前几轮的 [(问题, 结论摘要)]，或记忆模块压缩好的文字（ConversationMemory.render()）。
        preferences：用户的长期偏好，和 Skill 一起放进规划、写代码、写报告的提示词。"""
        t0 = time.time()
        log: list[dict] = []

        def emit(kind: str, payload: object = None) -> None:
            log.append({"time": round(time.time() - t0, 2), "event": kind, "detail": _brief(payload)})
            if on_event:
                on_event(kind, payload)

        # ① 感知
        prof = profile(df)
        ptext = profile_to_text(prof, df)
        emit("profile", ptext)

        # ② 选择 Skill：根据数据概况和问题，决定加载哪些专项规则
        chosen = select_skills(self.skills, question, prof)
        skills_text = render_skills(chosen) + render_preferences(preferences)
        emit("skills", [(s.name, why) for s, why in chosen])
        if preferences:
            emit("preferences", preferences)

        history_text = _format_history(history)

        # 任务恢复：同一任务有存档就读出来，跳过已完成的规划和步骤
        store = self.checkpoints
        tid = task_id("plan", question, df, history, preferences) if store else ""
        saved = store.load(tid) if store else None

        # ③ 规划（有存档时直接用存档里的计划）
        if saved:
            plan, steps = saved["plan"], saved["steps"]
            emit("resume", {"done": len(steps), "total": len(plan)})
        else:
            plan, steps = self._plan(question, ptext, skills_text, history_text, emit), []
            if store:
                store.save(tid, {"plan": plan, "steps": steps})
        emit("plan", plan)
        resumed = len(steps)

        # ④ 逐步执行；清洗后的数据交给后续步骤共用
        current_df = df
        data_note = f"df 是原始数据（{len(df)} 行），尚未清洗。"
        for rec in steps:                       # 恢复时：按存档里的步骤还原“当前数据”
            current_df, data_note = self._after_step(rec, current_df, data_note, emit)
        for item in plan[len(steps):]:
            rec = StepRecord(id=int(item.get("id", len(steps) + 1)),
                             goal=str(item.get("goal", "")), method=str(item.get("method", "")))
            emit("step_start", rec)
            self._run_step(rec, current_df, data_note, question, ptext, skills_text, steps, emit)
            steps.append(rec)
            current_df, data_note = self._after_step(rec, current_df, data_note, emit)
            emit("step_done", rec)
            if store:
                store.save(tid, {"plan": plan, "steps": steps})   # 每完成一步存一次档

        # ⑤ 报告 → ⑥ 核对数字 → ⑦ 独立审查（有严重矛盾就改一次）
        results_text = format_results(steps)
        report = self._report(question, ptext, results_text, skills_text, history_text, emit)
        known = profile_numbers(prof) + numbers_from_results([s.result for s in steps if s.result])
        known.append(float(len(current_df)))
        report, unverified, review = check_and_review(
            report, known, question, review_context(ptext, results_text), self.llm,
            self.reviewer if self.review else None, emit)
        if store:
            store.clear(tid)                    # 任务完成，删除存档
        return RunResult(question, prof, plan, steps, report, unverified, log,
                         round(time.time() - t0, 1), [(s.name, why) for s, why in chosen],
                         review=review, resumed_steps=resumed)

    @staticmethod
    def _after_step(rec: StepRecord, current_df, data_note, emit):
        """清洗步骤成功后，之后的步骤改用清洗后的数据。"""
        if rec.ok and rec.result.cleaned_df is not None:
            current_df = rec.result.cleaned_df
            data_note = (f"df 是第{rec.id}步清洗后的数据（{len(current_df)} 行），"
                         f"清洗说明：{rec.result.notes} 不要重复清洗。")
            emit("cleaned", {"step": rec.id, "rows": len(current_df)})
        return current_df, data_note

    # ---------- 规划 ----------
    def _plan(self, question, ptext, skills_text, history_text, emit) -> list[dict]:
        reply = self.llm.chat(prompts.PLANNER_SYSTEM, prompts.PLANNER_USER.format(
            question=question, profile=ptext, skills=skills_text, history=history_text))
        emit("llm_plan", reply)
        try:
            data = extract_json(reply)
            steps = data["steps"] if isinstance(data, dict) else data
            steps = [s for s in steps if isinstance(s, dict) and s.get("goal")]
            steps = self._drop_summary_step(steps, emit)
            if steps:
                return steps[: self.max_steps]
        except (ValueError, KeyError, TypeError) as e:
            emit("plan_parse_error", str(e))
        # 兜底：规划解析失败时，直接把整个问题当作一步
        return [{"id": 1, "goal": question, "method": "直接按问题分析"}]

    SUMMARY_WORDS = ("汇总", "综合", "建议", "取舍", "最值得", "结论")

    def _drop_summary_step(self, steps: list[dict], emit) -> list[dict]:
        """程序兜底：最后一步如果只是“汇总/综合判断/给建议”，去掉，交给报告完成。

        提示词里已经要求不要列这类步骤，但模型有时会换个说法（如“汇总指标供报告比较”）。
        这类步骤会把前面算过的指标重算一遍，既慢又容易出错（见调优记录第 5、6 次运行）。
        """
        if len(steps) >= 3 and any(w in str(steps[-1].get("goal", "")) for w in self.SUMMARY_WORDS):
            emit("plan_trimmed", steps[-1].get("goal"))
            return steps[:-1]
        return steps

    # ---------- 单步：写代码 → 检查 → 执行 → 修正 ----------
    def _run_step(self, rec: StepRecord, df, data_note, question, ptext, skills_text,
                  done: list[StepRecord], emit) -> None:
        rec.data_rows = len(df)
        previous = "\n".join(f"第{s.id}步（{s.goal}）：{s.result.notes if s.ok else '失败'}"
                             for s in done) or "（无）"
        reply = self.llm.chat(prompts.CODER_SYSTEM, prompts.CODER_USER.format(
            question=question, step_id=rec.id, goal=rec.goal, method=rec.method,
            data_note=data_note, profile=ptext, previous=previous, skills=skills_text))
        code = extract_code(reply)

        for attempt in range(1, self.max_retries + 2):
            rec.attempts, rec.code = attempt, code
            problems = static_check(code)
            if problems:
                error = "安全检查未通过：" + "；".join(problems)
                rec.result = ExecResult(ok=False, error=error)
            else:
                rec.result = run_code(code, df, timeout=self.timeout)
                error = rec.result.error if not rec.result.ok else (
                    "" if rec.result.has_output else "代码没有把结果写入 result 字典")
            if not error:
                rec.warnings = check_step(rec.result)
                return
            rec.errors.append(error)
            emit("step_error", {"step": rec.id, "attempt": attempt, "error": error})
            if attempt > self.max_retries:
                return
            reply = self.llm.chat(prompts.CODER_SYSTEM,
                                  prompts.FIXER_USER.format(code=code, error=error[-1500:]))
            code = extract_code(reply)

    # ---------- 报告 ----------
    def _report(self, question, ptext, results_text: str, skills_text, history_text,
                emit) -> str:
        reply = self.llm.chat(prompts.REPORTER_SYSTEM, prompts.REPORTER_USER.format(
            question=question, profile=ptext, results=results_text,
            skills=skills_text, history=history_text))
        emit("report", reply)
        return reply.strip()


def format_results(steps: list[StepRecord]) -> str:
    """把各步骤结果整理成文字，交给写报告的模型和审查员（两种模式共用）。"""
    parts = []
    for s in steps:
        if not s.ok:
            parts.append(f"### 第{s.id}步：{s.goal}\n结果：失败（{(s.errors or ['无结果'])[-1][:200]}）")
            continue
        r = s.result
        tables = "\n".join(f"表格「{k}」：\n{v.to_string(max_rows=15)}" for k, v in r.tables.items())
        parts.append(f"### 第{s.id}步：{s.goal}（使用 {s.data_rows} 行数据）\n"
                     f"说明：{r.notes}\n指标：{r.metrics}\n{tables}")
    return "\n\n".join(parts) or "（没有成功的计算步骤）"


def review_context(ptext: str, results_text: str, tool_outputs: list[str] | None = None) -> str:
    """交给审查员的“事实依据”：数据概况 + 计算结果（+ 工具模式下只读工具的返回）。

    数据概况和只读工具的结果都是程序算的，同样是真实数字。只给审查员看计算步骤时，
    它会把直接引用概况的正确报告误判成“编造”（见调优记录：升级 3 的 C11）。
    """
    parts = [f"### 数据概况（程序统计，真实）\n{ptext}", results_text]
    if tool_outputs:
        parts.append("### 只读工具返回的结果（程序统计，真实）\n" + "\n\n".join(tool_outputs))
    return "\n\n".join(parts)


def check_and_review(report: str, known: list[float], question: str, results_text: str,
                     writer: LLM, reviewer: LLM | None, emit):
    """⑥ 程序核对数字出处 → ⑦ 独立审查结论；有严重矛盾时让写报告的模型改一次，再重新核对数字。

    审查本身出错（网络、格式）时只记录，不影响已经写好的报告。
    """
    unverified = verify_report_numbers(report, known)
    emit("verify", unverified)
    if reviewer is None:
        return report, unverified, None
    emit("review_start", None)
    try:
        review = review_report(reviewer, question, results_text, report)
        if review.serious:
            report = revise_report(writer, question, results_text, report, review)
            review.revised = True
            unverified = verify_report_numbers(report, known)
            emit("verify", unverified)
    except Exception as e:  # noqa: BLE001
        review = ReviewResult(passed=True, error=f"审查未完成：{e}")
    emit("review", review)
    return report, unverified, review


def summarize_for_history(run: RunResult, limit: int = 600) -> str:
    """取报告的“结论”部分作为下一轮的对话历史，控制长度。"""
    text = run.report
    if "## 结论" in text:
        text = text.split("## 结论", 1)[1].split("\n## ", 1)[0]
    return text.strip()[:limit]


def _format_history(history: list[tuple[str, str]] | str | None, max_turns: int = 3) -> str:
    if isinstance(history, str):          # 记忆模块已经整理好（含压缩摘要）
        return history or "（这是第一个问题）"
    if not history:
        return "（这是第一个问题）"
    lines = []
    for i, (q, summary) in enumerate(history[-max_turns:], 1):
        lines.append(f"第{i}轮问题：{q}\n第{i}轮结论：{summary}")
    return "\n\n".join(lines)


def _brief(payload: object, limit: int = 2000) -> object:
    """日志里只保留能序列化的简要信息。"""
    if isinstance(payload, StepRecord):
        return {"step": payload.id, "goal": payload.goal, "attempts": payload.attempts,
                "ok": payload.ok, "code": payload.code, "errors": payload.errors,
                "data_rows": payload.data_rows,
                "metrics": payload.result.metrics if payload.result else {}}
    if isinstance(payload, ReviewResult):
        return payload.to_dict()
    if isinstance(payload, (list, dict, int, float)) or payload is None:
        return payload
    return str(payload)[:limit]
