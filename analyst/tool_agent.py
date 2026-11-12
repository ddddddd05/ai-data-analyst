"""工具模式的 Agent：不预先规划，由模型自己决定下一步调用哪个工具（Function Calling）。

和规划-执行模式（agent.py）的区别：
| | 规划-执行模式 | 工具模式 |
|---|---|---|
| 谁决定下一步 | 程序：先规划，再按步骤执行 | 模型：每轮看结果，自己选工具 |
| 优点 | 流程固定、稳定、好调试 | 灵活，可以先看某一列再决定怎么算 |
| 风险 | 规划错了只能硬着头皮做 | 可能绕圈、反复调用 → 用调用次数上限兜底 |

Agent Loop（一轮循环）：
    把对话 + 工具清单发给模型 → 模型回复“要调用某工具 + 参数”
    → ToolBox 检查权限并执行 → 把结果作为 tool 消息放回对话 → 再问模型……
    直到模型调用 submit_report，或达到工具调用次数上限（强制写报告）。

报告仍然交给 validator 核对数字出处，和规划-执行模式一样。
"""
from __future__ import annotations

import json
import time
from typing import Callable

import pandas as pd

from . import prompts
from .agent import RunResult, StepRecord, _brief, _format_history
from .llm import ToolLLM
from .profiler import profile, profile_numbers, profile_to_text
from .skills import Skill, load_skills
from .tools import FINISH, LEVEL_NAMES, ToolBox
from .validator import check_step, numbers_from_results, verify_report_numbers


class ToolCallingAgent:
    def __init__(self, llm: ToolLLM, max_tool_calls: int = 15, timeout: int = 60,
                 skills: list[Skill] | None = None, allowed_levels: set[str] | None = None):
        self.llm = llm
        self.max_tool_calls = max_tool_calls
        self.timeout = timeout
        self.skills = load_skills() if skills is None else skills
        self.allowed_levels = allowed_levels

    def run(self, df: pd.DataFrame, question: str,
            on_event: Callable[[str, object], None] | None = None,
            history: list[tuple[str, str]] | None = None) -> RunResult:
        t0 = time.time()
        log: list[dict] = []

        def emit(kind: str, payload: object = None) -> None:
            log.append({"time": round(time.time() - t0, 2), "event": kind, "detail": _brief(payload)})
            if on_event:
                on_event(kind, payload)

        prof = profile(df)
        ptext = profile_to_text(prof, df)
        emit("profile", ptext)
        box = ToolBox(df, self.skills, timeout=self.timeout, allowed_levels=self.allowed_levels)
        skill_list = "\n".join(f"- {s.name}：{s.description}" for s in self.skills) or "（无）"
        messages = [
            {"role": "system", "content": prompts.TOOL_AGENT_SYSTEM},
            {"role": "user", "content": prompts.TOOL_AGENT_USER.format(
                history=_format_history(history), question=question, profile=ptext,
                skill_list=skill_list)},
        ]

        steps: list[StepRecord] = []
        open_step: StepRecord | None = None   # run_python 失败后，下一次 run_python 算作同一步的重试
        trace: list[dict] = []
        extra_numbers: list[float] = []
        report, nudged, used = "", False, 0

        # 循环次数也设上限：防止模型一直只回文字、不调工具
        for _ in range(self.max_tool_calls + 3):
            reply = self.llm.chat_tools(messages, box.schemas())
            messages.append(reply["raw"])
            if not reply["tool_calls"]:
                if len(reply["content"]) > 80 or nudged:   # 模型直接写了报告
                    report = reply["content"]
                    break
                nudged = True
                messages.append({"role": "user", "content": prompts.TOOL_NUDGE})
                continue

            finished = False
            for call in reply["tool_calls"]:
                name, args = call["name"], call["arguments"]
                if used >= self.max_tool_calls:
                    # 每个 tool_call 都必须有对应的 tool 消息，超限的也要回复“拒绝”
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": "已达到工具调用次数上限，本次调用被拒绝。"})
                    continue
                used += 1
                level = box.tools[name].level if name in box.tools else "unknown"
                emit("tool_call", {"n": used, "name": name, "level": LEVEL_NAMES.get(level, "不存在"),
                                   "args": _short_args(args)})
                rows_before = len(box.df)
                out = box.call(name, args)
                trace.append({"n": used, "name": name, "level": LEVEL_NAMES.get(level, "不存在"),
                              "args": _short_args(args), "ok": out.ok, "result": out.text[:500]})
                emit("tool_result", {"n": used, "name": name, "ok": out.ok, "text": out.text[:300]})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": out.text})
                extra_numbers += out.numbers

                if name == "load_skill" and out.ok:
                    emit("skills", [(args.get("name"), "模型调用 load_skill 加载")])
                elif name == "run_python" and out.exec_result is not None:
                    open_step = self._record_step(open_step, steps, args, out, rows_before,
                                                   len(box.df), emit)
                elif name == "submit_report" and out.ok and level == FINISH:
                    report, finished = str(args.get("report", "")), True
            if finished:
                break
            if used >= self.max_tool_calls:
                emit("tool_limit", self.max_tool_calls)
                messages.append({"role": "user", "content": prompts.TOOL_LIMIT_REACHED.format(
                    limit=self.max_tool_calls)})
                report = self.llm.chat_tools(messages, [])["content"]
                break

        known = profile_numbers(prof) + numbers_from_results([s.result for s in steps if s.result])
        known += extra_numbers + [float(len(box.df))]
        unverified = verify_report_numbers(report, known)
        emit("verify", unverified)
        plan = [{"id": s.id, "goal": s.goal, "method": "工具模式：run_python"} for s in steps]
        return RunResult(question, prof, plan, steps, report.strip(), unverified, log,
                         round(time.time() - t0, 1),
                         [(n, "模型调用 load_skill 加载") for n in box.loaded_skills],
                         mode="tools", tool_calls=trace)

    @staticmethod
    def _record_step(open_step, steps, args, out, rows, rows_after, emit) -> StepRecord | None:
        """把 run_python 调用整理成“步骤”，失败后紧接着的重试算同一步（和规划模式的重试对齐）。"""
        rec = open_step
        if rec is None:
            rec = StepRecord(id=len(steps) + 1, goal=str(args.get("purpose", "") or "运行代码"),
                             method="工具模式：run_python")
            steps.append(rec)
        rec.attempts += 1
        rec.code, rec.result, rec.data_rows = str(args.get("code", "")), out.exec_result, rows
        if out.ok:
            rec.warnings = check_step(out.exec_result)
            emit("step_done", rec)
            if out.exec_result.cleaned_df is not None:
                emit("cleaned", {"step": rec.id, "rows": rows_after})
            return None
        rec.errors.append(out.text)
        emit("step_error", {"step": rec.id, "attempt": rec.attempts, "error": out.text})
        return rec


def _short_args(args: dict, limit: int = 300) -> dict:
    """日志和界面里只显示参数的前一部分（代码可能很长）。"""
    return {k: (v if len(json.dumps(v, ensure_ascii=False)) <= limit else str(v)[:limit] + "……")
            for k, v in args.items()}
