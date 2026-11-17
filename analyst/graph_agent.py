"""规划-执行模式的 LangGraph 版本（升级 12）。

手写版（agent.py 的 DataAnalystAgent._run）用 for 循环 + if 判断控制流程；
这里把同一套流程画成一张“状态图”，交给 LangGraph 执行：

    START → 感知 → 规划 ─┬→ 写代码 → 执行 ─┬→ 收尾本步 ─┐
                         │        ↑        │(出错且还能重试)│
                         │      修代码 ←───┘              │
                         │                                │
                         └──(还有没做的步骤)←─────────────┘
                         └──(步骤全部做完)→ 写报告 → 核对与审查 → END

三个 LangGraph 概念：
- State（状态）：在各节点之间传递的数据，相当于一张“共享表单”。每个节点读表单、只返回自己改的字段。
- Node（节点）：一个普通函数，做一件事（如“写代码”）。
- Edge（边）：节点之间的连线。条件边（conditional edge）根据状态决定下一步去哪，
  “出错就回到修代码”“还有步骤就继续写代码”都是条件边，手写版里是 for/if。

断点续跑：传入 checkpoints 时用 LangGraph 的 SqliteSaver（检查点保存器），每个节点执行完
自动存一次状态；中途失败后同一任务再次运行，从最后一个完成的节点继续。手写版的
CheckpointStore 要自己决定“什么时候存、存什么、怎么还原”，这里由框架负责。

模型调用、安全检查、子进程执行、数字核对、独立审查全部复用手写版的函数，
两个版本只有“流程怎么控制”不同，因此可以用同一套评测题直接对比。
"""
from __future__ import annotations

import copy
import pickle
import sqlite3
import time
from typing import Any, Callable, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from . import prompts, usage
from .agent import (DataAnalystAgent, RunResult, StepRecord, _brief, _format_history,
                    check_and_review, format_results, review_context, run_tracked)
from .checkpoint import CHECKPOINT_DIR, task_id
from .knowledge import knowledge_numbers, render_knowledge
from .llm import extract_code
from .memory import render_preferences
from .profiler import profile, profile_numbers, profile_to_text
from .sandbox import ExecResult, run_code, static_check
from .skills import render_skills, select_skills
from .validator import check_step, numbers_from_results

GRAPH_DB = CHECKPOINT_DIR / "langgraph.sqlite"


class AnalysisState(TypedDict, total=False):
    """在节点之间传递的状态。没有写 reducer（合并规则），节点返回的字段直接覆盖旧值。"""
    question: str
    history_text: str
    prof: dict
    ptext: str
    skills: list            # [(Skill 名称, 加载原因)]
    skills_text: str        # Skill + 偏好 + 检索到的业务知识，交给各角色
    kb_text: str
    kb_hits: list           # 检索到的知识片段（字典）
    kb_numbers: list        # 知识片段里的数字，核对报告时算“有出处”
    plan: list              # [{"id", "goal", "method"}]
    steps: list             # 已完成的 StepRecord
    cur: Any                # 正在做的 StepRecord（写代码 → 执行 → 修代码 之间传递）
    report: str
    unverified: list
    review: Any


class PickleSerializer:
    """检查点的序列化方式：直接用 pickle。

    状态里有 pandas 表格和 StepRecord 这类自定义对象，LangGraph 默认的 JSON/msgpack 序列化不认识。
    存档只在本机 outputs/checkpoints 下读写（和手写版 CheckpointStore 一样用 pickle），不接收外部文件。
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "pickle", pickle.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return pickle.loads(data[1])


def make_checkpointer(path=GRAPH_DB):
    """SQLite 检查点保存器。check_same_thread=False：后端在后台线程里运行 Agent。"""
    from langgraph.checkpoint.sqlite import SqliteSaver
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=PickleSerializer())


class LangGraphAnalystAgent(DataAnalystAgent):
    """和手写版同样的参数、同样的返回值（RunResult），mode="graph"。"""

    def __init__(self, *args, checkpointer=None, **kwargs):
        super().__init__(*args, **kwargs)
        # 传了 checkpoints（手写版的存档开关）就自动启用 LangGraph 的 SQLite 检查点
        # （存在同一个存档文件夹里的 langgraph.sqlite）
        self.checkpointer = checkpointer or (
            make_checkpointer(self.checkpoints.folder / GRAPH_DB.name) if self.checkpoints else None)

    def run(self, df: pd.DataFrame, question: str,
            on_event: Callable[[str, object], None] | None = None,
            history: list[tuple[str, str]] | str | None = None,
            preferences: list[str] | None = None) -> RunResult:
        return run_tracked(self._run_graph, df, question, on_event, history, preferences)

    # ---------- 建图 ----------
    def build_graph(self, df: pd.DataFrame, preferences: list[str] | None, emit):
        """每次提问建一张图：原始数据 df 不放进状态（避免每个检查点都存一份大表格），
        由节点函数直接引用；当前该用哪份数据，按已完成的步骤推算（清洗过就用清洗后的）。"""

        def current_data(steps: list[StepRecord]):
            cur_df, note = df, f"df 是原始数据（{len(df)} 行），尚未清洗。"
            for rec in steps:
                cur_df, note = self._after_step(rec, cur_df, note, lambda *_: None)
            return cur_df, note

        # ---- 节点：每个函数读状态，只返回要更新的字段 ----
        def perceive(s: AnalysisState) -> dict:
            prof = profile(df)
            ptext = profile_to_text(prof, df)
            emit("profile", ptext)
            chosen = select_skills(self.skills, s["question"], prof)
            skills_text = render_skills(chosen) + render_preferences(preferences)
            emit("skills", [(sk.name, why) for sk, why in chosen])
            if preferences:
                emit("preferences", preferences)
            hits = self.knowledge.search(s["question"]) if self.knowledge else []
            kb_text = render_knowledge(hits)
            if self.knowledge:
                emit("knowledge", [h.to_dict() for h in hits])
            return {"prof": prof, "ptext": ptext, "skills": [(sk.name, why) for sk, why in chosen],
                    "skills_text": skills_text + kb_text, "kb_text": kb_text,
                    "kb_hits": [h.to_dict() for h in hits], "kb_numbers": knowledge_numbers(hits),
                    "steps": []}

        def plan(s: AnalysisState) -> dict:
            p = self._plan(s["question"], s["ptext"], s["skills_text"], s["history_text"], emit)
            emit("plan", p)
            return {"plan": p}

        def write_code(s: AnalysisState) -> dict:
            steps = s["steps"]
            item = s["plan"][len(steps)]
            rec = StepRecord(id=int(item.get("id", len(steps) + 1)),
                             goal=str(item.get("goal", "")), method=str(item.get("method", "")))
            emit("step_start", rec)
            cur_df, note = current_data(steps)
            rec.data_rows = len(cur_df)
            previous = "\n".join(f"第{x.id}步（{x.goal}）：{x.result.notes if x.ok else '失败'}"
                                 for x in steps) or "（无）"
            with usage.stage("写代码"):
                reply = self.llm.chat(prompts.CODER_SYSTEM, prompts.CODER_USER.format(
                    question=s["question"], step_id=rec.id, goal=rec.goal, method=rec.method,
                    data_note=note, profile=s["ptext"], previous=previous, skills=s["skills_text"]))
            rec.code = extract_code(reply)
            return {"cur": rec}

        def execute(s: AnalysisState) -> dict:
            rec = copy.copy(s["cur"])
            rec.errors = list(rec.errors)
            rec.attempts += 1
            problems = static_check(rec.code)
            if problems:
                error = "安全检查未通过：" + "；".join(problems)
                rec.result = ExecResult(ok=False, error=error)
            else:
                rec.result = run_code(rec.code, current_data(s["steps"])[0], timeout=self.timeout)
                error = rec.result.error if not rec.result.ok else (
                    "" if rec.result.has_output else "代码没有把结果写入 result 字典")
            if error:
                rec.errors.append(error)
                emit("step_error", {"step": rec.id, "attempt": rec.attempts, "error": error})
            else:
                rec.warnings = check_step(rec.result)
            return {"cur": rec}

        def fix_code(s: AnalysisState) -> dict:
            rec = copy.copy(s["cur"])
            with usage.stage("修代码"):
                reply = self.llm.chat(prompts.CODER_SYSTEM, prompts.FIXER_USER.format(
                    code=rec.code, error=rec.errors[-1][-1500:]))
            rec.code = extract_code(reply)
            return {"cur": rec}

        def finish_step(s: AnalysisState) -> dict:
            rec = s["cur"]
            steps = s["steps"] + [rec]
            if rec.ok and rec.result.cleaned_df is not None:
                emit("cleaned", {"step": rec.id, "rows": len(rec.result.cleaned_df)})
            emit("step_done", rec)
            return {"steps": steps, "cur": None}

        def report(s: AnalysisState) -> dict:
            text = self._report(s["question"], s["ptext"], format_results(s["steps"]),
                                s["skills_text"], s["history_text"], emit)
            return {"report": text}

        def check_review(s: AnalysisState) -> dict:
            steps = s["steps"]
            results_text = format_results(steps)
            known = profile_numbers(s["prof"]) + numbers_from_results(
                [x.result for x in steps if x.result])
            known.append(float(len(current_data(steps)[0])))
            known += s["kb_numbers"]
            text, unverified, review = check_and_review(
                s["report"], known, s["question"],
                review_context(s["ptext"], results_text, knowledge_text=s["kb_text"]),
                self.llm, self.reviewer if self.review else None, emit)
            return {"report": text, "unverified": unverified, "review": review}

        # ---- 条件边：根据状态决定下一步 ----
        def next_step_or_report(s: AnalysisState) -> str:
            return "写代码" if len(s["steps"]) < len(s["plan"]) else "写报告"

        def after_execute(s: AnalysisState) -> str:
            rec = s["cur"]
            if rec.ok or rec.attempts > self.max_retries:   # 成功，或重试次数用完
                return "收尾本步"
            return "修代码"

        g = StateGraph(AnalysisState)
        for name, fn in [("感知", perceive), ("规划", plan), ("写代码", write_code),
                         ("执行", execute), ("修代码", fix_code), ("收尾本步", finish_step),
                         ("写报告", report), ("核对与审查", check_review)]:
            g.add_node(name, fn)
        g.add_edge(START, "感知")
        g.add_edge("感知", "规划")
        g.add_conditional_edges("规划", next_step_or_report, ["写代码", "写报告"])
        g.add_edge("写代码", "执行")
        g.add_conditional_edges("执行", after_execute, ["收尾本步", "修代码"])
        g.add_edge("修代码", "执行")
        g.add_conditional_edges("收尾本步", next_step_or_report, ["写代码", "写报告"])
        g.add_edge("写报告", "核对与审查")
        g.add_edge("核对与审查", END)
        return g.compile(checkpointer=self.checkpointer)

    # ---------- 运行 ----------
    def _run_graph(self, df, question, on_event, history, preferences) -> RunResult:
        t0 = time.time()
        log: list[dict] = []

        def emit(kind: str, payload: object = None) -> None:
            log.append({"time": round(time.time() - t0, 2), "event": kind, "detail": _brief(payload)})
            if on_event:
                on_event(kind, payload)

        graph = self.build_graph(df, preferences, emit)
        # 每个节点算一步；一个分析步骤最多走 写代码 + (执行 + 修代码) × 重试 + 执行 + 收尾
        limit = 10 + self.max_steps * (3 + 2 * (self.max_retries + 1))
        config: dict = {"recursion_limit": limit}
        resumed = 0
        inputs: dict | None = {"question": question, "history_text": _format_history(history)}
        tid = ""
        if self.checkpointer is not None:
            tid = task_id("graph", question, df, history, preferences)
            config["configurable"] = {"thread_id": tid}
            snap = graph.get_state(config)
            if snap.next:                          # 有没跑完的存档：传 None 表示“从断点继续”
                inputs = None
                done = snap.values.get("steps") or []
                resumed = len(done)
                emit("resume", {"done": resumed, "total": len(snap.values.get("plan") or []),
                                "next": list(snap.next)})
                if snap.values.get("plan"):
                    emit("plan", snap.values["plan"])

        final = graph.invoke(inputs, config)
        if tid:
            self.checkpointer.delete_thread(tid)   # 任务完成，删除存档
        return RunResult(question, final["prof"], final["plan"], final["steps"], final["report"],
                         final.get("unverified", []), log, round(time.time() - t0, 1),
                         final.get("skills", []), mode="graph", review=final.get("review"),
                         resumed_steps=resumed, knowledge=final.get("kb_hits", []))


def graph_mermaid(agent: LangGraphAnalystAgent | None = None) -> str:
    """导出流程图（Mermaid 格式），写文档用。"""
    a = agent or LangGraphAnalystAgent(llm=None, skills=[])   # 建图不需要真的调用模型
    return a.build_graph(pd.DataFrame({"x": [1]}), None, lambda *_: None).get_graph().draw_mermaid()
