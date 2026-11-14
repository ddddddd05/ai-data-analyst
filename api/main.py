"""后端：FastAPI 接口。前端（React）通过这些网址调用 Agent。

启动：.venv\\Scripts\\uvicorn api.main:app --port 8000
接口文档（自动生成，可直接在网页上测试）：http://localhost:8000/docs

| 接口 | 作用 |
|---|---|
| GET  /api/config | 模型名称、是否配置了密钥（不返回密钥本身） |
| POST /api/sessions | 新建一个对话（会话） |
| POST /api/sessions/{id}/sample | 加载示例数据 |
| POST /api/sessions/{id}/upload | 上传 CSV / Excel |
| GET  /api/sessions/{id} | 当前数据概况、前 50 行、对话摘要 |
| POST /api/sessions/{id}/ask | 提问。用 SSE（服务器推送事件）一边分析一边把进度推给前端，最后推送完整结果 |
| POST /api/sessions/{id}/clear | 清空对话（短期记忆） |
| GET / POST / DELETE /api/preferences | 长期偏好的查看、新增、删除 |
| GET  /api/runs/{run_id}/report.html、/log.json | 下载报告和运行日志 |

会话保存在后端内存里（重启后端就没了），适合单人本地使用。
"""
from __future__ import annotations

import io
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analyst.agent import DataAnalystAgent, RunResult, summarize_for_history
from analyst.checkpoint import CheckpointStore
from analyst.config import ROOT, get_llm_config, get_reviewer_config
from analyst.knowledge import default_knowledge
from analyst.memory import ConversationMemory, PreferenceStore, extract_preferences
from analyst.profiler import load_table, profile
from analyst.report import to_html, to_log_json
from analyst.tool_agent import ToolCallingAgent

from . import serialize

SAMPLE = ROOT / "sample_data" / "模拟电商订单.csv"
FRONTEND_DIST = ROOT / "frontend" / "dist"


@dataclass
class Session:
    df: pd.DataFrame | None = None
    data_name: str = ""
    memory: ConversationMemory = field(default_factory=lambda: ConversationMemory(keep_recent=3))
    lock: threading.Lock = field(default_factory=threading.Lock)   # 同一个对话一次只跑一个问题


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str = Field("plan", pattern="^(plan|tools|mcp)$")   # plan 规划-执行；tools 工具调用；mcp 工具经 MCP
    max_retries: int = Field(2, ge=0, le=3)
    max_tool_calls: int = Field(15, ge=5, le=25)
    review: bool = True
    knowledge: bool = True                                    # 检索业务知识库（RAG）


class PreferenceRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=50)
    content: str = Field(min_length=1, max_length=300)


def create_app(llm_factory=None, reviewer_factory=None, prefs: PreferenceStore | None = None,
               checkpoints: CheckpointStore | None = None, serve_frontend: bool = True) -> FastAPI:
    """llm_factory 等参数只在测试时传入（用 FakeLLM 代替真实模型）。"""
    app = FastAPI(title="电商经营分析 Agent API",
                  description="上传订单表，用中文提问，Agent 自动清洗、计算、写报告并核对数字。")
    sessions: dict[str, Session] = {}
    runs: dict[str, RunResult] = {}
    store = prefs if prefs is not None else PreferenceStore()
    ckpt = checkpoints if checkpoints is not None else CheckpointStore()

    def make_llm():
        if llm_factory:
            return llm_factory()
        from analyst.llm import OpenAICompatLLM
        cfg = get_llm_config()
        if not cfg.ready:
            raise HTTPException(400, "后端没有配置 API Key，请在项目根目录的 .env 里填写 LLM_API_KEY")
        return OpenAICompatLLM(cfg)

    def make_reviewer():
        if reviewer_factory:
            return reviewer_factory()
        if llm_factory:
            return None
        from analyst.llm import OpenAICompatLLM
        rcfg = get_reviewer_config()
        return OpenAICompatLLM(rcfg) if rcfg else None

    def get_session(sid: str) -> Session:
        if sid not in sessions:
            raise HTTPException(404, "对话不存在（后端可能重启过），请刷新页面")
        return sessions[sid]

    def session_info(s: Session) -> dict:
        info = {"data_name": s.data_name, "memory_summary": s.memory.summary,
                "compressed_turns": s.memory.compressed_turns, "data": None}
        if s.df is not None:
            p = profile(s.df)
            info["data"] = {"rows": p["rows"], "cols": p["cols"], "duplicate_rows": p["duplicate_rows"],
                            "columns": serialize.plain([{k: v for k, v in c.items() if k != "top"}
                                                        for c in p["columns"]]),
                            "preview": serialize.table(s.df, 50)}
        return info

    # ---------------- 配置与会话 ----------------
    @app.get("/api/config")
    def config():
        cfg = get_llm_config()
        rcfg = get_reviewer_config(cfg)
        return {"model": cfg.model, "ready": cfg.ready or bool(llm_factory),
                "reviewer_model": rcfg.model if rcfg else cfg.model}

    @app.post("/api/sessions")
    def new_session():
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = Session()
        return {"session_id": sid}

    @app.get("/api/sessions/{sid}")
    def get_info(sid: str):
        return session_info(get_session(sid))

    @app.post("/api/sessions/{sid}/sample")
    def use_sample(sid: str):
        s = get_session(sid)
        s.df, s.data_name, s.memory = load_table(SAMPLE), SAMPLE.name, ConversationMemory(keep_recent=3)
        return session_info(s)

    @app.post("/api/sessions/{sid}/upload")
    async def upload(sid: str, file: UploadFile = File(...)):
        s = get_session(sid)
        name = file.filename or "data.csv"
        if Path(name).suffix.lower() not in (".csv", ".xlsx", ".xls"):
            raise HTTPException(400, "只支持 .csv / .xlsx / .xls 文件")
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(400, "文件太大（上限 50MB）")
        try:
            df = load_table(io.BytesIO(content), name)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"读取失败：{e}") from e
        s.df, s.data_name, s.memory = df, name, ConversationMemory(keep_recent=3)
        return session_info(s)

    @app.post("/api/sessions/{sid}/clear")
    def clear(sid: str):
        s = get_session(sid)
        s.memory = ConversationMemory(keep_recent=3)
        return session_info(s)

    # ---------------- 提问：SSE 流式推送进度 ----------------
    @app.post("/api/sessions/{sid}/ask")
    def ask(sid: str, req: AskRequest):
        s = get_session(sid)
        if s.df is None:
            raise HTTPException(400, "请先上传数据或使用示例数据")
        llm = make_llm()
        if not s.lock.acquire(blocking=False):
            raise HTTPException(409, "上一个问题还在分析中，请稍等")
        events: queue.Queue = queue.Queue()

        def send(kind: str, data) -> None:
            events.put((kind, data))

        def work() -> None:
            """在后台线程里跑 Agent；进度事件放进队列，由下面的生成器推给前端。"""
            try:
                question = req.question.strip()
                new_prefs, has_question = extract_preferences(llm, question)
                for pf in new_prefs:
                    store.add(pf["topic"], pf["content"])
                if new_prefs:
                    send("preferences_saved", {"items": new_prefs, "analysis": has_question})
                if new_prefs and not has_question:
                    return
                reviewer = make_reviewer()
                kb = default_knowledge() if req.knowledge else None
                if req.mode in ("tools", "mcp"):
                    agent = ToolCallingAgent(llm, max_tool_calls=req.max_tool_calls, reviewer=reviewer,
                                             review=req.review, checkpoints=ckpt,
                                             tool_source="mcp" if req.mode == "mcp" else "builtin",
                                             knowledge=kb)
                else:
                    agent = DataAnalystAgent(llm, max_retries=req.max_retries, reviewer=reviewer,
                                             review=req.review, checkpoints=ckpt, knowledge=kb)

                def on_event(kind, payload):
                    text = serialize.event_text(kind, payload)
                    if text:
                        send("progress", {"text": text})

                run = agent.run(s.df, question, on_event=on_event, history=s.memory.render(),
                                preferences=store.contents())
                compressed = s.memory.add_turn(question, summarize_for_history(run), llm=llm)
                rid = uuid.uuid4().hex[:12]
                runs[rid] = run
                send("result", {**serialize.run(rid, run), "memory_compressed": compressed,
                                "memory_summary": s.memory.summary})
            except Exception as e:  # noqa: BLE001
                send("error", {"message": f"调用模型失败：{e}",
                               "resumable": True})   # 已完成的步骤已存档，重新提问同一问题会从断点继续
            finally:
                send("done", None)
                s.lock.release()

        threading.Thread(target=work, daemon=True).start()

        def stream():
            while True:
                kind, data = events.get()
                yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if kind == "done":
                    break

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------- 长期偏好 ----------------
    @app.get("/api/preferences")
    def list_prefs():
        return [p.__dict__ for p in store.items]

    @app.post("/api/preferences")
    def add_pref(req: PreferenceRequest):
        store.add(req.topic.strip(), req.content.strip())
        return [p.__dict__ for p in store.items]

    @app.delete("/api/preferences/{topic}")
    def delete_pref(topic: str):
        store.remove(topic)
        return [p.__dict__ for p in store.items]

    # ---------------- 下载 ----------------
    def get_run(rid: str) -> RunResult:
        if rid not in runs:
            raise HTTPException(404, "找不到这次分析（后端可能重启过）")
        return runs[rid]

    @app.get("/api/runs/{rid}/report.html", response_class=HTMLResponse)
    def report_html(rid: str):
        return HTMLResponse(to_html(get_run(rid)),
                            headers={"Content-Disposition": f'attachment; filename="report_{rid}.html"'})

    @app.get("/api/runs/{rid}/log.json")
    def report_log(rid: str):
        return Response(to_log_json(get_run(rid)), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="log_{rid}.json"'})

    # 打包好的前端（npm run build 之后）直接由后端提供，只开一个端口就能用
    if serve_frontend and FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


app = create_app()
