"""MCP 客户端：让本项目的 Agent 通过 MCP 协议使用工具——既能连自己的 MCP 服务器，也能连别人写的。

| | 内置工具箱（tools.py） | MCP 工具箱（本文件） |
|---|---|---|
| 工具从哪来 | 同一个程序里写好的 5 个工具，直接调用 | 启动时连接 mcp_servers.json 里的服务器，问它“你有哪些工具” |
| 想加新工具 | 改代码 | 配置里加一个服务器 |
| 速度 | 快 | 多一层进程间通信 |

流程：
    启动服务器子进程 → 读取工具清单（名称、说明、参数格式、只读标记）
    → 转成 DeepSeek 能懂的 tools 参数 → 模型决定调用哪个工具
    → 通过 MCP 转发给服务器执行 → 结果交回模型

权限（别人写的工具不能无条件信任）：
| 情况 | 处理 |
|---|---|
| 工具标了只读（read_only_hint=True） | 直接开放 |
| 没标只读（可能改东西） | 默认不开放；在配置 allow_tools 里写明才开放 |
| 配置 hide_tools 里的工具 | 不开放（如 load_data：数据由 Agent 统一加载，不让模型去读别的文件；analyze：避免 Agent 调用自己形成套娃） |
服务器执行出错时返回 is_error 标记，Agent 当作工具失败处理。

MCP 客户端是异步的，而 Agent 是一步一步同步执行的：这里用一个后台线程常驻连接，
Agent 每次调用工具时把请求放进队列、等结果返回，整个分析过程只连接一次。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import ROOT
from .sandbox import ExecResult
from .tools import EXEC, FINISH, READ, Tool, ToolOutcome

CONFIG_FILE = ROOT / "mcp_servers.json"
SUBMIT = Tool("submit_report", FINISH, "提交最终的 Markdown 分析报告，结束本轮分析。",
              {"type": "object", "properties": {"report": {"type": "string", "description": "完整报告"}},
               "required": ["report"]})
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class MCPConnection:
    """在后台线程里保持一个 MCP 连接。server 可以是 StdioServerParameters（启动子进程），
    也可以是 MCPServer 对象（测试时在同一进程里连接）。"""

    def __init__(self, server, timeout: float = 120):
        self.server, self.timeout = server, timeout
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("连接 MCP 服务器超时")
        if self._error:
            raise RuntimeError(f"连接 MCP 服务器失败：{self._error}") from self._error

    async def _main(self) -> None:
        from mcp import Client
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        try:
            # 连接的打开和关闭必须在同一个协程里完成，所以所有请求都排队交给这里处理
            async with Client(self.server) as client:
                self._ready.set()
                while True:
                    item = await self._queue.get()
                    if item is None:
                        break
                    fut, method, args = item
                    try:
                        fut.set_result(await getattr(client, method)(*args))
                    except Exception as e:  # noqa: BLE001
                        fut.set_exception(e)
        except Exception as e:  # noqa: BLE001
            self._error = e
            self._ready.set()

    def request(self, method: str, *args):
        if self._error:
            raise RuntimeError(f"MCP 连接已断开：{self._error}")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (fut, method, args))
        return fut.result(self.timeout)

    def close(self) -> None:
        if self._loop and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            self._thread.join(10)


@dataclass
class ServerSpec:
    name: str
    server: object                      # StdioServerParameters 或 MCPServer
    data_tool: str = ""                 # 用来加载数据的工具名（Agent 启动时调用一次）
    hide_tools: list[str] = field(default_factory=list)
    allow_tools: list[str] = field(default_factory=list)
    code_tools: list[str] = field(default_factory=list)   # 运行代码的工具，结果整理成“步骤”


def load_config(path: Path = CONFIG_FILE, env: dict | None = None) -> list[ServerSpec]:
    from mcp import StdioServerParameters
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for name, c in data.get("servers", {}).items():
        cmd = sys.executable if c["command"] in ("python", "python3") else c["command"]
        args = [str(ROOT / a) if a.endswith(".py") and not Path(a).is_absolute() else a
                for a in c.get("args", [])]
        params = StdioServerParameters(command=cmd, args=args, cwd=str(ROOT),
                                       env={**os.environ, **c.get("env", {}), **(env or {})})
        specs.append(ServerSpec(name, params, c.get("data_tool", ""), c.get("hide_tools", []),
                                c.get("allow_tools", []), c.get("code_tools", [])))
    return specs


class MCPToolBox:
    """和 tools.ToolBox 用法一样（schemas / call / tools / rows），只是工具来自 MCP 服务器。"""

    def __init__(self, df: pd.DataFrame, specs: list[ServerSpec] | None = None, timeout: float = 120):
        self.df = df
        self.rows = len(df)               # 当前数据行数（服务器清洗后会更新）
        self.cleaned = False
        self.loaded_skills: list[str] = []
        self.tools: dict[str, Tool] = {"submit_report": SUBMIT}
        self.hidden: dict[str, str] = {}  # 没开放的工具及原因，供界面和日志展示
        self._route: dict[str, tuple[MCPConnection, ServerSpec]] = {}
        self._conns: list[MCPConnection] = []
        self._tmp = tempfile.TemporaryDirectory(prefix="agent_mcp_")
        try:
            self._connect(specs if specs is not None else load_config(env={"ANALYST_DATA_DIR": self._tmp.name}),
                          timeout)
        except Exception:
            self.close()
            raise

    def _connect(self, specs: list[ServerSpec], timeout: float) -> None:
        data_file = Path(self._tmp.name) / "data.csv"
        for spec in specs:
            conn = MCPConnection(spec.server, timeout)
            self._conns.append(conn)
            for t in conn.request("list_tools").tools:
                read_only = bool(t.annotations and t.annotations.read_only_hint)
                if t.name in spec.hide_tools:
                    self.hidden[t.name] = "配置中隐藏"
                    continue
                if not read_only and t.name not in spec.allow_tools:
                    self.hidden[t.name] = "不是只读工具，且没有在 allow_tools 中允许"
                    continue
                if t.name in self.tools:
                    self.hidden[f"{spec.name}.{t.name}"] = "和已有工具重名"
                    continue
                self.tools[t.name] = Tool(t.name, READ if read_only else EXEC,
                                          t.description or "", t.input_schema or {"type": "object"})
                self._route[t.name] = (conn, spec)
            if spec.data_tool:
                # Agent 手里的数据先存成临时 CSV，再让服务器加载（服务器只被允许读这个临时目录）
                if not data_file.exists():
                    self.df.to_csv(data_file, index=False)
                res = conn.request("call_tool", spec.data_tool, {"path": str(data_file)})
                if res.is_error:
                    raise RuntimeError(f"MCP 服务器 {spec.name} 加载数据失败：{_text(res)}")

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> ToolOutcome:
        if name == "submit_report":
            return ToolOutcome(True, "报告已提交。")
        if name not in self._route:
            why = self.hidden.get(name, "没有这个工具")
            return ToolOutcome(False, f"不能调用“{name}”：{why}。")
        if "_bad_json" in args:
            return ToolOutcome(False, "参数不是合法的 JSON，请重新调用。")
        conn, spec = self._route[name]
        try:
            res = conn.request("call_tool", name, args)
        except Exception as e:  # noqa: BLE001
            return ToolOutcome(False, f"调用 MCP 工具出错：{e}")
        text = _text(res)
        ok = not res.is_error
        numbers = [float(x) for x in NUM_RE.findall(text)] if ok else []   # 工具返回的真实数字，供报告核对
        if ok and name == "load_skill" and args.get("name") and args["name"] not in self.loaded_skills:
            self.loaded_skills.append(args["name"])
        data = getattr(res, "structured_content", None) or {}
        if ok and "rows" in data:                       # 服务器给了结构化数据：直接用
            self.rows, self.cleaned = int(data["rows"]), bool(data.get("cleaned"))
        elif ok and (m := re.search(r"清洗后的数据（(\d+) 行）", text)):   # 没给就从文字里找
            self.rows, self.cleaned = int(m.group(1)), True
        exec_result = None
        if name in spec.code_tools:
            exec_result = _to_exec_result(ok, text, data)
        return ToolOutcome(ok, text[:4000], exec_result, numbers)

    def close(self) -> None:
        for c in self._conns:
            c.close()
        self._conns = []
        self._tmp.cleanup()


def _to_exec_result(ok: bool, text: str, data: dict) -> ExecResult:
    """把代码工具的返回还原成 ExecResult。有结构化数据时还原指标和表格（数字核对、评测都能用），
    没有时只保留文字说明（第三方服务器常见情况）。"""
    if not ok:
        return ExecResult(ok=False, error=text)
    tables = {}
    for k, t in (data.get("tables") or {}).items():
        try:
            tables[k] = pd.DataFrame(t["data"], columns=t["columns"])
        except (KeyError, TypeError, ValueError):
            continue
    metrics = {str(k): (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
               for k, v in (data.get("metrics") or {}).items()}
    return ExecResult(ok=True, metrics=metrics, tables=tables,
                      notes=str(data.get("notes") or text[:1500]))


def _text(res) -> str:
    return "\n".join(getattr(c, "text", "") for c in res.content).strip()
