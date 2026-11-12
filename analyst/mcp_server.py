"""MCP 服务器：把本项目的数据分析工具开放给 Claude 桌面软件、Claude Code 等支持 MCP 的 AI 使用。

MCP（Model Context Protocol，模型上下文协议）是一个“通用插座”标准：
工具按 MCP 的格式写一次，任何支持 MCP 的 AI 应用都能直接插上用，不用为每个应用单独对接。

| | 网页版（app.py） | MCP 服务器（本文件） |
|---|---|---|
| 谁是“大脑” | DeepSeek | 调用方自己的模型（如 Claude） |
| 本项目提供什么 | 完整的 Agent | 工具 + 护栏（安全检查、子进程执行、数字核对），也可以把整个 Agent 当一个工具调用 |

提供的工具和权限（沿用 tools.py 的等级，并用 MCP 的工具注解告诉调用方）：
| 工具 | 作用 | 注解 |
|---|---|---|
| load_data | 加载 CSV / Excel（只允许示例数据目录或 ANALYST_DATA_DIR 指定的目录） | 只读 |
| data_overview / preview_data / describe_column | 查看数据 | 只读 |
| list_skills / load_skill | 查看、加载 Skill（数据清洗规范、电商指标口径） | 只读 |
| run_python | 运行分析代码：先过静态安全检查，再在独立子进程运行，有超时 | 非只读，但不改动任何文件 |
| verify_report | 核对一份报告里的数字，在本次会话的计算结果里有没有出处 | 只读 |
| analyze | 让本项目的 DeepSeek Agent 完整分析一个问题（需要 .env 里的 API Key） | 只读，会访问外部模型接口 |
| reset_data | 撤销清洗，回到原始数据 | 非只读 |
不提供：写文件、删文件、联网下载、执行系统命令。

运行方式：python mcp_server.py（标准输入输出 stdio 方式，由 Claude 等客户端自动启动，配置见 README）。
注意：stdio 方式下标准输出被 MCP 协议占用，程序里不能 print。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from .config import ROOT
from .profiler import load_table, profile, profile_numbers, profile_to_text
from .skills import load_skills
from .tools import ToolBox
from .validator import numbers_from_results, verify_report_numbers

SAMPLE_DIR = ROOT / "sample_data"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
RUN_CODE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False,
                           open_world_hint=False)

INSTRUCTIONS = """电商经营分析工具箱（订单数据）。建议流程：
1. load_data 加载数据（不传路径则加载示例数据：模拟跨境电商渔具订单），再看 data_overview；
2. 概况里有重复、缺失、负数、极端值时，先 load_skill("数据清洗")，按规则用 run_python 清洗，把清洗后的数据放进 result["cleaned_df"]，之后的 df 自动变成清洗后的数据；
3. 涉及销售额、退货率、ROAS 等指标时先 load_skill("电商指标")，按其中口径计算；
4. 所有数字用 run_python 真实算出，写完报告后用 verify_report 核对数字出处；
5. 也可以直接用 analyze 让本项目自带的 Agent 完整分析一个问题。"""


class Session:
    """一个 MCP 连接对应一份会话状态：当前数据、工具箱、本次会话算出的所有结果。"""

    def __init__(self):
        self.name = ""
        self.raw: pd.DataFrame | None = None
        self.box: ToolBox | None = None
        self.results = []          # run_python 成功的结果，供 verify_report 核对
        self.extra_numbers = []    # 只读工具返回的数字

    def require(self) -> ToolBox:
        if self.box is None:
            self.load(None)        # 没加载过就自动加载示例数据
        return self.box

    def load(self, path: str | None) -> str:
        file = _resolve_data_path(path)
        self.raw = load_table(file)
        self.name = file.name
        self.box = ToolBox(self.raw, load_skills())
        self.results, self.extra_numbers = [], []
        return file.name


def _allowed_dirs() -> list[Path]:
    dirs = [SAMPLE_DIR.resolve()]
    if os.getenv("ANALYST_DATA_DIR"):
        dirs.append(Path(os.environ["ANALYST_DATA_DIR"]).expanduser().resolve())
    return dirs


def _resolve_data_path(path: str | None) -> Path:
    """权限：只能读允许目录里的 csv / xlsx / xls 文件，防止被用来读取电脑上的其他文件。"""
    if not path:
        return SAMPLE_DIR / "模拟电商订单.csv"
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = SAMPLE_DIR / p
    p = p.resolve()
    if p.suffix.lower() not in (".csv", ".xlsx", ".xls"):
        raise ValueError("只能加载 .csv / .xlsx / .xls 文件")
    if not any(p.is_relative_to(d) for d in _allowed_dirs()):
        raise ValueError(f"没有权限读取这个位置。允许的目录：{[str(d) for d in _allowed_dirs()]}；"
                         "如需分析自己的数据，请在 MCP 配置里设置环境变量 ANALYST_DATA_DIR。")
    if not p.exists():
        raise ValueError(f"文件不存在：{p}")
    return p


def _ok_or_raise(out) -> str:
    """失败时抛 ToolError：客户端收到的结果会带 is_error 标记（MCP 协议里表示“工具执行失败”的标准方式）。"""
    if not out.ok:
        raise ToolError(out.text)
    return out.text


def build_server(session: Session | None = None) -> MCPServer:
    s = session or Session()
    mcp = MCPServer(name="ecommerce-analyst", title="电商经营分析工具箱", instructions=INSTRUCTIONS)

    @mcp.tool(annotations=READ_ONLY)
    def load_data(path: str = "") -> str:
        """加载要分析的订单表（CSV/Excel）。不传 path 则加载示例数据；只允许示例数据目录或 ANALYST_DATA_DIR 下的文件。"""
        try:
            name = s.load(path or None)
        except ValueError as e:
            raise ToolError(f"加载失败：{e}") from e
        return f"已加载 {name}：{len(s.raw)} 行 × {s.raw.shape[1]} 列。\n\n" + profile_to_text(profile(s.raw), s.raw)

    @mcp.tool(annotations=READ_ONLY)
    def data_overview() -> str:
        """当前数据的概况：行列数、每列类型、缺失、重复、负数、极端值和前 5 行样例。"""
        box = s.require()
        state = "（清洗后）" if box.cleaned else "（原始数据）"
        return f"{s.name}{state}\n" + profile_to_text(profile(box.df), box.df)

    @mcp.tool(annotations=READ_ONLY)
    def preview_data(n: int = 5, columns: list[str] | None = None) -> str:
        """预览当前数据的前 n 行（1～20），可只看指定的列。"""
        return _ok_or_raise(s.require().call("preview_data", {"n": n, "columns": columns}))

    @mcp.tool(annotations=READ_ONLY)
    def describe_column(column: str) -> str:
        """查看某一列：类型、缺失、不同取值；数字列给分位数，文字列给最常见取值。"""
        out = s.require().call("describe_column", {"column": column})
        s.extra_numbers += out.numbers
        return _ok_or_raise(out)

    @mcp.tool(annotations=READ_ONLY)
    def list_skills() -> str:
        """列出可加载的 Skill（专项规则）及适用场景。"""
        return "\n".join(f"- {k.name}：{k.description}" for k in load_skills())

    @mcp.tool(annotations=READ_ONLY)
    def load_skill(name: str) -> str:
        """加载一个 Skill 的规则全文（如“数据清洗”“电商指标”），之后计算和写报告都要遵守。"""
        return _ok_or_raise(s.require().call("load_skill", {"name": name}))

    @mcp.tool(annotations=RUN_CODE)
    def run_python(code: str, purpose: str = "") -> CallToolResult:
        """运行一段 pandas 分析代码。已有 df、pd、np、plt；结果写进字典 result：
        result['metrics']={'指标名': 数字}、result['tables']={'表名': DataFrame}、result['notes']='说明'；
        清洗时把清洗后的完整数据放进 result['cleaned_df']，之后的 df 自动变成清洗后的数据。
        代码先过安全检查（只能导入 pandas/numpy/matplotlib/math/statistics/datetime/re，不能读写文件、联网），再在独立进程中运行，超时 60 秒。"""
        box = s.require()
        out = box.call("run_python", {"code": code, "purpose": purpose})
        if out.ok and out.exec_result is not None:
            s.results.append(out.exec_result)
        text = _ok_or_raise(out)
        # 文字给模型看；结构化数据（指标、表格）给程序用，客户端可以据此核对数字
        r = out.exec_result
        structured = {"metrics": r.metrics, "notes": r.notes, "rows": len(box.df), "cleaned": box.cleaned,
                      "tables": {k: json.loads(v.reset_index().to_json(orient="split", date_format="iso",
                                                                       force_ascii=False))
                                 for k, v in r.tables.items()}}
        return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)

    @mcp.tool(annotations=READ_ONLY)
    def verify_report(report: str) -> str:
        """核对报告里的数字：在本次会话的计算结果和数据概况里找不到出处的数字会被列出来。"""
        box = s.require()
        known = profile_numbers(profile(s.raw)) + numbers_from_results(s.results) + s.extra_numbers
        known.append(float(len(box.df)))
        bad = verify_report_numbers(report, known)
        if not bad:
            return "核对通过：报告中的数字都能在计算结果中找到出处。"
        return "以下数字在计算结果里找不到出处，请确认是否需要先用 run_python 计算：" + "、".join(bad)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def analyze(question: str, mode: str = "plan") -> str:
        """让本项目自带的 Agent（DeepSeek）完整分析一个问题，返回带数字核对和独立审查的报告。
        mode：plan（规划-执行）或 tools（工具调用）。需要项目 .env 里配置 API Key。"""
        from .agent import DataAnalystAgent
        from .config import get_llm_config
        from .llm import OpenAICompatLLM
        from .tool_agent import ToolCallingAgent

        cfg = get_llm_config()
        if not cfg.ready:
            return "没有配置 API Key（项目 .env 里的 LLM_API_KEY），无法调用自带 Agent；可以改用其他工具自己分析。"
        s.require()
        llm = OpenAICompatLLM(cfg)
        agent = ToolCallingAgent(llm) if mode == "tools" else DataAnalystAgent(llm)
        run = agent.run(s.raw, question)
        note = ("数字核对通过" if not run.unverified
                else "以下数字找不到出处，请人工核对：" + "、".join(run.unverified))
        review = ""
        if run.review and run.review.revised:
            review = "；独立审查发现矛盾，报告已修改一次"
        return f"{run.report}\n\n---\n（{mode} 模式，{len(run.steps)} 步，用时 {run.seconds} 秒；{note}{review}）"

    @mcp.tool(annotations=RUN_CODE)
    def reset_data() -> str:
        """撤销清洗，回到刚加载时的原始数据，并清空本次会话的计算结果。"""
        if s.raw is None:
            return "还没有加载数据。"
        s.box = ToolBox(s.raw, load_skills())
        s.results, s.extra_numbers = [], []
        return f"已恢复原始数据（{len(s.raw)} 行）。"

    return mcp


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
