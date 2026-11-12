"""工具箱：工具模式下，模型可以自己选择调用的工具，每个工具都有权限等级。

权限等级（越往下越危险，管得越严）：
| 等级 | 含义 | 工具 |
|---|---|---|
| 只读 read | 只看数据，不改任何东西，直接执行 | preview_data、describe_column、load_skill |
| 执行代码 exec | 运行模型写的 Python，必须先过 sandbox 的静态安全检查，再放进子进程运行 | run_python |
| 结束 finish | 提交最终报告，结束本轮 | submit_report |
| 不提供 | 读写文件、联网、执行系统命令——工具箱里根本没有这些工具，模型想调也调不到 | — |

模型“点名”要用某个工具时，由 ToolBox.call 统一检查：工具存不存在、权限等级是否被允许、
参数对不对，然后才真正执行。所有调用结果都以文字形式返回给模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from .sandbox import ExecResult, run_code, static_check
from .skills import Skill

READ, EXEC, FINISH = "read", "exec", "finish"
LEVEL_NAMES = {READ: "只读", EXEC: "执行代码", FINISH: "结束"}
NOT_PROVIDED = ["读写文件", "联网", "执行系统命令"]


@dataclass
class Tool:
    name: str
    level: str
    description: str
    parameters: dict

    def schema(self) -> dict:
        """OpenAI 兼容接口 tools 参数要求的格式。"""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


TOOLS = [
    Tool("preview_data", READ, "预览当前数据的前几行（只读）。清洗后预览的是清洗后的数据。",
         {"type": "object", "properties": {
             "n": {"type": "integer", "description": "行数，1～20，默认 5"},
             "columns": {"type": "array", "items": {"type": "string"}, "description": "只看这些列，可省略"}}}),
    Tool("describe_column", READ,
         "查看某一列的详细情况（只读）：类型、缺失、不同取值个数；数字列给出分位数，文字列给出最常见的取值。",
         {"type": "object", "properties": {"column": {"type": "string", "description": "列名"}},
          "required": ["column"]}),
    Tool("load_skill", READ, "加载一个 Skill（专项规则），返回规则全文。之后写代码和写报告都必须遵守。",
         {"type": "object", "properties": {"name": {"type": "string", "description": "Skill 名称"}},
          "required": ["name"]}),
    Tool("run_python", EXEC,
         "运行一段 Python 分析代码。已有变量 df、pd、np、plt；把结果写进字典 result："
         "result['metrics']={'指标名': 数字}、result['tables']={'表名': DataFrame}、result['notes']='说明'；"
         "清洗数据时把清洗后的完整数据放进 result['cleaned_df']，之后的 df 自动变成清洗后的数据。"
         "代码会先经过安全检查（只能导入 pandas、numpy、matplotlib、math、statistics、datetime、re，"
         "不能读写文件、不能联网），再在独立进程里运行。",
         {"type": "object", "properties": {
             "purpose": {"type": "string", "description": "这段代码要算什么（一句话）"},
             "code": {"type": "string", "description": "完整的 Python 代码"}},
          "required": ["purpose", "code"]}),
    Tool("submit_report", FINISH, "提交最终的 Markdown 分析报告，结束本轮分析。",
         {"type": "object", "properties": {"report": {"type": "string", "description": "完整报告"}},
          "required": ["report"]}),
]


@dataclass
class ToolOutcome:
    ok: bool
    text: str                              # 返回给模型的文字
    exec_result: ExecResult | None = None  # run_python 的完整结果
    numbers: list[float] = field(default_factory=list)  # 只读工具里出现的真实数字，供报告核对


class ToolBox:
    def __init__(self, df: pd.DataFrame, skills: list[Skill], timeout: int = 60,
                 allowed_levels: set[str] | None = None):
        self.df = df
        self.skills = {s.name: s for s in skills}
        self.timeout = timeout
        self.allowed = allowed_levels or {READ, EXEC, FINISH}
        self.tools = {t.name: t for t in TOOLS}
        self.loaded_skills: list[str] = []
        self.cleaned = False

    def schemas(self) -> list[dict]:
        """只把允许的工具告诉模型。"""
        return [t.schema() for t in self.tools.values() if t.level in self.allowed]

    # ---------- 统一入口：先查权限，再执行 ----------
    def call(self, name: str, args: dict) -> ToolOutcome:
        tool = self.tools.get(name)
        if tool is None:
            return ToolOutcome(False, f"没有工具“{name}”。本 Agent 不提供{'、'.join(NOT_PROVIDED)}的工具。")
        if tool.level not in self.allowed:
            return ToolOutcome(False, f"权限不足：工具“{name}”属于“{LEVEL_NAMES[tool.level]}”等级，当前不允许使用。")
        if "_bad_json" in args:
            return ToolOutcome(False, "参数不是合法的 JSON，请重新调用。")
        try:
            return getattr(self, "_" + name)(**args)
        except TypeError as e:
            return ToolOutcome(False, f"参数错误：{e}")

    # ---------- 只读工具 ----------
    def _preview_data(self, n: int = 5, columns: list[str] | None = None) -> ToolOutcome:
        n = max(1, min(int(n), 20))
        data = self.df
        if columns:
            missing = [c for c in columns if c not in data.columns]
            if missing:
                return ToolOutcome(False, f"没有这些列：{missing}。现有列：{list(data.columns)}")
            data = data[columns]
        state = "清洗后" if self.cleaned else "原始"
        return ToolOutcome(True, f"{state}数据共 {len(self.df)} 行，前 {n} 行：\n{data.head(n).to_string()}")

    def _describe_column(self, column: str) -> ToolOutcome:
        if column not in self.df.columns:
            return ToolOutcome(False, f"没有列“{column}”。现有列：{list(self.df.columns)}")
        s = self.df[column]
        lines = [f"列“{column}”：类型 {s.dtype}，共 {len(s)} 行，缺失 {int(s.isna().sum())}，"
                 f"不同取值 {int(s.nunique())} 个"]
        nums = [float(len(s)), float(s.isna().sum()), float(s.nunique())]
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4)
            lines.append(desc.to_string())
            nums += [float(v) for v in desc.values]
        else:
            vc = s.value_counts().head(15)
            lines.append("最常见的取值：\n" + vc.to_string())
            nums += [float(v) for v in vc.values]
        return ToolOutcome(True, "\n".join(lines), numbers=nums)

    def _load_skill(self, name: str) -> ToolOutcome:
        skill = self.skills.get(name)
        if skill is None:
            return ToolOutcome(False, f"没有 Skill“{name}”。可用的有：{list(self.skills)}")
        if name not in self.loaded_skills:
            self.loaded_skills.append(name)
        return ToolOutcome(True, f"### Skill：{skill.name}\n{skill.body}")

    # ---------- 执行代码：必须先过安全检查 ----------
    def _run_python(self, code: str, purpose: str = "") -> ToolOutcome:
        problems = static_check(code)
        if problems:
            err = "安全检查未通过，代码没有运行：" + "；".join(problems)
            return ToolOutcome(False, err, ExecResult(ok=False, error=err))
        res = run_code(code, self.df, timeout=self.timeout)
        if not res.ok:
            return ToolOutcome(False, "运行出错：\n" + res.error[-1500:], res)
        if not res.has_output:
            res.ok = False
            res.error = "代码没有把结果写入 result 字典"
            return ToolOutcome(False, res.error, res)
        parts = [f"运行成功（使用 {len(self.df)} 行数据）。", f"说明：{res.notes}", f"指标：{res.metrics}"]
        parts += [f"表格「{k}」：\n{v.to_string(max_rows=15)}" for k, v in res.tables.items()]
        if res.figures:
            parts.append(f"生成图表 {len(res.figures)} 张。")
        if res.cleaned_df is not None:
            self.df, self.cleaned = res.cleaned_df, True
            parts.append(f"已保存清洗后的数据（{len(self.df)} 行），之后所有工具使用的 df 都是清洗后的数据，不要重复清洗。")
        return ToolOutcome(True, "\n".join(parts)[:4000], res)

    def _submit_report(self, report: str) -> ToolOutcome:
        return ToolOutcome(True, "报告已提交。")
