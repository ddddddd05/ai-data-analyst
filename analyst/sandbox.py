"""工具：安全检查 + 执行 AI 写的代码（Agent 的“手脚”和第一道护栏）。

两层保护：
1. 静态检查：运行前分析代码结构，拦截导入危险模块、读写文件、执行系统命令等写法；
2. 子进程执行：代码在单独的进程里跑，有超时限制，出错或卡死不影响主程序。

注意：这只是演示级别的防护，不等于真正的安全沙箱。不要拿它运行来路不明的代码。
"""
from __future__ import annotations

import ast
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RUNNER = Path(__file__).with_name("_runner.py")

ALLOWED_MODULES = {"pandas", "numpy", "matplotlib", "math", "statistics", "datetime", "re"}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
    "vars", "setattr", "delattr", "breakpoint", "exit", "quit",
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "requests", "urllib",
}
BLOCKED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_json", "to_sql", "to_html",
    "to_feather", "to_hdf", "to_clipboard", "savefig", "show",
    "read_csv", "read_excel", "read_pickle", "read_sql", "read_json", "read_html",
    "read_parquet", "read_clipboard", "system", "popen", "remove", "unlink", "rmdir",
}


def static_check(code: str) -> list[str]:
    """返回发现的问题列表；空列表表示通过。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"语法错误：第 {e.lineno} 行 {e.msg}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    problems.append(f"不允许导入模块 {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                problems.append(f"不允许导入模块 {node.module}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            problems.append(f"不允许使用 {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"不允许访问双下划线属性 {node.attr}")
            elif node.attr in BLOCKED_ATTRS:
                problems.append(f"不允许调用 {node.attr}（禁止读写文件或显示窗口）")
    return sorted(set(problems))


@dataclass
class ExecResult:
    ok: bool
    error: str = ""
    stdout: str = ""
    metrics: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    notes: str = ""
    figures: list = field(default_factory=list)  # PNG 图片的字节内容
    cleaned_df: pd.DataFrame | None = None       # 清洗步骤产出的数据，后续步骤共用

    @property
    def has_output(self) -> bool:
        return bool(self.metrics or self.tables or self.figures or self.notes
                    or self.cleaned_df is not None)


def run_code(code: str, df: pd.DataFrame, timeout: int = 60) -> ExecResult:
    """在子进程里运行代码，返回结果。不做静态检查，调用前请先 static_check。"""
    with tempfile.TemporaryDirectory(prefix="agent_step_") as tmp:
        work = Path(tmp)
        df.to_pickle(work / "in.pkl")
        (work / "code.py").write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(RUNNER), str(work)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, error=f"运行超过 {timeout} 秒，已终止（可能有死循环或计算量过大）")
        out_file = work / "out.pkl"
        if not out_file.exists():
            return ExecResult(ok=False, error="执行进程异常退出：\n" + (proc.stderr or "")[-2000:])
        with open(out_file, "rb") as f:
            out = pickle.load(f)
        figures = [(work / name).read_bytes() for name in out.get("figures", [])]
        cleaned = pd.read_pickle(work / "cleaned.pkl") if out.get("has_cleaned") else None
        return ExecResult(
            ok=out["ok"], error=out["error"], stdout=out["stdout"],
            metrics=out["metrics"], tables=out["tables"], notes=out["notes"], figures=figures,
            cleaned_df=cleaned,
        )
