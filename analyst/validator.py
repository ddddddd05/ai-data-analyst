"""校验兜底：检查每步结果，核对报告里的数字是否真的被算出来过。

大模型写报告时可能“顺手”编一个数字或自己心算。这里把报告中的数字逐个
和真实计算结果比对，对不上的列出来，提醒人工复核。
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .sandbox import ExecResult

NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(-?\d{1,3}(?:,\d{3})+|-?\d+)(\.\d+)?\s*(%|万)?")
DATE_RE = re.compile(r"\d{4}[-/年.]\d{1,2}(?:[-/月.]\d{1,2}日?)?")


def check_step(res: ExecResult) -> list[str]:
    """检查单步结果是否可用。"""
    warnings = []
    if res.ok and not res.has_output:
        warnings.append("代码运行成功，但没有写入任何结果")
    for k, v in res.metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            warnings.append(f"指标“{k}”不是有效数字（{v}）")
    for name, t in res.tables.items():
        if t.empty:
            warnings.append(f"表格“{name}”是空的")
    return warnings


def numbers_from_results(results: list[ExecResult], limit_per_table: int = 2000) -> list[float]:
    """收集所有步骤里真实算出来的数字。"""
    nums: list[float] = []
    for r in results:
        nums += [v for v in r.metrics.values() if isinstance(v, float)]
        for t in r.tables.values():
            vals = pd.DataFrame(t).reset_index().select_dtypes("number").to_numpy().ravel()
            nums += [float(x) for x in vals[:limit_per_table] if np.isfinite(x)]
            nums.append(float(len(t)))
    return nums


def _matches(value: float, decimals: int, known: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - k) <= tol or (k and abs(value - k) / abs(k) <= 0.005) for k in known)


def verify_report_numbers(report: str, known: list[float]) -> list[str]:
    """返回报告里找不到出处的数字（原文写法）。"""
    text = DATE_RE.sub(" ", report)  # 日期不参与核对
    unverified = []
    for m in NUM_RE.finditer(text):
        int_part, frac, unit = m.group(1), m.group(2) or "", m.group(3)
        value = float(int_part.replace(",", "") + frac)
        decimals = len(frac) - 1 if frac else 0
        if not frac and not unit and (abs(value) <= 10 or 1900 <= value <= 2100):
            continue  # 小整数（步骤号、条数）和年份不核对
        if unit == "%":
            ok = _matches(value / 100, decimals + 2, known) or _matches(value, decimals, known)
        elif unit == "万":
            ok = _matches(value * 10000, decimals - 4, known)
        else:
            ok = _matches(value, decimals, known)
            # 表头写了“(%)”、数字本身不带 % 号的情况：7.16 也可能对应 0.0716
            if not ok and frac:
                ok = _matches(value / 100, decimals + 2, known)
        if not ok:
            unverified.append(m.group(0).strip())
    return list(dict.fromkeys(unverified))  # 去重并保持顺序
