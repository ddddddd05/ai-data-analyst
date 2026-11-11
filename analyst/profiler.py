"""数据概况：读入表格，统计每列的类型、缺失、重复、异常值。

交给大模型的只有这份概况和前几行样例，不会把整张表发出去。
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

DATE_HINTS = ("日期", "时间", "date", "time", "月份")


def load_table(source: Any, filename: str = "") -> pd.DataFrame:
    """读取 csv / xlsx。source 可以是路径，也可以是上传的文件对象。"""
    name = (filename or str(getattr(source, "name", source))).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        raw = source.read() if hasattr(source, "read") else open(source, "rb").read()
        for enc in ("utf-8-sig", "gbk"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法识别文件编码，请另存为 UTF-8 或 GBK 编码的 CSV")
    return parse_dates(df)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """列名像日期、且 80% 以上能解析成日期的列，转成日期类型。"""
    df = df.copy()
    for col in df.columns:
        is_text = df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
        if is_text and any(h in str(col).lower() for h in DATE_HINTS):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
    return df


def _iqr_outliers(s: pd.Series) -> int:
    s = s.dropna()
    if len(s) < 10:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())


def profile(df: pd.DataFrame) -> dict:
    """返回结构化的数据概况。"""
    cols = []
    for col in df.columns:
        s = df[col]
        item = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_rate": round(float(s.isna().mean()), 4),
            "unique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            item.update(kind="数值", min=float(d["min"]), max=float(d["max"]),
                        mean=round(float(d["mean"]), 4), median=float(s.median()),
                        negative=int((s < 0).sum()), outliers=_iqr_outliers(s))
        elif pd.api.types.is_datetime64_any_dtype(s):
            item.update(kind="日期", min=str(s.min().date()), max=str(s.max().date()))
        else:
            top = s.value_counts().head(5)
            item.update(kind="文本/类别", top={str(k): int(v) for k, v in top.items()})
        cols.append(item)
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": cols,
    }


def profile_to_text(p: dict, df: pd.DataFrame | None = None, sample_rows: int = 5) -> str:
    """把概况整理成给大模型看的中文文字。"""
    lines = [f"共 {p['rows']} 行、{p['cols']} 列，完全重复的行 {p['duplicate_rows']} 行。", "各列情况："]
    for c in p["columns"]:
        base = f"- {c['name']}（{c['kind']}，{c['dtype']}）：缺失 {c['missing']}，不同取值 {c['unique']}"
        if c["kind"] == "数值":
            base += (f"，最小 {c['min']:g}，最大 {c['max']:g}，均值 {c['mean']:g}，中位数 {c['median']:g}"
                     f"，负数 {c['negative']} 个，极端值 {c['outliers']} 个")
        elif c["kind"] == "日期":
            base += f"，范围 {c['min']} ~ {c['max']}"
        else:
            base += f"，常见取值 {c['top']}"
        lines.append(base)
    if df is not None and sample_rows:
        lines.append("前几行样例：")
        lines.append(df.head(sample_rows).to_string(index=False))
    return "\n".join(lines)


def profile_numbers(p: dict) -> list[float]:
    """概况里出现的数字，供报告校验使用。"""
    nums: list[float] = [p["rows"], p["cols"], p["duplicate_rows"]]
    for c in p["columns"]:
        nums += [c["missing"], c["missing_rate"], c["unique"]]
        for k in ("min", "max", "mean", "median", "negative", "outliers"):
            v = c.get(k)
            if isinstance(v, (int, float, np.number)):
                nums.append(float(v))
        nums += list((c.get("top") or {}).values())
    return [float(x) for x in nums]
