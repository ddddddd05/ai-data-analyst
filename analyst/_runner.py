"""在独立子进程里运行 AI 生成的代码（由 sandbox.run_code 调用，不直接使用）。

输入：工作目录里的 in.pkl（数据）和 code.py（代码）
输出：out.pkl（结果字典）和 fig_*.png（图表）
"""
import contextlib
import io
import pickle
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def to_number(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def main(workdir: Path) -> None:
    df = pd.read_pickle(workdir / "in.pkl")
    code = (workdir / "code.py").read_text(encoding="utf-8")
    ns = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "result": {}}
    out = {"ok": True, "error": "", "stdout": ""}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, "<AI生成的代码>", "exec"), ns)
    except Exception:
        out["ok"] = False
        out["error"] = traceback.format_exc(limit=3)
    out["stdout"] = buf.getvalue()[-3000:]

    result = ns.get("result") if isinstance(ns.get("result"), dict) else {}
    out["metrics"] = {str(k): to_number(v) for k, v in (result.get("metrics") or {}).items()}
    tables = {}
    for k, v in (result.get("tables") or {}).items():
        if isinstance(v, pd.Series):
            v = v.to_frame()
        if isinstance(v, pd.DataFrame):
            tables[str(k)] = v.head(50)
    out["tables"] = tables
    out["notes"] = str(result.get("notes", ""))[:1000]
    cleaned = result.get("cleaned_df")
    out["has_cleaned"] = isinstance(cleaned, pd.DataFrame) and not cleaned.empty
    if out["has_cleaned"]:
        cleaned.to_pickle(workdir / "cleaned.pkl")  # 清洗后的数据交给后续步骤使用

    figures = []
    for i, num in enumerate(plt.get_fignums()[:4]):
        path = workdir / f"fig_{i}.png"
        plt.figure(num).savefig(path, dpi=120, bbox_inches="tight")
        figures.append(path.name)
    out["figures"] = figures
    with open(workdir / "out.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
