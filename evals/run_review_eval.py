"""审查员专项评测：审查员能不能抓出“结论与数字矛盾”的报告，又会不会误报正确的报告。

用法：python evals/run_review_eval.py
题目在 evals/review_cases.json：5 份故意埋错的报告 + 3 份正确报告，用同一份真实步骤结果。
结果追加到 docs/评测报告/ 下的一份 Markdown。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.config import get_llm_config, get_reviewer_config  # noqa: E402
from analyst.llm import OpenAICompatLLM  # noqa: E402
from analyst.reviewer import review_report  # noqa: E402

CASES = ROOT / "evals" / "review_cases.json"
OUT_DIR = ROOT / "docs" / "评测报告"


def run(llm, spec: dict, log=print) -> list[dict]:
    rows = []
    for c in spec["cases"]:
        t0 = time.time()
        r = review_report(llm, c["question"], spec["results"], c["report"])
        found = bool(r.serious)
        rows.append({"id": c["id"], "kind": c["kind"], "expect": c["expect_serious"], "found": found,
                     "correct": found == c["expect_serious"], "issues": r.issues, "error": r.error,
                     "seconds": round(time.time() - t0, 1)})
        log(f"[{c['id']}] {c['kind']}：应{'报' if c['expect_serious'] else '不报'}严重问题，"
            f"实际{'报了' if found else '没报'} → {'✅' if found == c['expect_serious'] else '❌'}")
    return rows


def main() -> None:
    cfg = get_llm_config()
    if not cfg.ready:
        sys.exit("没有配置 API Key，请先在 .env 里填写")
    rcfg = get_reviewer_config(cfg) or cfg
    spec = json.loads(CASES.read_text(encoding="utf-8"))
    rows = run(OpenAICompatLLM(rcfg), spec)
    bad = [r for r in rows if r["expect"]]
    good = [r for r in rows if not r["expect"]]
    caught = sum(r["found"] for r in bad)
    false_alarm = sum(r["found"] for r in good)
    lines = [f"# 审查员专项评测", "", f"- 时间：{datetime.now():%Y-%m-%d %H:%M}", f"- 审查员模型：{rcfg.model}",
             f"- **埋错的报告抓出：{caught} / {len(bad)}**", f"- **正确报告误报：{false_alarm} / {len(good)}**", "",
             "| 编号 | 类型 | 应报严重问题 | 实际 | 结果 | 审查意见 |", "|---|---|---|---|---|---|"]
    for r in rows:
        note = "；".join(f"[{i['severity']}] {i['claim']} ← {i['evidence']}" for i in r["issues"]) or r["error"] or "—"
        lines.append(f"| {r['id']} | {r['kind']} | {'是' if r['expect'] else '否'} | "
                     f"{'报了' if r['found'] else '没报'} | {'✅' if r['correct'] else '❌'} | "
                     f"{note.replace('|', '/').replace(chr(10), ' ')} |")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M}_审查员专项评测.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n抓出 {caught}/{len(bad)}，误报 {false_alarm}/{len(good)}，报告：{out}")


if __name__ == "__main__":
    main()
