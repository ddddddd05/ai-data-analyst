"""评测集：用一套固定的问题“考”Agent，自动打分，生成评测报告。

用法：
    python evals/run_eval.py                 # 跑全部题目（需要 .env 里的 API Key）
    python evals/run_eval.py --only C01 C07  # 只跑指定题目
    python evals/run_eval.py --label 加入评测前   # 给这次评测起个名字，写进报告

每道题在 evals/cases.json 里有若干“检查标准”，全部通过才算这道题通过。
结果保存到 docs/评测报告/，每次一份 Markdown（给人看）和一份 JSON（给程序对比）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.agent import DataAnalystAgent, RunResult  # noqa: E402
from analyst.profiler import load_table  # noqa: E402
from analyst.validator import numbers_from_results  # noqa: E402

CASES = ROOT / "evals" / "cases.json"
OUT_DIR = ROOT / "docs" / "评测报告"


# ---------------- 检查标准 ----------------
# 每个检查函数返回 (是否通过, 说明)

def _all_numbers(run: RunResult) -> list[float]:
    return numbers_from_results([s.result for s in run.steps if s.result])


def chk_skills_include(run, names):
    loaded = [n for n, _ in run.skills]
    missing = [n for n in names if n not in loaded]
    return not missing, f"加载了 {loaded or '无'}" + (f"，缺少 {missing}" if missing else "")


def chk_no_failed_steps(run, flag):
    failed = [f"第{s.id}步" for s in run.steps if not s.ok]
    return (not failed) if flag else True, ("全部步骤成功" if not failed else f"失败：{'、'.join(failed)}")


def chk_max_steps(run, n):
    return len(run.steps) <= n, f"共 {len(run.steps)} 步（上限 {n}）"


def chk_result_has_number(run, items):
    nums = _all_numbers(run)
    misses = []
    for it in items:
        if not any(abs(x - v) <= it.get("tol", 0) + 1e-9 for v in it["values"] for x in nums):
            misses.append(f"{it['label']}（应为 {it['values'][0]}）")
    return not misses, "计算结果正确" if not misses else "计算结果里找不到：" + "；".join(misses)


def chk_report_contains(run, groups):
    """groups：[[A, B], [C]] 表示（A 或 B）并且 C。"""
    misses = [" / ".join(g) for g in groups if not any(w in run.report for w in g)]
    return not misses, "报告包含要求的内容" if not misses else "报告缺少：" + "；".join(misses)


def chk_report_not_contains(run, words):
    hits = [w for w in words if w in run.report]
    return not hits, "没有出现禁止内容" if not hits else "报告出现了：" + "、".join(hits)


def chk_max_unverified(run, n):
    return len(run.unverified) <= n, f"找不到出处的数字 {len(run.unverified)} 个（上限 {n}）：{run.unverified}"


def chk_max_seconds(run, n):
    return run.seconds <= n, f"用时 {run.seconds} 秒（上限 {n}）"


def chk_code_not_contains(run, words):
    # 只看真正运行成功的代码；被安全检查拦下的代码不算
    code = "\n".join(s.code for s in run.steps if s.ok)
    hits = [w for w in words if w in code]
    return not hits, "运行的代码里没有危险操作" if not hits else "运行的代码出现了：" + "、".join(hits)


CHECKS = {
    "skills_include": ("加载 Skill", chk_skills_include),
    "no_failed_steps": ("步骤成功", chk_no_failed_steps),
    "max_steps": ("步数", chk_max_steps),
    "result_has_number": ("数字正确", chk_result_has_number),
    "report_contains": ("报告内容", chk_report_contains),
    "report_not_contains": ("禁止内容", chk_report_not_contains),
    "max_unverified": ("数字出处", chk_max_unverified),
    "max_seconds": ("速度", chk_max_seconds),
    "code_not_contains": ("代码安全", chk_code_not_contains),
}


def grade(case: dict, run: RunResult) -> dict:
    details = []
    for key, arg in case["checks"].items():
        name, fn = CHECKS[key]
        ok, msg = fn(run, arg)
        details.append({"check": name, "ok": ok, "detail": msg})
    return {"id": case["id"], "category": case["category"], "question": case["question"],
            "passed": all(d["ok"] for d in details), "checks": details,
            "seconds": run.seconds, "steps": len(run.steps),
            "skills": [n for n, _ in run.skills], "unverified": run.unverified}


def evaluate(agent: DataAnalystAgent, df, cases: list[dict], log=print) -> list[dict]:
    results = []
    for case in cases:
        log(f"[{case['id']}] {case['question']}")
        t0 = time.time()
        try:
            run = agent.run(df, case["question"], history=[tuple(h) for h in case.get("history", [])])
            r = grade(case, run)
            r["report"] = run.report
        except Exception as e:  # noqa: BLE001  运行出错也记一笔，不中断整个评测
            r = {"id": case["id"], "category": case["category"], "question": case["question"],
                 "passed": False, "checks": [{"check": "运行", "ok": False, "detail": f"出错：{e}"}],
                 "seconds": round(time.time() - t0, 1), "steps": 0, "skills": [], "unverified": [],
                 "report": ""}
        log(f"      {'✅ 通过' if r['passed'] else '❌ 未通过'}（{r['seconds']} 秒）")
        results.append(r)
    return results


def to_markdown(results: list[dict], label: str, model: str) -> str:
    n, ok = len(results), sum(r["passed"] for r in results)
    lines = [f"# 评测报告：{label}", "",
             f"- 时间：{datetime.now():%Y-%m-%d %H:%M}",
             f"- 模型：{model}",
             f"- **通过：{ok} / {n}**",
             f"- 平均用时：{sum(r['seconds'] for r in results) / max(n, 1):.1f} 秒", "",
             "## 各题结果", "", "| 编号 | 类型 | 问题 | 结果 | 用时 | 没通过的检查 |", "|---|---|---|---|---|---|"]
    for r in results:
        bad = "；".join(f"{c['check']}：{c['detail']}" for c in r["checks"] if not c["ok"]) or "—"
        lines.append(f"| {r['id']} | {r['category']} | {r['question']} | "
                     f"{'✅' if r['passed'] else '❌'} | {r['seconds']} 秒 | {bad} |")
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    lines += ["", "## 按类型统计", "", "| 类型 | 通过 |", "|---|---|"]
    lines += [f"| {c} | {sum(v)} / {len(v)} |" for c, v in by_cat.items()]
    fails = [r for r in results if not r["passed"]]
    if fails:
        lines += ["", "## 未通过的题（badcase）", ""]
        for r in fails:
            lines.append(f"### {r['id']}：{r['question']}")
            for c in r["checks"]:
                lines.append(f"- {'✅' if c['ok'] else '❌'} {c['check']}：{c['detail']}")
            lines += ["", "<details><summary>报告原文</summary>", "", r.get("report", ""), "", "</details>", ""]
    return "\n".join(lines)


def main() -> None:
    from analyst.config import get_llm_config
    from analyst.llm import OpenAICompatLLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="只跑这些题号")
    ap.add_argument("--label", default="评测", help="这次评测的名字")
    args = ap.parse_args()

    spec = json.loads(CASES.read_text(encoding="utf-8"))
    cases = [c for c in spec["cases"] if not args.only or c["id"] in args.only]
    df = load_table(ROOT / spec["data"])
    cfg = get_llm_config()
    if not cfg.ready:
        sys.exit("没有配置 API Key，请先在 .env 里填写")
    agent = DataAnalystAgent(OpenAICompatLLM(cfg))

    results = evaluate(agent, df, cases)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    md = OUT_DIR / f"{stamp}_{args.label}.md"
    md.write_text(to_markdown(results, args.label, cfg.model), encoding="utf-8")
    (OUT_DIR / f"{stamp}_{args.label}.json").write_text(
        json.dumps({"label": args.label, "model": cfg.model, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(r["passed"] for r in results)
    print(f"\n通过 {ok} / {len(results)}，报告：{md}")


if __name__ == "__main__":
    main()
