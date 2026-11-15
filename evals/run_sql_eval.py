"""数据库模式（Text-to-SQL）评测：schema linking 专项（不调用大模型）+ 端到端执行准确率（调用大模型）。

用法：
    python evals/run_sql_eval.py --linking-only      # 只评 schema linking，几秒跑完，不花钱
    python evals/run_sql_eval.py                     # 规划-执行模式，全部题目（需要 .env 里的 API Key）
    python evals/run_sql_eval.py --mode tools        # 工具调用模式：模型自己查表结构、写 SQL
    python evals/run_sql_eval.py --only S06 S13 --label 修复扇出

题目在 evals/sql_cases.json。标准答案不手写：每道题带参考 SQL，评测时在数据库上实时算出，
Agent 的计算结果里能找到参考答案（相对误差 0.2% 以内）才算对，即 Text-to-SQL 常用的“执行准确率”。
数据库重新生成后不用改答案。

schema linking 指标：
- 召回率：需要的表有没有全部选中（漏一张表，SQL 就写不出来；选多了只是浪费 token）；
- 精确率：选中的表里有多少是真正需要的；
- 完全正确：选中的表和需要的表完全一致；
- 节省：只给相关的表结构时，提示词里的表结构比全部表少多少字。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.database import Database  # noqa: E402
from analyst.sql_agent import last_turn  # noqa: E402
from evals.run_eval import grade, to_markdown  # noqa: E402

CASES = ROOT / "evals" / "sql_cases.json"
OUT_DIR = ROOT / "docs" / "评测报告"
REL_TOL = 0.002


def materialize(case: dict, db_path: Path) -> dict:
    """把参考 SQL 换成具体数字，转成 run_eval 已有的检查标准（result_has_number、report_contains）。"""
    conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        checks = dict(case.get("checks", {}))
        items = []
        for e in case.get("expect", []):
            v = float(conn.execute(e["sql"]).fetchone()[0])
            items.append({"label": e["label"], "values": [v, v * 100] if e.get("percent") else [v], "rel": REL_TOL})
        if items:
            checks["result_has_number"] = items
        names = [[str(conn.execute(q).fetchone()[0])] for q in case.get("report_sql", [])]
        if names:
            checks["report_contains"] = checks.get("report_contains", []) + names
    finally:
        conn.close()
    return {**case, "checks": checks}


# ---------------- schema linking 专项 ----------------
def eval_linking(db: Database, cases: list[dict]) -> dict:
    rows = []
    full = len(db.schema_text())
    for c in cases:
        got = {t for t, _ in db.link(c["question"], context=last_turn([tuple(h) for h in c.get("history", [])]))}
        need = set(c["tables"])
        rows.append({"id": c["id"], "question": c["question"], "need": sorted(need), "got": sorted(got),
                     "recall": len(got & need) / len(need), "precision": len(got & need) / max(len(got), 1),
                     "exact": got == need, "chars": len(db.schema_text(sorted(got))), "missed": sorted(need - got)})
    n = len(rows)
    return {"rows": rows, "summary": {
        "recall": sum(r["recall"] for r in rows) / n, "precision": sum(r["precision"] for r in rows) / n,
        "exact": sum(r["exact"] for r in rows), "n": n, "full_recall": sum(r["recall"] == 1 for r in rows),
        "saving": 1 - sum(r["chars"] for r in rows) / (full * n)}}


def linking_markdown(res: dict, title: str = "schema linking（不调用大模型）") -> list[str]:
    s = res["summary"]
    lines = [f"## {title}", "",
             f"- 召回率 **{s['recall']:.1%}**（{s['full_recall']}/{s['n']} 题需要的表全部选中），"
             f"精确率 {s['precision']:.1%}，完全正确 {s['exact']}/{s['n']}",
             f"- 提示词里的表结构平均比全部表少 **{s['saving']:.0%}**", "",
             "| 编号 | 问题 | 需要的表 | 选中的表 | 漏选 |", "|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['id']} | {r['question']} | {', '.join(r['need'])} | {', '.join(r['got'])} | "
                     f"{', '.join(r['missed']) or '—'} |")
    return lines


# ---------------- 端到端 ----------------
def evaluate(agent, mode: str, db: Database, db_path: Path, cases: list[dict], log=print) -> list[dict]:
    results = []
    for case in cases:
        log(f"[{case['id']}] {case['question']}")
        t0, before = time.time(), db.fingerprint()
        try:
            history = [tuple(h) for h in case.get("history", [])]
            if mode == "plan":
                run = agent.run(case["question"], history=history)
            else:
                run = agent.run(None, case["question"], history=history)
            r = grade(materialize(case, db_path), run)
            r["report"], r["sql"] = run.report, [s.code for s in run.steps]
            r["linked"] = [t["table"] for t in run.sql]
            if run.review is not None:
                r["review"] = run.review.to_dict()
            if case.get("db_unchanged"):
                same = Database(db_path).fingerprint() == before
                r["checks"].append({"check": "数据库未被修改", "ok": same, "detail": "未修改" if same else "被修改了！"})
                r["passed"] = r["passed"] and same
        except Exception as e:  # noqa: BLE001
            r = {"id": case["id"], "category": case["category"], "question": case["question"], "passed": False,
                 "checks": [{"check": "运行", "ok": False, "detail": f"出错：{e}"}],
                 "seconds": round(time.time() - t0, 1), "steps": 0, "skills": [], "unverified": [], "report": ""}
        log(f"      {'✅ 通过' if r['passed'] else '❌ 未通过'}（{r['seconds']} 秒）")
        results.append(r)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="只跑这些题号")
    ap.add_argument("--label", default="数据库评测", help="这次评测的名字")
    ap.add_argument("--mode", choices=["plan", "tools"], default="plan")
    ap.add_argument("--linking-only", action="store_true", help="只评 schema linking，不调用大模型")
    ap.add_argument("--no-knowledge", action="store_true", help="不接业务知识库")
    args = ap.parse_args()

    spec = json.loads(CASES.read_text(encoding="utf-8"))
    db_path = ROOT / spec["database"]
    if not db_path.exists():
        sys.exit("示例数据库不存在，请先运行 python make_sample_db.py")
    db = Database(db_path)
    cases = [c for c in spec["cases"] if not args.only or c["id"] in args.only]
    link = eval_linking(db, cases)
    holdout = eval_linking(db, spec.get("linking_holdout", [])) if not args.only else None
    link_md = linking_markdown(link, "schema linking：评测题（调规则时用过）")
    if holdout:
        link_md += [""] + linking_markdown(holdout, "schema linking：留出题（调完规则后新写，只跑一次）")
    for line in link_md[2:4] + (linking_markdown(holdout)[2:4] if holdout else []):
        print(line)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.linking_only:
        md = OUT_DIR / f"{stamp}_{args.label}_schema_linking.md"
        md.write_text("\n".join([f"# {args.label}：schema linking", ""] + link_md), encoding="utf-8")
        print(f"报告：{md}")
        return

    from analyst.config import get_llm_config, get_reviewer_config
    from analyst.knowledge import default_knowledge
    from analyst.llm import OpenAICompatLLM
    cfg = get_llm_config()
    if not cfg.ready:
        sys.exit("没有配置 API Key，请先在 .env 里填写")
    rcfg = get_reviewer_config(cfg)
    reviewer = OpenAICompatLLM(rcfg) if rcfg else None
    kb = None if args.no_knowledge else default_knowledge()
    if args.mode == "tools":
        from analyst.tool_agent import ToolCallingAgent
        agent = ToolCallingAgent(OpenAICompatLLM(cfg), reviewer=reviewer, knowledge=kb, db=db)
    else:
        from analyst.sql_agent import SQLAnalystAgent
        agent = SQLAnalystAgent(OpenAICompatLLM(cfg), db, reviewer=reviewer, knowledge=kb)

    results = evaluate(agent, args.mode, db, db_path, cases)
    text = to_markdown(results, args.label, f"{cfg.model}（数据库 {args.mode} 模式）")
    sql_parts = ["", "## 各题执行的 SQL", ""]
    for r in results:
        sql_parts.append(f"<details><summary>{r['id']}：{r['question']}</summary>\n")
        sql_parts += [f"```sql\n{q}\n```" for q in r.get("sql", [])]
        sql_parts.append("\n</details>\n")
    text += "\n\n" + "\n".join(link_md + sql_parts)
    md = OUT_DIR / f"{stamp}_{args.label}_{args.mode}.md"
    md.write_text(text, encoding="utf-8")
    (OUT_DIR / f"{stamp}_{args.label}_{args.mode}.json").write_text(
        json.dumps({"label": args.label, "model": cfg.model, "mode": args.mode, "linking": link["summary"],
                    "linking_holdout": holdout["summary"] if holdout else None,
                    "results": results}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n通过 {sum(r['passed'] for r in results)} / {len(results)}，报告：{md}")


if __name__ == "__main__":
    main()
