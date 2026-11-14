"""业务知识库检索专项评测：比较关键词、向量、混合三种检索方式。

用法：python evals/run_kb_eval.py [--k 3]
题目在 evals/kb_cases.json。不调用大模型，只用本机的向量模型，几十秒跑完。

指标：
- 命中率（Recall@k）：应该找到的小节，有没有排进前 k 名（有一个就算命中）；
- MRR：第一个正确结果排第几名的倒数，排第 1 得 1 分、第 2 得 0.5 分……越大说明越靠前；
- 无关问题：知识库里没有相关内容时，应该返回空，而不是硬塞几段无关资料。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.knowledge import KnowledgeBase  # noqa: E402

CASES = ROOT / "evals" / "kb_cases.json"
OUT_DIR = ROOT / "docs" / "评测报告"


def evaluate(kb: KnowledgeBase, cases: list[dict], k: int = 3) -> dict:
    """返回 {mode: {"rows": [...], "summary": {...}}}。"""
    out = {}
    for mode in ("keyword", "vector", "hybrid"):
        rows = []
        for c in cases:
            titles = [h.chunk.title for h in kb.search(c["query"], k=k, mode=mode)]
            if c["relevant"]:
                ranks = [i for i, t in enumerate(titles, 1) if t in c["relevant"]]
                rows.append({"id": c["id"], "kind": c["kind"], "hit": bool(ranks),
                             "rr": 1 / ranks[0] if ranks else 0.0, "got": titles})
            else:
                rows.append({"id": c["id"], "kind": c["kind"], "hit": not titles, "rr": None, "got": titles})
        by_kind = {}
        for kind in dict.fromkeys(r["kind"] for r in rows):
            rs = [r for r in rows if r["kind"] == kind]
            by_kind[kind] = f"{sum(r['hit'] for r in rs)}/{len(rs)}"
        scored = [r for r in rows if r["rr"] is not None]
        out[mode] = {"rows": rows, "summary": {
            "recall": f"{sum(r['hit'] for r in scored)}/{len(scored)}",
            "mrr": round(sum(r["rr"] for r in scored) / len(scored), 3),
            "by_kind": by_kind}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--label", default="知识库检索")
    args = ap.parse_args()
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    kb = KnowledgeBase()
    res = evaluate(kb, cases, args.k)
    kinds = list(res["hybrid"]["summary"]["by_kind"])
    names = {"keyword": "关键词（BM25）", "vector": "向量（bge-small-zh）", "hybrid": "混合（RRF）"}
    lines = [f"# 评测报告：{args.label}（{datetime.now():%Y-%m-%d %H:%M}）", "",
             f"- 知识库：{len(kb.chunks)} 段；{kb.status}；取前 {args.k} 名；向量相似度阈值 {kb.min_similarity}", "",
             "| 检索方式 | 命中率 | MRR | " + " | ".join(kinds) + " |",
             "|---|---|---|" + "---|" * len(kinds)]
    for mode, r in res.items():
        s = r["summary"]
        lines.append(f"| {names[mode]} | {s['recall']} | {s['mrr']} | "
                     + " | ".join(s["by_kind"][k] for k in kinds) + " |")
    lines += ["", "## 每题结果（混合检索）", "", "| 题号 | 类型 | 命中 | 检索到的前几段 |", "|---|---|---|---|"]
    for row in res["hybrid"]["rows"]:
        lines.append(f"| {row['id']} | {row['kind']} | {'✅' if row['hit'] else '❌'} | "
                     + ("；".join(row["got"]) or "（空）") + " |")
    lines += ["", "## 三种方式结果不同的题", ""]
    for i, c in enumerate(cases):
        hits = {m: res[m]["rows"][i]["hit"] for m in res}
        if len(set(hits.values())) > 1:
            lines.append(f"- {c['id']}（{c['kind']}）“{c['query']}”："
                         + "，".join(f"{names[m]}{'✅' if h else '❌'}" for m, h in hits.items()))
    text = "\n".join(lines)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M}_{args.label}.md"
    path.write_text(text + "\n", encoding="utf-8")
    (path.with_suffix(".json")).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
