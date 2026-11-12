"""任务恢复专项评测（故障注入）：故意让模型接口在中途“断网”，再从断点继续，看能省下多少重复工作。

做法（每道题）：
1. 完整跑一遍，记录模型调用次数和用时（对照组）；
2. 再跑一遍，但在第 N 次模型调用时故意抛出“连接断开”的错误 → 任务失败，进度留在存档里；
3. 用正常的模型重新提交同一个问题 → 从断点继续，记录这次补跑的调用次数和用时；
4. 检查补跑后报告是否完整、步骤是否全部成功。

用法：python evals/run_resume_eval.py            （默认规划-执行模式，C01、C05、C12 三道题）
      python evals/run_resume_eval.py --mode tools
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.agent import DataAnalystAgent  # noqa: E402
from analyst.checkpoint import CheckpointStore  # noqa: E402
from analyst.config import get_llm_config  # noqa: E402
from analyst.llm import OpenAICompatLLM  # noqa: E402
from analyst.profiler import load_table  # noqa: E402
from analyst.tool_agent import ToolCallingAgent  # noqa: E402

OUT_DIR = ROOT / "docs" / "评测报告"


class CountingLLM:
    """包一层：数一数调用了几次模型；fail_at=N 时，第 N 次调用直接报“连接断开”。"""

    def __init__(self, inner, fail_at: int | None = None):
        self.inner, self.fail_at, self.n = inner, fail_at, 0

    def _tick(self):
        self.n += 1
        if self.fail_at and self.n == self.fail_at:
            raise ConnectionError(f"（故障注入）第 {self.n} 次调用模型时连接断开")

    def chat(self, system, user):
        self._tick()
        return self.inner.chat(system, user)

    def chat_tools(self, messages, tools):
        self._tick()
        return self.inner.chat_tools(messages, tools)


def make_agent(mode, llm, store):
    if mode == "tools":
        return ToolCallingAgent(llm, checkpoints=store)
    return DataAnalystAgent(llm, checkpoints=store)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "tools"], default="plan")
    ap.add_argument("--only", nargs="*", default=["C01", "C05", "C12"])
    ap.add_argument("--fail-at", type=int, default=4, help="第几次模型调用时断网（默认第 4 次）")
    args = ap.parse_args()

    cfg = get_llm_config()
    if not cfg.ready:
        sys.exit("没有配置 API Key，请先在 .env 里填写")
    real = OpenAICompatLLM(cfg)
    spec = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
    df = load_table(ROOT / spec["data"])
    cases = [c for c in spec["cases"] if c["id"] in args.only]
    rows = []
    for c in cases:
        q = c["question"]
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            # ① 对照组：完整跑一遍（不存档，避免影响后面）
            full = CountingLLM(real)
            t0 = time.time()
            make_agent(args.mode, full, None).run(df, q)
            full_sec = time.time() - t0
            # ② 故障注入：第 N 次调用断网
            flaky = CountingLLM(real, fail_at=args.fail_at)
            t0 = time.time()
            failed = False
            try:
                make_agent(args.mode, flaky, store).run(df, q)
            except ConnectionError:
                failed = True
            fail_sec = time.time() - t0
            # ③ 从断点继续
            again = CountingLLM(real)
            t0 = time.time()
            run = make_agent(args.mode, again, store).run(df, q)
            resume_sec = time.time() - t0
        row = {"id": c["id"], "question": q, "failed": failed,
               "full_calls": full.n, "full_sec": round(full_sec, 1),
               "calls_before_fail": flaky.n, "fail_sec": round(fail_sec, 1),
               "resume_calls": again.n, "resume_sec": round(resume_sec, 1),
               "resumed_steps": run.resumed_steps, "all_steps_ok": all(s.ok for s in run.steps),
               "has_report": "## 结论" in run.report}
        rows.append(row)
        print(f"[{c['id']}] 完整 {full.n} 次调用 {row['full_sec']} 秒；断网前完成 {flaky.n - 1} 次；"
              f"续跑 {again.n} 次调用 {row['resume_sec']} 秒，恢复 {run.resumed_steps} 步，"
              f"步骤全部成功={row['all_steps_ok']}，报告完整={row['has_report']}")

    lines = [f"# 任务恢复专项评测（故障注入，{args.mode} 模式）", "",
             f"- 时间：{datetime.now():%Y-%m-%d %H:%M}", f"- 模型：{cfg.model}",
             f"- 故障：第 {args.fail_at} 次模型调用时抛出“连接断开”", "",
             "| 题号 | 完整运行（调用次数 / 秒） | 断网前已完成调用 | 续跑（调用次数 / 秒） | 恢复的步骤 | 步骤全部成功 | 报告完整 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['full_calls']} / {r['full_sec']} | {r['calls_before_fail'] - 1} | "
                     f"{r['resume_calls']} / {r['resume_sec']} | {r['resumed_steps']} | "
                     f"{'✅' if r['all_steps_ok'] else '❌'} | {'✅' if r['has_report'] else '❌'} |")
    lines += ["", "说明：同一个问题每次运行的步数不完全相同（模型有随机性），所以“完整运行”只是参照，"
              "不能把两者直接相减当成精确节省量。"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M}_任务恢复专项评测_{args.mode}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告：{out}")


if __name__ == "__main__":
    main()
