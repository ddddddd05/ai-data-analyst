"""独立审查 Agent：报告写完后，另起一次模型调用，专门挑“结论和计算结果对不上”的毛病。

为什么需要它（和 validator 的分工）：
| | validator.py 数字核对 | reviewer.py 审查 Agent |
|---|---|---|
| 检查什么 | 报告里的每个数字，在计算结果里有没有出处 | 结论、排名、判断和计算结果是否矛盾 |
| 怎么检查 | 程序逐个比对，100% 确定 | 模型阅读理解，能发现“数字都对、结论却说反了” |
| 例子 | 报告写“增长 30.5%”，结果里没有 30.5 → 标出 | 表格里渔轮退货率最低，报告却说“钓竿退货率最低” → 标出 |

“独立”体现在：
1. 审查员看不到写报告时的对话，只拿到“问题 + 各步骤结果 + 报告”，角色是挑错而不是写作；
2. 可以用另一个模型担任审查员（.env 里设置 REVIEWER_MODEL），避免“自己检查自己”。

发现“严重”问题时，把问题清单交回写报告的模型修改一次，改完重新核对数字。只修改一次，防止来回拉扯。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .llm import LLM, extract_json


@dataclass
class ReviewResult:
    passed: bool = True                              # 没有严重问题
    issues: list[dict] = field(default_factory=list)  # [{"severity": "严重/提醒", "claim", "evidence"}]
    revised: bool = False                            # 是否因为严重问题改过报告
    error: str = ""                                  # 审查本身失败的原因（失败时不影响报告）

    @property
    def serious(self) -> list[dict]:
        return [i for i in self.issues if i.get("severity") == "严重"]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "issues": self.issues, "revised": self.revised,
                "error": self.error}


def review_report(llm: LLM, question: str, results_text: str, report: str) -> ReviewResult:
    """让审查员检查报告；解析失败时视为“审查未完成”，不阻断流程。"""
    reply = llm.chat(prompts.REVIEWER_SYSTEM, prompts.REVIEWER_USER.format(
        question=question, results=results_text, report=report))
    try:
        data = extract_json(reply)
        issues = data.get("issues", []) if isinstance(data, dict) else data
        issues = [{"severity": "严重" if str(i.get("severity", "")).strip() == "严重" else "提醒",
                   "claim": str(i.get("claim", "")), "evidence": str(i.get("evidence", ""))}
                  for i in issues if isinstance(i, dict) and i.get("claim")]
    except (ValueError, TypeError, AttributeError) as e:
        return ReviewResult(passed=True, error=f"审查结果无法解析：{e}")
    return ReviewResult(passed=not any(i["severity"] == "严重" for i in issues), issues=issues)


def revise_report(llm: LLM, question: str, results_text: str, report: str,
                  review: ReviewResult) -> str:
    """把严重问题交回写报告的模型，修改一次。"""
    problems = "\n".join(f"- 报告说：{i['claim']}\n  但结果显示：{i['evidence']}" for i in review.serious)
    reply = llm.chat(prompts.REPORTER_SYSTEM, prompts.REVISE_USER.format(
        question=question, results=results_text, report=report, problems=problems))
    return reply.strip() or report
