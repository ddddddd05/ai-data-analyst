"""记忆：短期记忆（长对话自动压缩）+ 长期记忆（用户偏好）。

| | 短期记忆 ConversationMemory | 长期记忆 PreferenceStore |
|---|---|---|
| 记什么 | 这次对话里问过什么、得出什么结论 | 用户的固定习惯，如“退货率按金额算” |
| 存多久 | 只在本次对话里，清空对话就没了 | 存进本地 JSON 文件，下次打开还在 |
| 怎么用 | 追问时理解“那”“它”指什么 | 每次分析都放进提示词，必须遵守 |
| 怎么控制长度 | 最近 3 轮保留原文，更早的压缩成一段摘要 | 同一主题只保留最新一条 |

为什么要压缩：每轮都把全部历史发给模型，对话越长提示词越长 → 更慢、更贵，
还可能超出模型的上下文长度；只保留最近几轮又会把早期的关键结论直接丢掉。
做法和人记笔记一样：最近的事记原话，早的事只记要点。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import prompts
from .config import ROOT
from .llm import LLM, extract_json

PREF_FILE = ROOT / "memory" / "preferences.json"   # 本地文件，已加入 .gitignore

# 出现这些词才去判断是不是“偏好”，避免每个问题都多调一次模型
PREF_CUES = ("记住", "以后", "今后", "默认", "一律", "统一按", "都按", "都用", "偏好", "习惯", "不要再", "别再")


# ---------------- 短期记忆：长对话压缩 ----------------
@dataclass
class ConversationMemory:
    keep_recent: int = 3                  # 最近几轮保留原文
    summary: str = ""                     # 更早的对话压缩成的摘要
    turns: list[tuple[str, str]] = field(default_factory=list)   # 最近几轮 [(问题, 结论)]
    compressed_turns: int = 0             # 已经压缩进摘要的轮数

    def add_turn(self, question: str, conclusion: str, llm: LLM | None = None) -> bool:
        """记一轮对话；超过 keep_recent 轮时，把最早的一轮压缩进摘要。返回是否发生了压缩。"""
        self.turns.append((question, conclusion))
        if len(self.turns) <= self.keep_recent:
            return False
        old = self.turns.pop(0)
        self.summary = compress(llm, self.summary, [old]) if llm else _naive_merge(self.summary, [old])
        self.compressed_turns += 1
        return True

    def render(self) -> str:
        """变成放进提示词的文字。"""
        if not self.summary and not self.turns:
            return "（这是第一个问题）"
        parts = []
        if self.summary:
            parts.append(f"更早对话的摘要（共 {self.compressed_turns} 轮，已压缩）：\n{self.summary}")
        start = self.compressed_turns + 1
        for i, (q, c) in enumerate(self.turns, start):
            parts.append(f"第{i}轮问题：{q}\n第{i}轮结论：{c}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {"keep_recent": self.keep_recent, "summary": self.summary, "turns": self.turns,
                "compressed_turns": self.compressed_turns}


def compress(llm: LLM, summary: str, turns: list[tuple[str, str]], limit: int = 600) -> str:
    """把旧摘要 + 要移出的几轮对话，压缩成一段新摘要（滚动压缩）。模型出错时退回简单拼接。"""
    text = "\n".join(f"问题：{q}\n结论：{c}" for q, c in turns)
    try:
        reply = llm.chat(prompts.COMPRESS_SYSTEM, prompts.COMPRESS_USER.format(
            summary=summary or "（无）", turns=text, limit=limit))
        return reply.strip()[: limit + 200] or _naive_merge(summary, turns)
    except Exception:  # noqa: BLE001
        return _naive_merge(summary, turns)


def _naive_merge(summary: str, turns: list[tuple[str, str]], limit: int = 800) -> str:
    """不调用模型的兜底：每轮只留问题和结论的前 80 个字。"""
    lines = [summary] if summary else []
    lines += [f"- 问：{q[:60]} → {c[:80]}" for q, c in turns]
    return "\n".join(lines)[-limit:]


# ---------------- 长期记忆：用户偏好 ----------------
@dataclass
class Preference:
    topic: str        # 主题，如“退货率口径”；同一主题只保留最新一条
    content: str      # 具体要求，如“退货率按金额计算：退货订单销售额 ÷ 总销售额”
    created: str = ""


class PreferenceStore:
    """用户偏好的增删查，保存在本地 JSON 文件；path=None 时只存在内存里（测试、评测用）。"""

    def __init__(self, path: Path | None = PREF_FILE):
        self.path = path
        self.items: list[Preference] = []
        if path and path.exists():
            try:
                self.items = [Preference(**x) for x in json.loads(path.read_text(encoding="utf-8"))]
            except (ValueError, TypeError):
                self.items = []

    def add(self, topic: str, content: str) -> None:
        self.items = [p for p in self.items if p.topic != topic]    # 新的覆盖旧的
        self.items.append(Preference(topic, content, datetime.now().strftime("%Y-%m-%d %H:%M")))
        self.save()

    def remove(self, topic: str) -> None:
        self.items = [p for p in self.items if p.topic != topic]
        self.save()

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([p.__dict__ for p in self.items], ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    def contents(self) -> list[str]:
        return [f"{p.topic}：{p.content}" for p in self.items]


def render_preferences(prefs: list[str] | None) -> str:
    """把偏好拼进提示词。偏好优先于 Skill 里的默认口径。"""
    if not prefs:
        return ""
    return ("\n\n### 用户偏好（长期记忆，必须遵守；与 Skill 默认口径冲突时以这里为准，"
            "并在报告里说明用的是哪种口径）\n" + "\n".join(f"- {p}" for p in prefs))


def extract_preferences(llm: LLM, message: str) -> tuple[list[dict], bool]:
    """从用户这句话里找“以后都要这样”的偏好。

    返回 ([{"topic", "content"}], 这句话里是否还有需要分析的问题)。
    先用关键词过滤：没有“记住、以后、默认……”等词就直接跳过，不额外调用模型。
    """
    if not any(c in message for c in PREF_CUES):
        return [], True
    try:
        data = extract_json(llm.chat(prompts.PREF_SYSTEM, prompts.PREF_USER.format(message=message)))
        prefs = [{"topic": str(p["topic"]).strip(), "content": str(p["content"]).strip()}
                 for p in data.get("preferences", []) if p.get("topic") and p.get("content")]
        return prefs, bool(data.get("has_question", True))
    except (ValueError, KeyError, TypeError, AttributeError):
        return [], True
