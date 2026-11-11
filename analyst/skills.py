"""Skill：按需加载的专项规则。

每个 Skill 是 skills/ 下的一个文件夹，里面有一个 SKILL.md：
- 开头的 --- 之间是“简介”（名称、描述、什么时候用）；
- 下面是正文（具体规则）。

程序根据数据概况和用户问题判断要不要加载某个 Skill，
只有用到时才把正文放进提示词，平时不占篇幅。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT

SKILLS_DIR = ROOT / "skills"


@dataclass
class Skill:
    name: str
    description: str
    body: str
    when: list[str] = field(default_factory=list)       # 数据概况满足这些条件时加载
    keywords: list[str] = field(default_factory=list)   # 问题里出现这些词时加载


def _split_list(value: str) -> list[str]:
    return [x.strip() for x in value.replace("，", ",").split(",") if x.strip()]


def parse_skill(text: str) -> Skill:
    """解析 SKILL.md：--- 之间是简介（key: value），之后是正文。"""
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return Skill(
        name=meta.get("name", "未命名"),
        description=meta.get("description", ""),
        body=body.strip(),
        when=_split_list(meta.get("when", "")),
        keywords=_split_list(meta.get("keywords", "")),
    )


def load_skills(folder: Path = SKILLS_DIR) -> list[Skill]:
    if not folder.exists():
        return []
    return [parse_skill(p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob("*/SKILL.md"))]


def profile_flags(profile: dict) -> dict[str, bool]:
    """把数据概况翻译成几个“条件”，供 Skill 判断要不要加载。"""
    cols = profile.get("columns", [])
    return {
        "重复行": profile.get("duplicate_rows", 0) > 0,
        "缺失值": any(c.get("missing", 0) > 0 for c in cols),
        "负数": any(c.get("negative", 0) > 0 for c in cols),
        "极端值": any(c.get("outliers", 0) > 0 for c in cols),
    }


def select_skills(skills: list[Skill], question: str, profile: dict) -> list[tuple[Skill, str]]:
    """返回 [(Skill, 加载原因)]。"""
    flags = profile_flags(profile)
    chosen = []
    for s in skills:
        reasons = [f"数据有{c}" for c in s.when if flags.get(c)]
        reasons += [f"问题提到“{k}”" for k in s.keywords if k in question]
        if reasons:
            chosen.append((s, "、".join(reasons)))
    return chosen


def render_skills(chosen: list[tuple[Skill, str]]) -> str:
    """拼成放进提示词的文字。"""
    if not chosen:
        return "（无）"
    return "\n\n".join(f"### Skill：{s.name}\n{s.body}" for s, _ in chosen)
