"""成本与可观测性：记录每一次模型调用花了多少 token、多少钱、多少时间，按环节汇总。

为什么需要它：Agent 一次分析要调用模型 5～15 次（规划、写代码、修代码、写报告、审查……），
不统计就不知道钱花在哪、哪个环节慢、改提示词之后是省了还是更贵了。

怎么接入（不用改每个调用点的参数）：
| 东西 | 作用 |
|---|---|
| UsageTracker | 一次任务的“账本”，记下每次调用的环节、模型、token、用时 |
| track(tracker) | with 块：块里所有模型调用都记进这本账（Agent.run 里自动开启） |
| stage("写代码") | with 块：标明接下来的调用属于哪个环节 |
| record(...) | 由 llm.py 在每次调用后自动调用，业务代码不用管 |

用 contextvars 实现：每个线程（后端每个提问一个线程）各记各的，互不串账。

费用 = 未命中缓存的输入 × 输入单价 + 命中缓存的输入 × 缓存单价 + 输出 × 输出单价。
单价在 .env 里配置（PRICE_INPUT / PRICE_INPUT_CACHED / PRICE_OUTPUT，单位：每百万 token），
没配置时用 DEFAULT_PRICES 里的参考价；都没有时只统计 token，不算钱。
DeepSeek 返回 prompt_cache_hit_tokens（命中缓存的输入），OpenAI 格式返回
prompt_tokens_details.cached_tokens，两种都认。
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

# 参考单价（美元 / 百万 token）：输入（未命中缓存）、输入（命中缓存）、输出。
# 价格会调整，以官网为准；在 .env 里设置 PRICE_* 会覆盖这里的值。
DEFAULT_PRICES = {
    "deepseek-chat": (0.28, 0.028, 0.42),
}
DEFAULT_CURRENCY = "USD"


@dataclass
class Price:
    input: float
    input_cached: float
    output: float
    currency: str = DEFAULT_CURRENCY

    def cost(self, miss: int, hit: int, out: int) -> float:
        return (miss * self.input + hit * self.input_cached + out * self.output) / 1e6


def price_for(model: str) -> Price | None:
    """.env 里的单价优先；否则按模型名查参考价；都没有返回 None（只统计 token）。"""
    env = [os.getenv(k, "").strip() for k in ("PRICE_INPUT", "PRICE_INPUT_CACHED", "PRICE_OUTPUT")]
    currency = os.getenv("PRICE_CURRENCY", DEFAULT_CURRENCY).strip() or DEFAULT_CURRENCY
    if env[0] and env[2]:
        try:
            return Price(float(env[0]), float(env[1] or env[0]), float(env[2]), currency)
        except ValueError:
            pass
    ref = DEFAULT_PRICES.get(model)
    return Price(*ref) if ref else None


@dataclass
class Call:
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    seconds: float = 0.0
    cost: float | None = None

    def to_dict(self) -> dict:
        return {"stage": self.stage, "model": self.model, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cached_tokens": self.cached_tokens,
                "seconds": round(self.seconds, 2),
                "cost": None if self.cost is None else round(self.cost, 6)}


@dataclass
class UsageTracker:
    calls: list[Call] = field(default_factory=list)

    def add(self, call: Call) -> None:
        self.calls.append(call)

    def summary(self) -> dict:
        """总量 + 按环节汇总。cost 为 None 表示有调用没有单价，总费用不完整。"""
        by_stage: dict[str, dict] = {}
        for c in self.calls:
            s = by_stage.setdefault(c.stage, {"stage": c.stage, "calls": 0, "prompt_tokens": 0,
                                              "completion_tokens": 0, "cached_tokens": 0,
                                              "seconds": 0.0, "cost": 0.0})
            s["calls"] += 1
            s["prompt_tokens"] += c.prompt_tokens
            s["completion_tokens"] += c.completion_tokens
            s["cached_tokens"] += c.cached_tokens
            s["seconds"] += c.seconds
            s["cost"] = None if s["cost"] is None or c.cost is None else s["cost"] + c.cost
        stages = list(by_stage.values())
        for s in stages:
            s["seconds"] = round(s["seconds"], 2)
            s["cost"] = None if s["cost"] is None else round(s["cost"], 6)
        prompt = sum(c.prompt_tokens for c in self.calls)
        cached = sum(c.cached_tokens for c in self.calls)
        costs = [c.cost for c in self.calls]
        return {
            "calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "cached_tokens": cached,
            "cache_hit_rate": round(cached / prompt, 4) if prompt else 0.0,
            "llm_seconds": round(sum(c.seconds for c in self.calls), 2),
            "cost": round(sum(costs), 6) if costs and None not in costs else None,
            "currency": _currency(self.calls),
            "by_stage": stages,
        }


def _currency(calls: list[Call]) -> str:
    for c in calls:
        p = price_for(c.model)
        if p:
            return p.currency
    return DEFAULT_CURRENCY


_TRACKER: ContextVar[UsageTracker | None] = ContextVar("usage_tracker", default=None)
_STAGE: ContextVar[str] = ContextVar("usage_stage", default="其他")


def current() -> UsageTracker | None:
    return _TRACKER.get()


@contextmanager
def track(tracker: UsageTracker):
    token = _TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _TRACKER.reset(token)


@contextmanager
def stage(name: str):
    token = _STAGE.set(name)
    try:
        yield
    finally:
        _STAGE.reset(token)


def record(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
           seconds: float = 0.0) -> None:
    """记一笔。没有开启 track 时什么也不做（例如单独调用模型做测试）。"""
    tracker = _TRACKER.get()
    if tracker is None:
        return
    p = price_for(model)
    cost = p.cost(prompt_tokens - cached_tokens, cached_tokens, completion_tokens) if p else None
    tracker.add(Call(_STAGE.get(), model, prompt_tokens, completion_tokens, cached_tokens, seconds, cost))


def record_response(model: str, usage, seconds: float) -> None:
    """从 OpenAI 兼容接口返回的 usage 里取数（字段缺失时按 0）。"""
    if usage is None:
        return
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    out = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = getattr(usage, "prompt_cache_hit_tokens", None)          # DeepSeek
    if cached is None:
        details = getattr(usage, "prompt_tokens_details", None)       # OpenAI 格式
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    record(model, prompt, out, int(cached or 0), seconds)
