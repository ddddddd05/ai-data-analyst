"""大模型接口（Agent 的“大脑”）。

用 OpenAI 兼容格式调用，Gemini / DeepSeek / 通义千问等都能接。
测试时用 FakeLLM 代替，不需要联网和密钥。
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .config import LLMConfig


class LLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class ToolLLM(Protocol):
    """支持工具调用（Function Calling）的模型。

    chat_tools 返回 {"content": 文字, "tool_calls": [{"id", "name", "arguments"(dict)}],
    "raw": 原样放回对话历史的 assistant 消息}。
    """
    def chat_tools(self, messages: list[dict], tools: list[dict]) -> dict: ...


class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig, temperature: float = 0.2):
        from openai import DefaultHttpxClient, OpenAI  # 只在真正调用时才需要安装

        # use_proxy=False 时忽略系统/环境变量里的代理设置，直接连接（适合国内模型）
        http_client = None if cfg.use_proxy else DefaultHttpxClient(trust_env=False)
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=120,
                             max_retries=1, http_client=http_client)
        self.model = cfg.model
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    def chat_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """用 OpenAI 兼容接口的 tools 参数，让模型自己决定调用哪个工具。"""
        kwargs = {"tools": tools} if tools else {}
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, messages=messages, **kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_bad_json": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": msg.content or "", "tool_calls": calls,
                "raw": msg.model_dump(exclude_none=True)}


class FakeLLM:
    """按顺序返回预先写好的回答，用于离线测试 Agent 的流程。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise RuntimeError("FakeLLM 的预设回答用完了")
        return self.replies.pop(0)

    def chat_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """预设回答可以是字符串（模型直接回复文字）或
        {"tool_calls": [("工具名", {参数}), ...]}（模型调用工具）。"""
        self.calls.append(("tools", messages[-1].get("content", "")))
        if not self.replies:
            raise RuntimeError("FakeLLM 的预设回答用完了")
        r = self.replies.pop(0)
        if isinstance(r, str):
            return {"content": r, "tool_calls": [], "raw": {"role": "assistant", "content": r}}
        calls = [{"id": f"call_{len(self.calls)}_{i}", "name": n, "arguments": a}
                 for i, (n, a) in enumerate(r.get("tool_calls", []))]
        raw = {"role": "assistant", "content": r.get("content", ""),
               "tool_calls": [{"id": c["id"], "type": "function",
                               "function": {"name": c["name"],
                                            "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                              for c in calls]}
        return {"content": r.get("content", ""), "tool_calls": calls, "raw": raw}


def extract_json(text: str):
    """从模型回答里取出 JSON（兼容 ```json 代码块和前后多余文字）。"""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = m.group(1) if m else text
    start = min([i for i in (candidate.find("{"), candidate.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("回答里没有 JSON")
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    return json.loads(candidate[start:end + 1])


def extract_code(text: str) -> str:
    """从模型回答里取出 Python 代码。"""
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()
