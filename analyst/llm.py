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
