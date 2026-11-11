"""AI 数据分析助手：用自然语言提问，Agent 自动规划、写代码、执行、校验并生成报告。"""
from .agent import DataAnalystAgent, RunResult, StepRecord
from .config import get_llm_config
from .llm import FakeLLM, OpenAICompatLLM

__all__ = ["DataAnalystAgent", "RunResult", "StepRecord", "get_llm_config", "FakeLLM", "OpenAICompatLLM"]
