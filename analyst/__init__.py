"""电商经营分析 Agent：运营用中文提问，Agent 自动清洗数据、规划、写代码、执行、校验并生成报告。"""
from .agent import DataAnalystAgent, RunResult, StepRecord
from .config import get_llm_config
from .llm import FakeLLM, OpenAICompatLLM

__all__ = ["DataAnalystAgent", "RunResult", "StepRecord", "get_llm_config", "FakeLLM", "OpenAICompatLLM"]
