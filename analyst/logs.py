"""运行日志：每次提问记一条“开始”、一条“结束”（或“失败”），写进 outputs/logs/app.log。

为什么要有：网页上只看得到当前这次的结果；线上出了问题（某天突然变慢、费用变高、
某类问题总失败），要靠日志回头查。每行一条 JSON，方便用程序统计，比如
“今天平均用时多少”“哪种模式失败最多”。

文件按大小轮转（每个 5MB，最多留 5 个），不会无限变大；outputs/ 已在 .gitignore 里，不会上传。
"""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler

from .config import ROOT

LOG_DIR = ROOT / "outputs" / "logs"
logger = logging.getLogger("analyst")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"), "level": record.levelname,
                "event": record.getMessage(), **getattr(record, "fields", {})}
        if record.exc_info:
            data["traceback"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def setup_logging(folder=LOG_DIR) -> logging.Logger:
    """重复调用不会重复加 handler（测试里会多次创建 app）。"""
    if not any(getattr(h, "_analyst", False) for h in logger.handlers):
        folder.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(folder / "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        fh._analyst = True
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger


def log_event(event: str, level: int = logging.INFO, exc_info: bool = False, **fields) -> None:
    logger.log(level, event, extra={"fields": fields}, exc_info=exc_info)


def run_summary(run) -> dict:
    """从一次分析结果里挑出值得记录的数字（不记报告全文和数据内容）。"""
    u = run.usage or {}
    return {
        "mode": run.mode, "seconds": run.seconds,
        "steps_ok": sum(1 for s in run.steps if s.ok), "steps": len(run.steps),
        "retries": sum(max(0, s.attempts - 1) for s in run.steps),
        "tool_calls": len(run.tool_calls), "resumed_steps": run.resumed_steps,
        "unverified": len(run.unverified),
        "review_passed": None if run.review is None else run.review.passed,
        "review_revised": None if run.review is None else run.review.revised,
        "llm_calls": u.get("calls"), "cost": u.get("cost"), "currency": u.get("currency"),
        "cache_hit_rate": u.get("cache_hit_rate"),
    }
