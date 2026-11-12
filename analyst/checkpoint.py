"""任务恢复：每完成一步就把进度存盘，中途失败（断网、模型接口报错、程序被关掉）后可以从断点继续。

比喻：玩游戏的“存档”。没有存档时，打到第 4 关掉线就得从第 1 关重来；
有了存档，重新进入直接从第 4 关开始。

存什么（存在本地 outputs/checkpoints/<任务编号>.pkl，已加入 .gitignore）：
| 模式 | 存档内容 | 什么时候存 |
|---|---|---|
| 规划-执行 | 计划 + 已完成步骤的结果（含清洗后的数据） | 规划完成后、每一步完成后 |
| 工具调用 | 完整对话记录 + 已完成步骤 + 工具箱状态（清洗后的数据、已加载的 Skill） | 每一轮工具调用之后 |

任务编号 = 模式 + 问题 + 数据指纹 + 对话历史 + 偏好 的哈希。同一个任务重新运行时自动找到存档；
数据或问题变了就是另一个任务，不会误用旧存档。任务成功完成后删除存档。
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

from .config import ROOT

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"


def data_fingerprint(df: pd.DataFrame) -> str:
    """数据指纹：行列数 + 每行内容的哈希。数据改了一个格子，指纹就不同。"""
    h = pd.util.hash_pandas_object(df, index=True).values
    return f"{df.shape}-{hashlib.sha1(h.tobytes()).hexdigest()[:16]}"


def task_id(mode: str, question: str, df: pd.DataFrame, history: object = None,
            preferences: list[str] | None = None) -> str:
    raw = "|".join([mode, question, data_fingerprint(df), repr(history), repr(preferences)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class CheckpointStore:
    def __init__(self, folder: Path = CHECKPOINT_DIR):
        self.folder = folder

    def path(self, tid: str) -> Path:
        return self.folder / f"{tid}.pkl"

    def save(self, tid: str, state: dict) -> None:
        """先写临时文件再改名：存到一半程序崩了，也不会留下一个坏掉的存档。"""
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.path(tid).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        tmp.replace(self.path(tid))

    def load(self, tid: str) -> dict | None:
        p = self.path(tid)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:  # noqa: BLE001  存档损坏就当没有，从头开始
            return None

    def clear(self, tid: str) -> None:
        self.path(tid).unlink(missing_ok=True)

    def exists(self, tid: str) -> bool:
        return self.path(tid).exists()
