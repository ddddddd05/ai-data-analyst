"""业务知识库（RAG）：从 knowledge/ 下的业务文档里检索和问题相关的片段，交给模型参考。

为什么需要它（和计算、Skill 的分工）：
| | 计算（run_code） | Skill | 业务知识库（本文件） |
|---|---|---|---|
| 解决什么 | 数字类问题：答案要从表格里算出来 | 固定的做事规范（怎么清洗、指标怎么算） | 模型不知道的业务事实：字段含义、公司口径、平台规则、往期结论 |
| 怎么选 | — | 关键词 + 数据体检结果，整篇加载 | 按问题检索最相关的几段 |
| 规模 | — | 2 个，几十行 | 可以扩展到很多文档 |

表格数据不适合直接做 RAG：检索只能拿到几段内容，算不出“全部钓竿订单的退货率”这种整体指标，
所以数字仍由代码计算；知识库只补充“这一列什么意思”“公司规定怎么算”“平台红线是多少”。

混合检索（hybrid）：
| | 关键词检索（BM25） | 向量检索（bge-small-zh） |
|---|---|---|
| 怎么找 | 字面上有相同的词 | 意思相近（把文字变成向量，算余弦相似度） |
| 擅长 | 专有名词：ACOS、ROAS、Prime Day | 换了说法：“售后问题多” ≈ “退货率高” |
| 弱点 | 换个说法就找不到 | 专有名词、数字有时不如字面匹配准 |
两路各取排名，用 RRF（倒数排名融合）合并：score = Σ 1 / (60 + 排名)，不需要把两种分数换算到同一尺度。

切分：按 Markdown 的“## 小节”切，每段是一个完整的话题（一列的含义、一条规则），
检索结果能直接告诉模型“来源：数据字典 · 广告花费”，方便报告注明出处。

向量模型在本机运行（fastembed + onnxruntime，模型约 100MB，第一次使用时下载到 outputs/models/）。
模型加载失败（没网、没装 fastembed）时自动退回只用关键词检索，并在 status 里说明原因。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ROOT

KB_DIR = ROOT / "knowledge"
MODEL_DIR = ROOT / "outputs" / "models"
CACHE_DIR = ROOT / "outputs" / "kb_cache"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# bge 系列建议：检索时给“问题”加上这句指令，文档本身不加
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
RRF_K = 60
MODES = ("hybrid", "vector", "keyword")


@dataclass
class Chunk:
    id: int
    doc: str        # 文档名，如“数据字典”
    section: str    # 小节标题，如“广告花费”
    text: str

    @property
    def title(self) -> str:
        return f"{self.doc} · {self.section}"


@dataclass
class Hit:
    chunk: Chunk
    score: float                  # 融合后的分数（越大越相关）
    keyword_rank: int | None = None
    vector_rank: int | None = None
    similarity: float | None = None

    def to_dict(self) -> dict:
        return {"title": self.chunk.title, "score": round(self.score, 4),
                "keyword_rank": self.keyword_rank, "vector_rank": self.vector_rank,
                "similarity": None if self.similarity is None else round(self.similarity, 3)}


# ---------------- 读取和切分 ----------------
def load_chunks(folder: Path = KB_DIR) -> list[Chunk]:
    """每个 .md 文件：“# 标题”是文档名，每个“## 小节”切成一段；“>”开头的说明行不进入正文。"""
    chunks: list[Chunk] = []
    if not folder.exists():
        return chunks
    for path in sorted(folder.glob("*.md")):
        doc, section, lines = path.stem, None, []

        def flush():
            body = "\n".join(lines).strip()
            if section and body:
                chunks.append(Chunk(len(chunks), doc, section, body))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                section, lines = line[3:].strip(), []
            elif line.startswith("# ") or line.startswith(">"):
                continue
            else:
                lines.append(line)
        flush()
    return chunks


# ---------------- 关键词检索：BM25 ----------------
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """不依赖分词库：英文/数字按整词，中文按“相邻两个字”切（bigram），单字的词也保留。"""
    tokens: list[str] = []
    for w in _WORD.findall(text.lower()):
        if w[0].isascii():
            tokens.append(w)
        elif len(w) == 1:
            tokens.append(w)
        else:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]
    return tokens


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in set(query):
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


# ---------------- 向量检索 ----------------
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """本机运行的中文向量模型；第一次调用时才加载（加载约 30 秒，之后很快）。"""

    def __init__(self, model: str = EMBED_MODEL, cache_dir: Path = MODEL_DIR):
        self.name, self.cache_dir, self._model = model, cache_dir, None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.name, cache_dir=str(self.cache_dir))
        return np.array(list(self._model.embed(texts)), dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)


# ---------------- 知识库 ----------------
class KnowledgeBase:
    def __init__(self, folder: Path = KB_DIR, embedder: Embedder | None | str = "auto",
                 mode: str = "hybrid", min_similarity: float = 0.45, kw_min_ratio: float = 0.3,
                 cache: bool = True):
        """embedder="auto"：用本机 bge-small-zh；None：只用关键词检索；也可以传入自定义对象（测试用）。
        min_similarity：向量相似度低于它的片段，这一路不算命中（无关问题的相似度一般只有 0.35～0.4）。
        kw_min_ratio：关键词得分低于本次最高分这个比例的片段，这一路不算命中。中文按两个字切词，
        “退货”这类常见词会让很多片段沾上一点分，不过滤的话，融合时这些片段两路都有分，
        会把只在关键词里排第 1 的正确答案挤下去（检索评测 K17“GMV 包不包含退货”）。"""
        if mode not in MODES:
            raise ValueError(f"mode 只能是 {MODES}")
        self.chunks = load_chunks(folder)
        self.mode, self.min_similarity, self.cache = mode, min_similarity, cache
        self.kw_min_ratio = kw_min_ratio
        self.embedder = FastEmbedder() if embedder == "auto" else embedder
        self.bm25 = BM25([tokenize(c.title + " " + c.text) for c in self.chunks])
        self._vectors: np.ndarray | None = None
        self.status = "未加载向量模型" if self.embedder else "只用关键词检索（没有配置向量模型）"
        if self.embedder is None and mode != "keyword":
            self.mode = "keyword"

    # 向量第一次检索时才计算，并缓存到 outputs/kb_cache（文档不变就不用重算）
    def _ensure_vectors(self) -> bool:
        if self._vectors is not None:
            return True
        if self.embedder is None or not self.chunks:
            return False
        texts = [c.title + "\n" + c.text for c in self.chunks]
        key = hashlib.sha1((self.embedder.name + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = CACHE_DIR / f"{key}.npy"
        try:
            if self.cache and path.exists():
                self._vectors = np.load(path)
            else:
                self._vectors = _normalize(self.embedder.embed(texts))
                if self.cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, self._vectors)
            self.status = f"混合检索可用（向量模型 {self.embedder.name}，{len(self.chunks)} 段）"
            return True
        except Exception as e:  # noqa: BLE001
            self.embedder, self.mode = None, "keyword"
            self.status = f"向量模型不可用，已退回关键词检索：{type(e).__name__}: {str(e)[:120]}"
            return False

    def search(self, query: str, k: int = 4, mode: str | None = None) -> list[Hit]:
        mode = mode or self.mode
        if not self.chunks or not query.strip():
            return []
        kw = self.bm25.scores(tokenize(query))
        floor = max(kw, default=0) * self.kw_min_ratio
        kw_order = [i for i in sorted(range(len(kw)), key=lambda i: -kw[i]) if kw[i] > 0 and kw[i] >= floor]
        kw_rank = {i: r for r, i in enumerate(kw_order, 1)}

        sims, vec_rank = None, {}
        if mode != "keyword" and self._ensure_vectors():
            q = _normalize(self.embedder.embed([QUERY_PREFIX + query]))[0]
            sims = self._vectors @ q
            vec_order = [i for i in np.argsort(-sims) if sims[i] >= self.min_similarity]
            vec_rank = {int(i): r for r, i in enumerate(vec_order, 1)}
        elif mode == "vector":
            mode = "keyword"          # 向量不可用时退回关键词，不返回空结果

        use_kw = mode in ("hybrid", "keyword")
        use_vec = mode in ("hybrid", "vector") and sims is not None
        scores: dict[int, float] = {}
        if use_kw:
            for i, r in kw_rank.items():
                scores[i] = scores.get(i, 0) + 1 / (RRF_K + r)
        if use_vec:
            for i, r in vec_rank.items():
                scores[i] = scores.get(i, 0) + 1 / (RRF_K + r)
        top = sorted(scores, key=lambda i: -scores[i])[:k]
        return [Hit(self.chunks[i], scores[i], kw_rank.get(i) if use_kw else None,
                    vec_rank.get(i) if use_vec else None,
                    float(sims[i]) if sims is not None else None) for i in top]


@lru_cache(maxsize=1)
def default_knowledge() -> KnowledgeBase:
    """网页、MCP 服务器、评测共用一个知识库（向量模型只加载一次）。"""
    return KnowledgeBase()


# ---------------- 交给模型 / 核对数字 ----------------
def render_knowledge(hits: list[Hit]) -> str:
    if not hits:
        return ""
    parts = [f"[{n}] 来源：{h.chunk.title}\n{h.chunk.text}" for n, h in enumerate(hits, 1)]
    return ("\n\n### 业务知识库检索结果（模拟的公司资料，按需参考）\n"
            "用于理解字段含义、公司口径、平台规则和往期结论；引用平台规则、行业参考值或往期数字时注明来源；"
            "往期数字只能用来对比，不能当成本期的计算结果；资料里没有的规则不要编造。\n\n" + "\n\n".join(parts))


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def knowledge_numbers(hits: list[Hit]) -> list[float]:
    """检索结果里出现的数字（如平台红线 10%、往期退货率 8.1%）：报告引用它们算“有出处”。"""
    return [float(x) for h in hits for x in _NUM.findall(h.chunk.text)]
