"""业务知识库（RAG）的离线测试：不下载向量模型，用一个按字符计数的假向量模型代替。

运行：.venv\\Scripts\\python -m pytest -q tests/test_knowledge.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyst.knowledge as K  # noqa: E402
from analyst.agent import DataAnalystAgent  # noqa: E402
from analyst.knowledge import KnowledgeBase, load_chunks, render_knowledge, tokenize  # noqa: E402
from analyst.llm import FakeLLM  # noqa: E402
from analyst.tool_agent import ToolCallingAgent  # noqa: E402
from analyst.tools import ToolBox  # noqa: E402


class CharEmbedder:
    """假向量模型：把每个字哈希到 64 维里计数。意思相近靠“有相同的字”，够测试流程用。"""
    name = "char-test"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.replace(K.QUERY_PREFIX, ""):
                out[i, hash(ch) % 64] += 1
        return out


class BrokenEmbedder:
    name = "broken"

    def embed(self, texts):
        raise OSError("模型下载失败")


@pytest.fixture
def df():
    return pd.DataFrame({"品类": ["钓竿", "鱼线", "钓竿"], "销售额": [100.0, 20.0, 150.0],
                         "是否退货": ["是", "否", "否"]})


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "CACHE_DIR", tmp_path / "kb_cache")


def kw_kb():
    return KnowledgeBase(embedder=None)


# ---------- 读取、切分、分词 ----------
def test_load_chunks_splits_by_section_and_skips_notes():
    chunks = load_chunks()
    docs = {c.doc for c in chunks}
    assert docs == {"数据字典", "指标口径与业务规则", "平台与行业规则", "历史经营报告"}
    titles = [c.title for c in chunks]
    assert "数据字典 · 广告花费" in titles and "平台与行业规则 · 平台退货率红线" in titles
    assert not any("模拟资料" in c.text for c in chunks)          # “>”开头的说明行不进正文


def test_tokenize_bigrams_and_ascii_words():
    assert tokenize("ACOS 多少") == ["acos", "多少"]
    assert tokenize("退货率") == ["退货", "货率"]


# ---------- 关键词检索 ----------
def test_keyword_search_finds_jargon_and_returns_empty_for_unrelated():
    kb = kw_kb()
    assert kb.mode == "keyword"
    top = [h.chunk.title for h in kb.search("GMV 包不包含退货？", k=3)]
    # 专有名词 GMV 能被关键词找到；但“包含”这个词在“广告花费”一节出现了两次，可能排得更前——
    # 这正是只用关键词的弱点，混合检索评测里 K17 就是这道题
    assert "指标口径与业务规则 · 销售额与净销售额" in top
    assert kb.search("帮我把电脑上的临时文件都删掉") == []


def test_weak_keyword_matches_are_filtered():
    kb = kw_kb()
    loose = KnowledgeBase(embedder=None, kw_min_ratio=0.0)
    q = "GMV 包不包含退货？"
    assert len(kb.search(q, k=50)) < len(loose.search(q, k=50))   # “退货”沾边的弱匹配被去掉


# ---------- 混合检索 ----------
def test_hybrid_search_fuses_both_rankings():
    emb = CharEmbedder()
    kb = KnowledgeBase(embedder=emb, min_similarity=0.0)
    hits = kb.search("广告花费包含哪些费用", k=3)
    assert hits[0].chunk.title == "数据字典 · 广告花费"
    assert hits[0].keyword_rank is not None and hits[0].vector_rank is not None
    assert hits[0].score == pytest.approx(1 / (K.RRF_K + hits[0].keyword_rank) + 1 / (K.RRF_K + hits[0].vector_rank))
    assert "混合检索可用" in kb.status
    # 只用向量：没有关键词排名
    assert all(h.keyword_rank is None for h in kb.search("广告花费包含哪些费用", mode="vector"))


def test_vectors_are_cached_between_instances():
    emb = CharEmbedder()
    KnowledgeBase(embedder=emb, min_similarity=0.0).search("退货率")
    first = emb.calls                                   # 文档向量 1 次 + 问题 1 次
    emb2 = CharEmbedder()
    KnowledgeBase(embedder=emb2, min_similarity=0.0).search("退货率")
    assert first == 2 and emb2.calls == 1               # 第二次文档向量来自缓存，只算问题


def test_broken_embedder_falls_back_to_keyword():
    kb = KnowledgeBase(embedder=BrokenEmbedder())
    hits = kb.search("GMV 包不包含退货？")
    assert hits and hits[0].vector_rank is None
    assert kb.mode == "keyword" and "退回关键词检索" in kb.status


def test_render_and_numbers():
    hits = kw_kb().search("钓竿的退货率有没有超过平台红线？", k=2)
    text = render_knowledge(hits)
    assert "来源：平台与行业规则 · 平台退货率红线" in text and "往期数字只能用来对比" in text
    assert 10.0 in K.knowledge_numbers(hits)
    assert render_knowledge([]) == ""


# ---------- 接入规划-执行模式 ----------
PLAN = '{"steps": [{"id": 1, "goal": "各品类退货率", "method": "分组"}]}'
CODE = ("```python\nr = df.assign(退=df['是否退货'].eq('是')).groupby('品类')['退'].mean()\n"
        "result['tables'] = {'退货率': r.to_frame()}\nresult['notes'] = '按品类'\n```")
REPORT = "## 结论\n钓竿退货率 0.5，超过平台 10% 的红线（来源：平台与行业规则）。"


def test_plan_mode_puts_knowledge_into_prompts_and_review(df):
    llm = FakeLLM([PLAN, CODE, REPORT, '{"issues": []}'])
    run = DataAnalystAgent(llm, skills=[], knowledge=kw_kb()).run(df, "各品类退货率有没有超过平台红线？")
    for _, user in llm.calls[:3]:                       # 规划、写代码、写报告都看得到
        assert "业务知识库检索结果" in user and "平台退货率红线" in user
    assert "平台退货率红线" in llm.calls[3][1]           # 审查员也看得到，不会误判“10%”是编造
    assert any(k["title"] == "平台与行业规则 · 平台退货率红线" for k in run.knowledge)
    assert "10" not in " ".join(run.unverified)         # 10% 来自知识库，算有出处


def test_plan_mode_without_knowledge_is_unchanged(df):
    llm = FakeLLM([PLAN, CODE, "## 结论\n完成。"])
    run = DataAnalystAgent(llm, skills=[], review=False).run(df, "各品类退货率？")
    assert run.knowledge == [] and not any("业务知识库" in u for _, u in llm.calls)


# ---------- 接入工具模式 ----------
def test_search_tool_only_exists_with_knowledge(df):
    assert "search_knowledge" not in {s["function"]["name"] for s in ToolBox(df, []).schemas()}
    box = ToolBox(df, [], knowledge=kw_kb())
    assert "search_knowledge" in {s["function"]["name"] for s in box.schemas()}
    out = box.call("search_knowledge", {"query": "评分为空是不是差评"})
    assert out.ok and "来源：数据字典 · 评分" in out.text and box.kb_hits
    none = box.call("search_knowledge", {"query": "今天天气怎么样"})
    assert none.ok and "没有找到" in none.text


def test_tool_mode_model_decides_to_search(df):
    llm = FakeLLM([
        {"tool_calls": [("search_knowledge", {"query": "平台退货率红线"})]},
        {"tool_calls": [("submit_report", {"report": "## 结论\n平台红线是 10%（来源：平台与行业规则）。"})]},
        '{"issues": []}',
    ])
    run = ToolCallingAgent(llm, skills=[], knowledge=kw_kb()).run(df, "平台的退货率红线是多少？")
    assert run.tool_calls[0]["name"] == "search_knowledge"
    assert any("平台退货率红线" in k["title"] for k in run.knowledge)
    assert run.unverified == []                          # 10 来自检索结果
    assert "平台退货率红线" in llm.calls[-1][1]           # 检索结果作为只读工具结果交给审查员


# ---------- MCP、网页、评测 ----------
def test_mcp_server_offers_search_knowledge(monkeypatch):
    import analyst.mcp_server as ms
    from tests.test_agent import _mcp_calls
    monkeypatch.setattr(ms, "default_knowledge", kw_kb)      # 测试里不加载真实向量模型
    tools, outs = _mcp_calls([("search_knowledge", {"query": "GMV 包不包含退货"})])
    assert tools["search_knowledge"].annotations.read_only_hint is True
    assert "销售额与净销售额" in outs[0]


def test_event_text_and_eval_check():
    from api.serialize import event_text
    from evals.run_eval import chk_knowledge_include
    assert "📚" in event_text("knowledge", [{"title": "数据字典 · 评分"}])
    assert "没有找到" in event_text("knowledge", [])

    class R:
        knowledge = [{"title": "数据字典 · 评分"}]
    assert chk_knowledge_include(R, [["数据字典 · 评分"]])[0]
    assert not chk_knowledge_include(R, [["数据字典 · 评分"], ["平台与行业规则 · 平台退货率红线"]])[0]


def test_eval_value_check_is_not_fooled_by_substrings():
    # 对照实验里真实发生过：没接知识库的报告说“无法判断”，却因为“68.18”含“8.1”、“7.16%”含“6%”通过了字面检查
    from evals.run_eval import chk_report_has_value

    class Bad:
        report = "单价为中位数的 68.18 倍；退货金额率 7.16%。"

    class Good:
        report = "较 2024 年的 8.1% 下降，但未达到 6% 以下的目标；超 10 % 审查线。"
    assert not chk_report_has_value(Bad, ["8.1%", "6%"])[0]
    assert chk_report_has_value(Good, ["8.1%", "6%", "10%"])[0]
