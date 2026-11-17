"""LangGraph 版规划-执行模式的离线测试（升级 12）。

重点：同样的模型回答，LangGraph 版和手写版要走出同样的流程、得到同样的结果；
再单独测图的结构、重试上限、断点续跑。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst.agent import DataAnalystAgent  # noqa: E402
from analyst.graph_agent import LangGraphAnalystAgent, graph_mermaid, make_checkpointer  # noqa: E402
from analyst.llm import FakeLLM  # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame({
        "品类": ["钓竿", "鱼线", "钓竿", "鱼饵", "鱼线", "鱼线"],
        "销售额": [100.0, 20.0, 150.0, 8.0, 24.0, 24.0],
        "评分": [5, 4, np.nan, 3, 4, 4],
    })


PLAN1 = '{"steps": [{"id": 1, "goal": "统计各品类销售额", "method": "groupby 求和"}]}'
BAD = "```python\nresult['metrics'] = {'总销售额': df['销量'].sum()}\n```"      # 列名写错
GOOD = ("```python\ng = df.groupby('品类')['销售额'].sum()\n"
        "result['metrics'] = {'总销售额': float(g.sum())}\n"
        "result['tables'] = {'各品类销售额': g.to_frame()}\n"
        "result['notes'] = '钓竿最高'\n```")
REPORT = "## 结论\n- 总销售额 326，钓竿 250 最高。\n- 预计明年增长 30.5%。"

PLAN2 = ('{"steps": [{"id": 1, "goal": "清洗", "method": "去重"}, '
         '{"id": 2, "goal": "各品类销售额", "method": "分组"}]}')
CLEAN = "```python\nresult['cleaned_df'] = df.drop_duplicates()\nresult['notes'] = '去重'\n```"
SUM = ("```python\nresult['metrics'] = {'行数': len(df)}\n"
       "result['tables'] = {'品类': df.groupby('品类')['销售额'].sum().to_frame()}\n```")


def test_graph_has_expected_nodes_and_loops():
    g = LangGraphAnalystAgent(llm=None, skills=[]).build_graph(
        pd.DataFrame({"x": [1]}), None, lambda *_: None).get_graph()
    assert set(g.nodes) >= {"感知", "规划", "写代码", "执行", "修代码", "收尾本步", "写报告", "核对与审查"}
    edges = {(e.source, e.target, e.conditional) for e in g.edges}
    assert ("执行", "修代码", True) in edges and ("修代码", "执行", False) in edges   # 出错重试的回环
    assert ("收尾本步", "写代码", True) in edges                                      # 还有步骤就继续
    assert "核对与审查" in graph_mermaid()


def test_end_to_end_with_retry(df):
    llm = FakeLLM([PLAN1, BAD, GOOD, REPORT])
    run = LangGraphAnalystAgent(llm, review=False, skills=[]).run(df, "哪个品类卖得最好？")
    step = run.steps[0]
    assert run.mode == "graph"
    assert step.ok and step.attempts == 2 and "销量" in step.errors[0]
    assert step.result.metrics["总销售额"] == pytest.approx(326.0)
    assert run.unverified == ["30.5%"]
    assert "钓竿 250" in run.report


def test_same_answers_give_same_result_as_handwritten(df):
    answers = [PLAN2, CLEAN, BAD, SUM, "## 结论\n清洗后 5 行。"]
    hand_llm, graph_llm = FakeLLM(list(answers)), FakeLLM(list(answers))
    hand = DataAnalystAgent(hand_llm, review=False, skills=[]).run(df, "各品类销售额？")
    graph = LangGraphAnalystAgent(graph_llm, review=False, skills=[]).run(df, "各品类销售额？")
    assert hand_llm.calls == graph_llm.calls                    # 每次发给模型的内容完全一样
    assert graph.report == hand.report
    assert [(s.ok, s.attempts, s.data_rows) for s in graph.steps] == \
           [(s.ok, s.attempts, s.data_rows) for s in hand.steps]
    assert [e["event"] for e in graph.log] == [e["event"] for e in hand.log]


def test_cleaned_data_is_shared(df):
    llm = FakeLLM([PLAN2, CLEAN, SUM, "## 结论\n完成。"])
    run = LangGraphAnalystAgent(llm, review=False, skills=[]).run(df, "清洗后有多少行？")
    assert run.steps[1].result.metrics["行数"] == 5
    assert "第1步清洗后的数据（5 行）" in llm.calls[2][1]


def test_retry_limit_then_report_anyway(df):
    llm = FakeLLM([PLAN1, BAD, BAD, "## 结论\n计算失败。"])
    run = LangGraphAnalystAgent(llm, review=False, skills=[], max_retries=1).run(df, "销售额？")
    assert not run.steps[0].ok and run.steps[0].attempts == 2
    assert run.report.startswith("## 结论")


def test_review_runs_inside_graph(df):
    review = '{"issues": [{"severity": "严重", "claim": "钓竿大幅领先", "evidence": "250 vs 68"}]}'
    llm = FakeLLM([PLAN1, GOOD, "## 结论\n钓竿大幅领先。", review, "## 结论\n钓竿 250 最高。"])
    run = LangGraphAnalystAgent(llm, skills=[]).run(df, "哪个品类最好？")
    assert run.review is not None and run.review.revised
    assert run.report == "## 结论\n钓竿 250 最高。"


def test_usage_is_tracked_per_stage(df):
    run = LangGraphAnalystAgent(FakeLLM([PLAN1, BAD, GOOD, REPORT]), review=False,
                                skills=[]).run(df, "销售额？")
    stages = {s["stage"] for s in run.usage["by_stage"]}
    assert {"规划", "写代码", "修代码", "写报告"} <= stages


def test_resume_from_sqlite_checkpoint(df, tmp_path):
    saver = make_checkpointer(tmp_path / "g.sqlite")
    # 第一次：规划和第 1 步完成后，模型接口“断了”（预设回答用完会报错）
    with pytest.raises(RuntimeError):
        LangGraphAnalystAgent(FakeLLM([PLAN2, CLEAN]), review=False, skills=[],
                              checkpointer=saver).run(df, "各品类销售额？")
    # 第二次：同一任务，从第 2 步的“写代码”继续——不再规划、不重跑第 1 步
    llm2 = FakeLLM([SUM, "## 结论\n完成。"])
    run = LangGraphAnalystAgent(llm2, review=False, skills=[],
                                checkpointer=saver).run(df, "各品类销售额？")
    assert len(llm2.calls) == 2 and run.resumed_steps == 1
    assert [s.ok for s in run.steps] == [True, True]
    assert run.steps[1].data_rows == 5                    # 用的是存档里清洗后的数据
    resume = next(e for e in run.log if e["event"] == "resume")
    assert resume["detail"]["next"] == ["写代码"]
    # 完成后存档已删除：再问一次会从头开始
    llm3 = FakeLLM([PLAN1, GOOD, "## 结论\n完成。"])
    again = LangGraphAnalystAgent(llm3, review=False, skills=[],
                                  checkpointer=saver).run(df, "各品类销售额？")
    assert again.resumed_steps == 0 and len(llm3.calls) == 3
