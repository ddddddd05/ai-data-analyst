"""离线测试：用 FakeLLM 代替真实模型，检查各个环节是否按预期工作。

运行：.venv\\Scripts\\python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst.agent import DataAnalystAgent  # noqa: E402
from analyst.llm import FakeLLM, extract_code, extract_json  # noqa: E402
from analyst.profiler import profile, profile_to_text  # noqa: E402
from analyst.sandbox import run_code, static_check  # noqa: E402
from analyst.validator import verify_report_numbers  # noqa: E402
from make_sample_data import make  # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame({
        "品类": ["钓竿", "鱼线", "钓竿", "鱼饵", "鱼线", "鱼线"],
        "销售额": [100.0, 20.0, 150.0, 8.0, 24.0, 24.0],
        "评分": [5, 4, np.nan, 3, 4, 4],
    })


# ---------- 感知：数据概况 ----------
def test_profile_counts_missing_and_duplicates(df):
    p = profile(df)
    assert p["rows"] == 6 and p["duplicate_rows"] == 1
    rating = next(c for c in p["columns"] if c["name"] == "评分")
    assert rating["missing"] == 1
    assert "重复的行 1 行" in profile_to_text(p)


def test_date_columns_are_parsed(tmp_path):
    from analyst.profiler import load_table
    f = tmp_path / "d.csv"
    pd.DataFrame({"下单日期": ["2025-01-01", "2025-02-03"], "销售额": [1, 2]}).to_csv(f, index=False)
    assert pd.api.types.is_datetime64_any_dtype(load_table(f)["下单日期"])


def test_sample_data_has_planted_problems():
    data = make()
    p = profile(data)
    assert p["duplicate_rows"] == 30
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["评分"]["missing"] > 0
    assert cols["数量"]["negative"] == 3
    assert cols["单价"]["outliers"] >= 6


# ---------- 护栏：静态安全检查 ----------
@pytest.mark.parametrize("code,keyword", [
    ("import os\nos.remove('a')", "os"),
    ("open('x.txt','w')", "open"),
    ("df.to_csv('out.csv')", "to_csv"),
    ("().__class__.__bases__", "__class__"),
    ("import requests", "requests"),
])
def test_static_check_blocks_dangerous_code(code, keyword):
    problems = static_check(code)
    assert problems and any(keyword in p for p in problems)


def test_static_check_allows_normal_analysis():
    code = "import numpy as np\nresult['metrics']={'总销售额': df['销售额'].sum()}"
    assert static_check(code) == []


# ---------- 工具：执行代码 ----------
def test_run_code_returns_metrics_tables_and_figures(df):
    code = """
g = df.groupby('品类')['销售额'].sum().sort_values(ascending=False)
result['metrics'] = {'总销售额': float(df['销售额'].sum())}
result['tables'] = {'各品类销售额': g.to_frame()}
result['notes'] = '钓竿销售额最高'
g.plot(kind='bar', title='各品类销售额')
"""
    res = run_code(code, df)
    assert res.ok, res.error
    assert res.metrics["总销售额"] == pytest.approx(326.0)
    assert "各品类销售额" in res.tables
    assert len(res.figures) == 1 and res.figures[0][:4] == b"\x89PNG"


def test_run_code_reports_errors(df):
    res = run_code("result['metrics'] = {'x': df['不存在的列'].sum()}", df)
    assert not res.ok and "不存在的列" in res.error


def test_run_code_timeout(df):
    res = run_code("while True:\n    pass", df, timeout=3)
    assert not res.ok and "超过" in res.error


# ---------- 校验：报告数字核对 ----------
def test_verify_report_numbers():
    known = [326.0, 0.4601, 150.0]
    report = "总销售额 326 元，钓竿占比 46.0%，最高单笔 150，平均客单价 88.5 元，共 3 个品类，2025-05-01 起。"
    assert verify_report_numbers(report, known) == ["88.5"]


# ---------- 解析 ----------
def test_extract_helpers():
    assert extract_json('好的：```json\n{"steps": [{"id": 1}]}\n```')["steps"][0]["id"] == 1
    assert extract_code("```python\nprint(1)\n```") == "print(1)"


# ---------- 端到端：规划 → 写代码 → 出错修正 → 报告 → 校验 ----------
def test_agent_end_to_end_with_retry(df):
    plan = '{"steps": [{"id": 1, "goal": "统计各品类销售额", "method": "groupby 求和"}]}'
    bad_code = "```python\nresult['metrics'] = {'总销售额': df['销量'].sum()}\n```"   # 列名写错
    good_code = ("```python\ng = df.groupby('品类')['销售额'].sum()\n"
                 "result['metrics'] = {'总销售额': float(g.sum())}\n"
                 "result['tables'] = {'各品类销售额': g.to_frame()}\n"
                 "result['notes'] = '钓竿最高'\n```")
    report = "## 结论\n- 总销售额 326，钓竿 250 最高。\n- 预计明年增长 30.5%。"
    llm = FakeLLM([plan, bad_code, good_code, report])

    run = DataAnalystAgent(llm).run(df, "哪个品类卖得最好？")

    step = run.steps[0]
    assert step.ok and step.attempts == 2 and "销量" in step.errors[0]
    assert step.result.metrics["总销售额"] == pytest.approx(326.0)
    assert run.unverified == ["30.5%"]          # 编造的增长率被识别出来
    assert "钓竿 250" in run.report             # 250 来自汇总表，能核对上


def test_agent_blocks_unsafe_code_then_recovers(df):
    plan = '{"steps": [{"id": 1, "goal": "总销售额", "method": "求和"}]}'
    unsafe = "```python\nimport os\nresult['metrics']={'x': 1}\n```"
    safe = "```python\nresult['metrics']={'总销售额': float(df['销售额'].sum())}\n```"
    llm = FakeLLM([plan, unsafe, safe, "## 结论\n总销售额 326。"])
    run = DataAnalystAgent(llm).run(df, "总销售额是多少？")
    assert run.steps[0].ok and "安全检查未通过" in run.steps[0].errors[0]
    assert run.unverified == []


def test_agent_falls_back_when_plan_is_not_json(df):
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    llm = FakeLLM(["我觉得应该先看看数据", code, "## 结论\n共有数据。"])
    run = DataAnalystAgent(llm).run(df, "数据有多少行？")
    assert len(run.plan) == 1 and run.steps[0].ok


# ---------- Skill：按需加载 ----------
def test_skill_file_parses_and_loads():
    from analyst.skills import load_skills
    skills = {s.name: s for s in load_skills()}
    s = skills["数据清洗"]
    assert "极端值" in s.when and "清洗" in s.keywords
    assert "同类别中位数 10 倍" in s.body


def test_skill_selected_only_when_needed(df):
    from analyst.skills import load_skills, select_skills
    skills = load_skills()
    names = lambda chosen: sorted(s.name for s, _ in chosen)  # noqa: E731
    clean = pd.DataFrame({"品类": ["A", "B", "C"], "销售额": [1.0, 2.0, 3.0]})

    # 数据清洗：看数据状况
    dirty = dict((s.name, why) for s, why in select_skills(skills, "数据有多少行？", profile(df)))
    assert list(dirty) == ["数据清洗"] and "重复行" in dirty["数据清洗"]
    assert select_skills(skills, "数据有多少行？", profile(clean)) == []
    assert names(select_skills(skills, "帮我做数据清洗", profile(clean))) == ["数据清洗"]

    # 电商指标：看问题里有没有电商相关的词
    assert names(select_skills(skills, "各品类销售额？", profile(clean))) == ["电商指标"]
    assert names(select_skills(skills, "各品类销售额？", profile(df))) == ["数据清洗", "电商指标"]


def test_ecommerce_skill_defines_metrics():
    from analyst.skills import load_skills
    s = {k.name: k for k in load_skills()}["电商指标"]
    assert s.when == []                                   # 只看问题关键词，不看数据状况
    for word in ("净销售额", "ROAS", "ACOS", "不要在报告里心算", "不要**自己设定权重"):
        assert word in s.body


def test_skill_text_reaches_prompts(df):
    plan = '{"steps": [{"id": 1, "goal": "清洗", "method": "去重"}]}'
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    llm = FakeLLM([plan, code, "## 结论\n完成。"])
    run = DataAnalystAgent(llm).run(df, "数据质量如何？")
    assert run.skills and run.skills[0][0] == "数据清洗"
    planner_user, coder_user, reporter_user = (c[1] for c in llm.calls)
    for text in (planner_user, coder_user, reporter_user):
        assert "Skill：数据清洗" in text


# ---------- 清洗后的数据交给后续步骤共用 ----------
def test_cleaned_data_is_shared_by_later_steps(df):
    plan = ('{"steps": [{"id": 1, "goal": "清洗", "method": "去重"},'
            ' {"id": 2, "goal": "统计行数", "method": "len"}]}')
    clean_code = ("```python\nd = df.drop_duplicates()\nresult['cleaned_df'] = d\n"
                  "result['notes'] = '删除重复 1 行'\n```")
    count_code = "```python\nresult['metrics'] = {'行数': len(df)}\n```"
    llm = FakeLLM([plan, clean_code, count_code, "## 结论\n清洗后 5 行。"])
    run = DataAnalystAgent(llm).run(df, "清洗后有多少行？")
    assert run.steps[1].result.metrics["行数"] == 5           # 第 2 步拿到的是清洗后的数据
    assert run.steps[1].data_rows == 5
    assert "第1步清洗后的数据（5 行）" in llm.calls[2][1]     # 告诉了模型数据已清洗


# ---------- 多轮对话：历史会带进提示词 ----------
def test_history_is_passed_to_planner(df):
    from analyst.agent import summarize_for_history
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    llm = FakeLLM(["不是JSON", code, "## 结论\n- 钓竿最高。\n## 依据\n略"])
    first = DataAnalystAgent(llm, skills=[]).run(df, "哪个品类最好？")
    assert summarize_for_history(first) == "- 钓竿最高。"
    llm2 = FakeLLM(["不是JSON", code, "## 结论\n原因是……"])
    DataAnalystAgent(llm2, skills=[]).run(df, "那它为什么好？", history=[("哪个品类最好？", "- 钓竿最高。")])
    assert "第1轮结论：- 钓竿最高。" in llm2.calls[0][1]


# ---------- 程序兜底：去掉“汇总/综合判断”类的最后一步 ----------
def test_summary_step_is_dropped(df):
    plan = ('{"steps": [{"id": 1, "goal": "清洗", "method": "去重"},'
            ' {"id": 2, "goal": "各品类销售额", "method": "分组"},'
            ' {"id": 3, "goal": "汇总各品类指标供报告比较", "method": "合并"}]}')
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    llm = FakeLLM([plan, code, code, "## 结论\n完成。"])
    run = DataAnalystAgent(llm, skills=[]).run(df, "各品类表现？")
    assert [s.goal for s in run.steps] == ["清洗", "各品类销售额"]
    assert any(e["event"] == "plan_trimmed" for e in run.log)


def test_short_plan_keeps_last_step(df):
    plan = ('{"steps": [{"id": 1, "goal": "清洗", "method": "去重"},'
            ' {"id": 2, "goal": "综合比较各品类", "method": "分组"}]}')
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    run = DataAnalystAgent(FakeLLM([plan, code, code, "## 结论\n完成。"]), skills=[]).run(df, "比较品类")
    assert len(run.steps) == 2                       # 只有两步时不删，避免把唯一的计算步骤删掉


def test_percent_without_sign_is_verified():
    assert verify_report_numbers("退货金额率(%)：钓竿 7.16，配件 1.68", [0.0716, 0.0168]) == []
