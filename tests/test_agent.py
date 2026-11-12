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

    run = DataAnalystAgent(llm, review=False).run(df, "哪个品类卖得最好？")

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
    run = DataAnalystAgent(llm, review=False).run(df, "总销售额是多少？")
    assert run.steps[0].ok and "安全检查未通过" in run.steps[0].errors[0]
    assert run.unverified == []


def test_agent_falls_back_when_plan_is_not_json(df):
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    llm = FakeLLM(["我觉得应该先看看数据", code, "## 结论\n共有数据。"])
    run = DataAnalystAgent(llm, review=False).run(df, "数据有多少行？")
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
    run = DataAnalystAgent(llm, review=False).run(df, "数据质量如何？")
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
    run = DataAnalystAgent(llm, review=False).run(df, "清洗后有多少行？")
    assert run.steps[1].result.metrics["行数"] == 5           # 第 2 步拿到的是清洗后的数据
    assert run.steps[1].data_rows == 5
    assert "第1步清洗后的数据（5 行）" in llm.calls[2][1]     # 告诉了模型数据已清洗


# ---------- 多轮对话：历史会带进提示词 ----------
def test_history_is_passed_to_planner(df):
    from analyst.agent import summarize_for_history
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    llm = FakeLLM(["不是JSON", code, "## 结论\n- 钓竿最高。\n## 依据\n略"])
    first = DataAnalystAgent(llm, skills=[], review=False).run(df, "哪个品类最好？")
    assert summarize_for_history(first) == "- 钓竿最高。"
    llm2 = FakeLLM(["不是JSON", code, "## 结论\n原因是……"])
    DataAnalystAgent(llm2, skills=[], review=False).run(df, "那它为什么好？", history=[("哪个品类最好？", "- 钓竿最高。")])
    assert "第1轮结论：- 钓竿最高。" in llm2.calls[0][1]


# ---------- 程序兜底：去掉“汇总/综合判断”类的最后一步 ----------
def test_summary_step_is_dropped(df):
    plan = ('{"steps": [{"id": 1, "goal": "清洗", "method": "去重"},'
            ' {"id": 2, "goal": "各品类销售额", "method": "分组"},'
            ' {"id": 3, "goal": "汇总各品类指标供报告比较", "method": "合并"}]}')
    code = "```python\nresult['metrics']={'行数': len(df)}\n```"
    llm = FakeLLM([plan, code, code, "## 结论\n完成。"])
    run = DataAnalystAgent(llm, skills=[], review=False).run(df, "各品类表现？")
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


# ---------- 评测集：打分逻辑本身要可靠 ----------
def test_eval_cases_file_is_valid():
    import json
    from evals.run_eval import CHECKS, CASES
    spec = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(spec["cases"]) >= 10
    for c in spec["cases"]:
        assert c["checks"] and set(c["checks"]) <= set(CHECKS), c["id"]


def test_eval_grading_pass_and_fail(df):
    from evals.run_eval import evaluate
    plan = '{"steps": [{"id": 1, "goal": "汇总", "method": "分组"}]}'
    code = ("```python\ng = df.drop_duplicates().groupby('品类')['销售额'].sum()\n"
            "result['tables'] = {'品类': g.to_frame()}\n```")
    case_ok = {"id": "T1", "category": "测试", "question": "钓竿销售额？",
               "checks": {"no_failed_steps": True,
                          "result_has_number": [{"label": "钓竿", "values": [250.0], "tol": 0.01}],
                          "report_contains": [["钓竿"]], "report_not_contains": ["综合评分"]}}
    case_bad = {"id": "T2", "category": "测试", "question": "钓竿销售额？",
                "checks": {"result_has_number": [{"label": "钓竿", "values": [999.0], "tol": 0.01}],
                           "report_not_contains": ["综合评分"]}}
    llm = FakeLLM([plan, code, "## 结论\n钓竿 250。", plan, code, "## 结论\n钓竿综合评分最高。"])
    agent = DataAnalystAgent(llm, skills=[], review=False)
    ok, bad = evaluate(agent, df, [case_ok, case_bad], log=lambda *_: None)
    assert ok["passed"] is True
    assert bad["passed"] is False
    failed = {c["check"] for c in bad["checks"] if not c["ok"]}
    assert failed == {"数字正确", "禁止内容"}


# ---------- 工具模式：Function Calling + 工具权限 ----------
from analyst.tool_agent import ToolCallingAgent  # noqa: E402
from analyst.tools import READ, ToolBox  # noqa: E402

GOOD_CODE = ("s = df.groupby('品类')['销售额'].sum()\n"
             "result['metrics'] = {'钓竿销售额': float(s['钓竿'])}\nresult['notes'] = '按品类汇总'")


def test_tool_agent_full_loop(df):
    llm = FakeLLM([
        {"tool_calls": [("describe_column", {"column": "品类"})]},
        {"tool_calls": [("run_python", {"purpose": "算钓竿销售额", "code": GOOD_CODE})]},
        {"tool_calls": [("submit_report", {"report": "## 结论\n钓竿销售额 250.0。"})]},
    ])
    run = ToolCallingAgent(llm, skills=[], review=False).run(df, "钓竿销售额是多少？")
    assert run.mode == "tools"
    assert [t["name"] for t in run.tool_calls] == ["describe_column", "run_python", "submit_report"]
    assert len(run.steps) == 1 and run.steps[0].ok
    assert run.steps[0].result.metrics["钓竿销售额"] == 250.0
    assert run.report.startswith("## 结论") and run.unverified == []


def test_tool_agent_unsafe_code_is_blocked_then_retried(df):
    llm = FakeLLM([
        {"tool_calls": [("run_python", {"purpose": "算销售额", "code": "import os\nos.listdir('.')"})]},
        {"tool_calls": [("run_python", {"purpose": "算销售额", "code": GOOD_CODE})]},
        {"tool_calls": [("submit_report", {"report": "钓竿销售额 250.0"})]},
    ])
    run = ToolCallingAgent(llm, skills=[], review=False).run(df, "钓竿销售额？")
    first = run.tool_calls[0]
    assert not first["ok"] and "安全检查未通过" in first["result"]
    # 失败后紧接着的重试算同一步：1 步，尝试 2 次，最终成功
    assert len(run.steps) == 1 and run.steps[0].attempts == 2 and run.steps[0].ok


def test_tool_call_limit_stops_endless_loop(df):
    endless = [{"tool_calls": [("preview_data", {"n": 3})]}] * 3
    llm = FakeLLM(endless + ["## 结论\n达到上限后写的报告"])
    run = ToolCallingAgent(llm, max_tool_calls=3, skills=[], review=False).run(df, "随便看看")
    assert len(run.tool_calls) == 3
    assert "达到上限后写的报告" in run.report
    assert any(e["event"] == "tool_limit" for e in run.log)


def test_toolbox_permissions(df):
    box = ToolBox(df, skills=[], allowed_levels={READ})
    # 只读权限下：run_python 不出现在工具清单里，强行调用也会被拒绝
    assert "run_python" not in [t["function"]["name"] for t in box.schemas()]
    out = box.call("run_python", {"purpose": "x", "code": "result['notes']='x'"})
    assert not out.ok and "权限不足" in out.text
    # 不存在的工具（比如读写文件）根本不提供
    out = box.call("write_file", {"path": "a.txt"})
    assert not out.ok and "不提供" in out.text
    assert box.call("preview_data", {"n": 2}).ok


def test_toolbox_cleaned_data_used_by_later_calls(df):
    box = ToolBox(df, skills=[])
    out = box.call("run_python", {"purpose": "清洗", "code":
                   "c = df.drop_duplicates()\nresult['cleaned_df'] = c\nresult['notes'] = '去重'"})
    assert out.ok and len(box.df) == 5
    assert "共 5 行" in box.call("preview_data", {"n": 2}).text


def test_tool_agent_loads_skill_by_model_choice(df):
    from analyst.skills import load_skills
    llm = FakeLLM([
        {"tool_calls": [("load_skill", {"name": "电商指标"})]},
        {"tool_calls": [("submit_report", {"report": "无"})]},
    ])
    run = ToolCallingAgent(llm, skills=load_skills(), review=False).run(df, "退货率？")
    assert [n for n, _ in run.skills] == ["电商指标"]
    assert "退货率" in run.tool_calls[0]["result"]


def test_eval_forbidden_word_ignores_negation():
    from evals.run_eval import _affirmative_hit
    assert _affirmative_hit("给出综合评分：钓竿 90 分", "综合评分")
    assert not _affirmative_hit("分别列示，未做综合评分", "综合评分")
    assert _affirmative_hit("未做综合评分。但最后给出综合评分 8 分", "综合评分")


# ---------- 独立审查 Agent ----------
PLAN1 = '{"steps": [{"id": 1, "goal": "各品类销售额", "method": "分组求和"}]}'
CODE1 = ("```python\ng = df.drop_duplicates().groupby('品类')['销售额'].sum()\n"
         "result['tables'] = {'品类销售额': g.to_frame()}\nresult['notes'] = '按品类汇总'\n```")


def test_reviewer_passes_consistent_report(df):
    llm = FakeLLM([PLAN1, CODE1, "## 结论\n钓竿销售额最高，为 250。", '{"issues": []}'])
    run = DataAnalystAgent(llm, skills=[]).run(df, "哪个品类销售额最高？")
    assert run.review.passed and not run.review.revised and run.review.issues == []
    # 审查员拿到的是“问题 + 步骤结果 + 报告”，看不到写报告时的提示词
    reviewer_user = llm.calls[-1][1]
    assert "待审查的报告" in reviewer_user and "品类销售额" in reviewer_user


def test_reviewer_catches_contradiction_and_report_is_revised(df):
    bad = "## 结论\n鱼饵销售额最高。"
    review = ('{"issues": [{"severity": "严重", "claim": "鱼饵销售额最高", '
              '"evidence": "表格中钓竿 250 最高，鱼饵 8 最低"}]}')
    llm = FakeLLM([PLAN1, CODE1, bad, review, "## 结论\n钓竿销售额最高，为 250。"])
    run = DataAnalystAgent(llm, skills=[]).run(df, "哪个品类销售额最高？")
    assert run.review.revised and not run.review.passed
    assert "钓竿销售额最高" in run.report and "鱼饵销售额最高" not in run.report
    # 修改报告时，审查意见交给了写报告的模型
    assert "鱼饵销售额最高" in llm.calls[-1][1] and "审查员" in llm.calls[-1][1]


def test_reviewer_minor_issue_does_not_trigger_revision(df):
    review = '{"issues": [{"severity": "提醒", "claim": "大幅领先", "evidence": "表述略夸张"}]}'
    llm = FakeLLM([PLAN1, CODE1, "## 结论\n钓竿大幅领先。", review])
    run = DataAnalystAgent(llm, skills=[]).run(df, "哪个品类销售额最高？")
    assert run.review.passed and not run.review.revised and len(run.review.issues) == 1


def test_reviewer_failure_does_not_break_run(df):
    llm = FakeLLM([PLAN1, CODE1, "## 结论\n钓竿最高。", "我觉得还行"])   # 审查员没按格式回答
    run = DataAnalystAgent(llm, skills=[]).run(df, "哪个品类销售额最高？")
    assert run.report == "## 结论\n钓竿最高。" and "无法解析" in run.review.error


def test_reviewer_can_be_a_different_model(df):
    writer = FakeLLM([PLAN1, CODE1, "## 结论\n钓竿最高。"])
    reviewer = FakeLLM(['{"issues": []}'])
    run = DataAnalystAgent(writer, skills=[], reviewer=reviewer).run(df, "哪个品类销售额最高？")
    assert len(reviewer.calls) == 1 and len(writer.calls) == 3 and run.review.passed


def test_tool_mode_is_reviewed_too(df):
    llm = FakeLLM([
        {"tool_calls": [("run_python", {"purpose": "算钓竿销售额", "code": GOOD_CODE})]},
        {"tool_calls": [("submit_report", {"report": "## 结论\n钓竿销售额 250.0。"})]},
        '{"issues": []}',
    ])
    run = ToolCallingAgent(llm, skills=[]).run(df, "钓竿销售额？")
    assert run.review is not None and run.review.passed


def test_reviewer_sees_profile_and_read_tool_results(df):
    # 模型只用只读工具就回答了行数：审查员必须能看到概况和只读工具结果，才不会误判为“编造”
    llm = FakeLLM([
        {"tool_calls": [("describe_column", {"column": "品类"})]},
        {"tool_calls": [("submit_report", {"report": "## 结论\n共 6 行、3 列。"})]},
        '{"issues": []}',
    ])
    ToolCallingAgent(llm, skills=[]).run(df, "数据有多少行多少列？")
    reviewer_user = llm.calls[-1][1]
    assert "数据概况（程序统计，真实）" in reviewer_user
    assert "只读工具返回的结果" in reviewer_user and "列“品类”" in reviewer_user


def test_eval_negation_includes_bu_she():
    from evals.run_eval import _affirmative_hit
    assert not _affirmative_hit("重点投入候选的对比（不设综合评分，分维度列出）", "综合评分")
