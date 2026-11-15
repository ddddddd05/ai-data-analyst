"""数据库模式（Text-to-SQL）和成本统计的离线测试：用 FakeLLM 代替真实模型，用小号的示例数据库。

运行：.venv\\Scripts\\python -m pytest -q tests/test_sql.py
"""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst import usage  # noqa: E402
from analyst.agent import DataAnalystAgent  # noqa: E402
from analyst.checkpoint import task_id  # noqa: E402
from analyst.database import Database, check_sql  # noqa: E402
from analyst.llm import FakeLLM, extract_sql  # noqa: E402
from analyst.sql_agent import SQLAnalystAgent  # noqa: E402
from analyst.tool_agent import ToolCallingAgent  # noqa: E402
from analyst.tools import ToolBox  # noqa: E402
from make_sample_db import build  # noqa: E402

REVIEW_OK = '{"issues": []}'


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    return build(tmp_path_factory.mktemp("db") / "shop.db", n_orders=300)


@pytest.fixture
def db(db_path):
    return Database(db_path)


def sql_block(sql: str) -> str:
    return f"```sql\n{sql}\n```"


CAT_SALES = ("SELECT p.category AS 品类, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS 销售额 "
             "FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category")


# ---------- 护栏：只读、单条、超时、行数上限 ----------
@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE order_items SET quantity = 0",
    "DROP TABLE orders",
    "SELECT 1; DROP TABLE orders",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'evil.db' AS evil",
    "INSERT INTO shops VALUES ('S9', 'x', 'y', '2025-01-01')",
])
def test_check_sql_blocks_writes(db, sql):
    res = db.run_sql(sql)
    assert not res.ok and "安全检查未通过" in res.error


def test_write_hidden_in_cte_is_denied_by_authorizer(db, db_path):
    """WITH 开头能骗过静态检查，但授权回调和只读连接会拦下真正的写操作。"""
    before = db.fingerprint()
    res = db.run_sql("WITH x AS (SELECT 1) DELETE FROM orders")
    assert check_sql("WITH x AS (SELECT 1) DELETE FROM orders") == []
    assert not res.ok and "只读" in res.error
    assert Database(db_path).fingerprint() == before


def test_semicolon_inside_string_is_allowed(db):
    assert check_sql("SELECT 'a;b' AS x") == []
    assert db.run_sql("SELECT 'a;b' AS x;").df.iloc[0, 0] == "a;b"


def test_query_timeout(db_path):
    slow = Database(db_path, timeout=0.5)
    res = slow.run_sql("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c")
    assert not res.ok and "中断" in res.error


def test_row_limit_truncates(db_path):
    small = Database(db_path, max_rows=10)
    res = small.run_sql("SELECT * FROM order_items")
    assert res.ok and len(res.df) == 10 and res.truncated


def test_schema_has_docs_samples_and_foreign_keys(db):
    text = db.schema_text(["order_items"])
    assert "退货率按订单计算" in text and "order_items.product_id → products.product_id" in text
    assert "钓竿" in db.schema_text(["products"])          # 类别列的取值样例
    assert "OD2" not in db.schema_text(["orders"])          # 订单号这种取值很多的列不列样例
    assert "ROAS" in db.metrics_text("各店铺的投入产出比")


# ---------- schema linking ----------
@pytest.mark.parametrize("question,expected", [
    ("各品类的销售额是多少？", {"order_items", "products"}),
    ("各店铺的订单数和销售额", {"order_items", "orders", "shops"}),
    ("哪个店铺的广告投入产出比最高？", {"ad_spend", "order_items", "orders", "shops"}),
    ("每个月的销售额趋势", {"order_items", "orders"}),
    ("各地区的客单价", {"order_items", "orders"}),
])
def test_schema_linking_selects_needed_tables(db, question, expected):
    assert {t for t, _ in db.link(question)} == expected


def test_value_linking_prefers_dimension_table_and_adds_bridge(db):
    """“钓竿”在 products 和 ad_spend 里都有，应取 products；products 和 shops 之间要经过 order_items、orders。"""
    linked = dict(db.link("店铺C 在加州卖了多少钓竿？"))
    assert set(linked) == {"order_items", "orders", "products", "shops"}
    assert "钓竿" in linked["products"] and "ad_spend" not in linked


def test_unrelated_question_gets_all_tables(db):
    assert len(db.link("今天天气怎么样")) == len(db.tables)


# ---------- 规划-执行模式：每一步一条 SQL ----------
def test_sql_agent_end_to_end(db):
    llm = FakeLLM([
        '{"steps": [{"id": 1, "goal": "各品类销售额", "method": "明细关联商品，按品类求和"}]}',
        sql_block(CAT_SALES),
        "## 结论\n钓竿销售额最高。",
        REVIEW_OK,
    ])
    run = SQLAnalystAgent(llm, db).run("各品类的销售额是多少？")
    assert run.mode == "sql" and run.steps[0].ok
    assert {t["table"] for t in run.sql} == {"order_items", "products"}
    planner_prompt = llm.calls[0][1]
    assert "ad_spend" in planner_prompt                     # 一行简介里有全部表
    assert "表 ad_spend" not in planner_prompt              # 但完整结构只给相关的表
    assert "SUM(order_items.unit_price * order_items.quantity)" in planner_prompt   # 指标口径
    assert set(run.steps[0].result.tables["查询结果"]["品类"]) >= {"钓竿", "鱼线"}
    assert run.unverified == []


def test_sql_error_is_fixed_with_full_schema(db):
    """写了不存在的列 → 报错交回模型，并附上全部表结构（弥补 schema linking 漏表）。"""
    llm = FakeLLM([
        '{"steps": [{"id": 1, "goal": "各品类销售额", "method": "求和"}]}',
        sql_block("SELECT category, SUM(sales) FROM order_items GROUP BY category"),
        sql_block(CAT_SALES),
        "## 结论\n完成。",
        REVIEW_OK,
    ])
    events = []
    run = SQLAnalystAgent(llm, db).run("各品类的销售额是多少？", on_event=lambda k, p: events.append(k))
    assert run.steps[0].ok and run.steps[0].attempts == 2
    assert "no such column" in run.steps[0].errors[0]
    assert "schema_widen" in events and "表 ad_spend" in llm.calls[2][1]


def test_empty_result_triggers_fix(db):
    llm = FakeLLM([
        '{"steps": [{"id": 1, "goal": "钓竿销售额", "method": "筛选品类"}]}',
        sql_block("SELECT SUM(oi.unit_price * oi.quantity) AS 销售额 FROM order_items oi "
                  "JOIN products p ON oi.product_id = p.product_id WHERE p.category = '鱼竿' HAVING 销售额 > 0"),
        sql_block("SELECT ROUND(SUM(oi.unit_price * oi.quantity), 2) AS 销售额 FROM order_items oi "
                  "JOIN products p ON oi.product_id = p.product_id WHERE p.category = '钓竿'"),
        "## 结论\n完成。",
        REVIEW_OK,
    ])
    run = SQLAnalystAgent(llm, db).run("钓竿的销售额是多少？")
    assert "结果为空" in run.steps[0].errors[0]
    assert run.steps[0].ok and "销售额" in run.steps[0].result.metrics      # 一行结果的数字列也作为指标


def test_sql_agent_checkpoint_uses_db_fingerprint(db, db_path, tmp_path):
    assert task_id("sql", "q", db) == task_id("sql", "q", Database(db_path))
    other = build(tmp_path / "other.db", n_orders=50)
    assert task_id("sql", "q", db) != task_id("sql", "q", Database(other))


def test_extract_sql():
    assert extract_sql("好的\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


# ---------- 工具模式：模型自己查表结构、写 SQL ----------
def test_toolbox_db_tools_and_permissions(db):
    box = ToolBox(None, [], db=db)
    names = {s["function"]["name"] for s in box.schemas()}
    assert {"list_tables", "describe_table", "run_sql", "run_python"} <= names
    assert "preview_data" not in names and "load_skill" not in names
    assert "ROAS" in box.call("list_tables", {}).text
    assert not box.call("describe_table", {"table": "users"}).ok
    assert not box.call("run_python", {"code": "result['notes']='x'"}).ok      # 还没有查询结果
    assert not box.call("run_sql", {"sql": "DELETE FROM orders", "purpose": "删"}).ok
    read_only = ToolBox(None, [], db=db, allowed_levels={"read"})
    assert "权限不足" in read_only.call("run_sql", {"sql": "SELECT 1", "purpose": "x"}).text


def test_tool_agent_sql_then_python(db):
    llm = FakeLLM([
        {"tool_calls": [("list_tables", {})]},
        {"tool_calls": [("describe_table", {"table": "order_items"})]},
        {"tool_calls": [("run_sql", {"purpose": "各品类销售额", "sql": CAT_SALES})]},
        {"tool_calls": [("run_python", {"purpose": "算占比",
                                        "code": "result['metrics'] = {'品类数': float(len(df))}"})]},
        {"tool_calls": [("submit_report", {"report": "## 结论\n共 5 个品类。"})]},
        REVIEW_OK,
    ])
    run = ToolCallingAgent(llm, db=db).run(None, "各品类的销售额是多少？")
    assert run.mode == "sql-tools"
    assert [s.method for s in run.steps] == ["工具模式：run_sql", "工具模式：run_python"]
    assert run.steps[0].code.startswith("SELECT") and run.steps[1].result.metrics["品类数"] == 5.0
    assert run.tool_calls[2]["level"] == "只读查询"


def test_tool_agent_db_rejects_mcp(db):
    with pytest.raises(ValueError):
        ToolCallingAgent(FakeLLM([]), db=db, tool_source="mcp")


# ---------- 成本统计 ----------
def test_tracker_sums_by_stage_and_prices(monkeypatch):
    monkeypatch.setenv("PRICE_INPUT", "2")
    monkeypatch.setenv("PRICE_INPUT_CACHED", "0.5")
    monkeypatch.setenv("PRICE_OUTPUT", "8")
    monkeypatch.setenv("PRICE_CURRENCY", "CNY")
    t = usage.UsageTracker()
    with usage.track(t):
        with usage.stage("规划"):
            usage.record("m", 1000, 100, cached_tokens=400)
        with usage.stage("写代码"):
            usage.record("m", 2000, 300)
            usage.record("m", 1000, 100)
    s = t.summary()
    assert s["calls"] == 3 and s["prompt_tokens"] == 4000 and s["cached_tokens"] == 400
    assert s["cache_hit_rate"] == 0.1 and s["currency"] == "CNY"
    # 规划：600×2 + 400×0.5 + 100×8 = 2200；写代码：3000×2 + 400×8 = 9200（每百万 token）
    assert s["cost"] == pytest.approx(0.0114)
    assert [x["stage"] for x in s["by_stage"]] == ["规划", "写代码"]
    assert s["by_stage"][1]["calls"] == 2


def test_unknown_model_has_tokens_but_no_cost(monkeypatch):
    for k in ("PRICE_INPUT", "PRICE_INPUT_CACHED", "PRICE_OUTPUT"):
        monkeypatch.delenv(k, raising=False)
    t = usage.UsageTracker()
    with usage.track(t):
        usage.record("some-model", 10, 5)
    assert t.summary()["cost"] is None and t.summary()["prompt_tokens"] == 10


def test_record_response_reads_deepseek_and_openai_cache_fields():
    t = usage.UsageTracker()
    with usage.track(t):
        usage.record_response("m", SimpleNamespace(prompt_tokens=100, completion_tokens=10,
                                                   prompt_cache_hit_tokens=64), 0.5)
        usage.record_response("m", SimpleNamespace(prompt_tokens=50, completion_tokens=5,
                                                   prompt_tokens_details=SimpleNamespace(cached_tokens=32)), 0.5)
        usage.record_response("m", None, 0.1)                      # 接口没返回 usage：跳过
    assert [c.cached_tokens for c in t.calls] == [64, 32]


def test_record_without_tracker_is_noop():
    usage.record("m", 1, 1)          # 不报错、不记录
    assert usage.current() is None


def test_trackers_do_not_mix_between_threads():
    a, b = usage.UsageTracker(), usage.UsageTracker()

    def work(tr, n):
        with usage.track(tr):
            for _ in range(n):
                usage.record("m", 1, 1)

    threads = [threading.Thread(target=work, args=(a, 3)), threading.Thread(target=work, args=(b, 5))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(a.calls) == 3 and len(b.calls) == 5


def test_agent_run_reports_usage_by_stage():
    import pandas as pd
    df = pd.DataFrame({"品类": ["钓竿", "鱼线"], "销售额": [100.0, 20.0]})
    llm = FakeLLM([
        '{"steps": [{"id": 1, "goal": "总销售额", "method": "求和"}]}',
        "```python\nresult['metrics'] = {'总销售额': float(df['销售额'].sum())}\n```",
        "## 结论\n总销售额 120。",
        REVIEW_OK,
    ])
    run = DataAnalystAgent(llm, skills=[]).run(df, "总销售额是多少？")
    stages = [s["stage"] for s in run.usage["by_stage"]]
    assert stages == ["规划", "写代码", "写报告", "审查"] and run.usage["calls"] == 4


def test_sql_agent_usage_stages(db):
    llm = FakeLLM([
        '{"steps": [{"id": 1, "goal": "各品类销售额", "method": "求和"}]}',
        sql_block("SELECT x FROM nowhere"),
        sql_block(CAT_SALES),
        "## 结论\n完成。",
        REVIEW_OK,
    ])
    run = SQLAnalystAgent(llm, db).run("各品类的销售额是多少？")
    assert [s["stage"] for s in run.usage["by_stage"]] == ["规划", "写 SQL", "修 SQL", "写报告", "审查"]


# ---------- 第二轮调优 ----------
def test_verify_accepts_absolute_difference():
    """报告写“低 0.80”，代码算出的差值是 -0.8：方向由文字表达，数字算有出处。"""
    from analyst.validator import verify_report_numbers
    assert verify_report_numbers("店铺B 的 ROAS 比店铺A 低 0.80", [-0.8]) == []
    assert verify_report_numbers("比店铺A 低 0.95", [-0.8]) == ["0.95"]


def test_run_python_key_error_lists_columns(db):
    box = ToolBox(None, [], db=db)
    box.call("run_sql", {"purpose": "x", "sql": CAT_SALES})
    out = box.call("run_python", {"purpose": "y", "code": "result['metrics'] = {'a': float(df['acos'].sum())}"})
    assert not out.ok and "df 现有的列" in out.text and "品类" in out.text


def test_return_rate_metric_is_order_level(db):
    """退货率按订单（和知识库里公司的口径一致）；按明细行的另设“退货明细率”。"""
    assert "COUNT(DISTINCT CASE WHEN is_returned = 1" in db.metrics_text("钓竿的退货率是多少？")
    assert "AVG(order_items.is_returned)" in db.metrics_text("按明细算的退货明细率")


def test_eval_value_check_accepts_trailing_zeros():
    """“8.10%”就是 8.1%；但“68.18%”“8.15%”不是。"""
    import re
    from evals.run_eval import _value_pattern
    hit = lambda v, text: bool(re.search(_value_pattern(v), text))  # noqa: E731
    assert hit("8.1%", "较 2024 年的 8.10% 下降") and hit("6%", "目标 6.00%")
    assert not hit("8.1%", "68.18%") and not hit("8.1%", "8.15%") and not hit("6%", "7.16%")
