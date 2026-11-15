"""后端接口测试：用 FastAPI 的 TestClient 模拟前端发请求，用 FakeLLM 代替真实模型。

运行：.venv\\Scripts\\python -m pytest -q tests/test_api.py
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst.checkpoint import CheckpointStore  # noqa: E402
from analyst.llm import FakeLLM  # noqa: E402
from analyst.memory import PreferenceStore  # noqa: E402
from api.main import create_app  # noqa: E402

PLAN = '{"steps": [{"id": 1, "goal": "各品类销售额", "method": "分组求和"}]}'
CODE = ("```python\ng = df.groupby('品类')['销售额'].sum()\n"
        "result['metrics'] = {'钓竿销售额': float(g['钓竿'])}\n"
        "result['tables'] = {'品类销售额': g.to_frame()}\nresult['notes'] = '按品类汇总'\n```")


def parse_sse(text: str) -> list[tuple[str, object]]:
    """把 SSE 文本拆成 [(事件名, 数据)]。"""
    out = []
    for block in text.strip().split("\n\n"):
        kind = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if kind:
            out.append((kind, data))
    return out


@pytest.fixture
def make_client(tmp_path):
    def _make(replies):
        llm = FakeLLM(list(replies))
        app = create_app(llm_factory=lambda: llm, prefs=PreferenceStore(tmp_path / "prefs.json"),
                         checkpoints=CheckpointStore(tmp_path / "ckpt"), serve_frontend=False)
        return TestClient(app), llm
    return _make


def new_session_with_data(client) -> str:
    sid = client.post("/api/sessions").json()["session_id"]
    info = client.post(f"/api/sessions/{sid}/sample").json()
    assert info["data"]["rows"] == 2030 and info["data"]["cols"] == 11
    return sid


def test_config_does_not_leak_key(make_client):
    client, _ = make_client([])
    cfg = client.get("/api/config").json()
    assert set(cfg) == {"model", "ready", "reviewer_model"}     # 只返回模型名和是否就绪，不返回密钥


def test_ask_streams_progress_then_result(make_client):
    client, _ = make_client([PLAN, CODE, "## 结论\n钓竿销售额最高。", '{"issues": []}'])
    sid = new_session_with_data(client)
    resp = client.post(f"/api/sessions/{sid}/ask", json={"question": "各品类销售额？", "mode": "plan"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and "result" in kinds and "progress" in kinds
    texts = [d["text"] for k, d in events if k == "progress"]
    assert any("分析计划" in t for t in texts) and any("核对报告中的数字" in t for t in texts)
    result = next(d for k, d in events if k == "result")
    assert result["report"].startswith("## 结论") and result["review"]["passed"]
    step = result["steps"][0]
    assert step["ok"] and step["tables"][0]["columns"] == ["品类", "销售额"]
    # 下载报告和日志
    rid = result["run_id"]
    assert "电商经营分析报告" in client.get(f"/api/runs/{rid}/report.html").text
    assert json.loads(client.get(f"/api/runs/{rid}/log.json").text)["question"] == "各品类销售额？"


def test_ask_requires_data_and_valid_mode(make_client):
    client, _ = make_client([])
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/ask", json={"question": "x"}).status_code == 400
    sid = new_session_with_data(client)
    bad = client.post(f"/api/sessions/{sid}/ask", json={"question": "x", "mode": "hack"})
    assert bad.status_code == 422                                  # 参数不合法，FastAPI 自动拒绝
    assert client.post("/api/sessions/nope/ask", json={"question": "x"}).status_code == 404


def test_preference_only_message_saves_without_analysis(make_client):
    reply = '{"preferences": [{"topic": "退货率口径", "content": "按金额计算"}], "has_question": false}'
    client, llm = make_client([reply])
    sid = new_session_with_data(client)
    events = parse_sse(client.post(f"/api/sessions/{sid}/ask",
                                   json={"question": "记住，以后退货率都按金额算"}).text)
    kinds = [k for k, _ in events]
    assert "preferences_saved" in kinds and "result" not in kinds   # 只记偏好，不启动分析
    assert client.get("/api/preferences").json()[0]["topic"] == "退货率口径"
    assert client.delete("/api/preferences/退货率口径").json() == []


def test_upload_csv_and_reject_other_types(make_client):
    client, _ = make_client([])
    sid = client.post("/api/sessions").json()["session_id"]
    csv = pd.DataFrame({"品类": ["钓竿", "鱼线"], "销售额": [1.0, 2.0]}).to_csv(index=False).encode("utf-8")
    info = client.post(f"/api/sessions/{sid}/upload",
                       files={"file": ("orders.csv", io.BytesIO(csv), "text/csv")}).json()
    assert info["data_name"] == "orders.csv" and info["data"]["rows"] == 2
    bad = client.post(f"/api/sessions/{sid}/upload", files={"file": ("a.exe", io.BytesIO(b"x"), "x")})
    assert bad.status_code == 400


def test_error_is_reported_and_marked_resumable(make_client):
    client, _ = make_client([PLAN])                     # 规划之后模型“断网”
    sid = new_session_with_data(client)
    events = parse_sse(client.post(f"/api/sessions/{sid}/ask", json={"question": "各品类销售额？"}).text)
    err = next(d for k, d in events if k == "error")
    assert err["resumable"] and events[-1][0] == "done"


def test_memory_summary_after_many_turns(make_client):
    replies = []
    for i in range(4):
        replies += [PLAN, CODE, f"## 结论\n第{i + 1}轮结论。", '{"issues": []}']
        if i == 3:
            replies.append("- 第1轮：钓竿销售额最高")          # 第 4 轮结束时压缩第 1 轮
    client, _ = make_client(replies)
    sid = new_session_with_data(client)
    for i in range(4):
        events = parse_sse(client.post(f"/api/sessions/{sid}/ask", json={"question": f"问题{i + 1}"}).text)
        result = next(d for k, d in events if k == "result")
    assert result["memory_compressed"] and "钓竿销售额最高" in result["memory_summary"]
    assert client.get(f"/api/sessions/{sid}").json()["compressed_turns"] == 1


# ---------- 数据库模式（Text-to-SQL）和成本统计 ----------
SQL_PLAN = '{"steps": [{"id": 1, "goal": "各品类销售额", "method": "明细关联商品"}]}'
SQL_CODE = ("```sql\nSELECT p.category AS 品类, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS 销售额 "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category\n```")


@pytest.fixture
def sample_db(tmp_path, monkeypatch):
    import api.main as main
    from make_sample_db import build
    monkeypatch.setattr(main, "SAMPLE_DB", build(tmp_path / "shop.db", n_orders=200))


def test_sample_db_and_sql_plan_mode(make_client, sample_db):
    client, _ = make_client([SQL_PLAN, SQL_CODE, "## 结论\n钓竿销售额最高。", '{"issues": []}'])
    sid = client.post("/api/sessions").json()["session_id"]
    info = client.post(f"/api/sessions/{sid}/sample_db").json()
    assert info["data"] is None and {t["name"] for t in info["database"]["tables"]} >= {"orders", "order_items"}
    assert any(m["name"] == "ROAS" for m in info["database"]["metrics"])
    events = parse_sse(client.post(f"/api/sessions/{sid}/ask",
                                   json={"question": "各品类的销售额是多少？", "mode": "plan"}).text)
    assert any("选出相关的表" in d["text"] for k, d in events if k == "progress")
    result = next(d for k, d in events if k == "result")
    assert result["mode"] == "sql" and {t["table"] for t in result["sql"]} == {"order_items", "products"}
    assert result["steps"][0]["code"].startswith("SELECT")
    # 成本：规划、写 SQL、写报告、审查各 1 次（没有“记住”等词不识别偏好；第 1 轮不压缩记忆）
    assert result["usage"]["calls"] == 4 and result["usage"]["prompt_tokens"] > 0
    rid = result["run_id"]
    assert "模型调用 4 次" in client.get(f"/api/runs/{rid}/report.html").text
    assert json.loads(client.get(f"/api/runs/{rid}/log.json").text)["usage"]["calls"] == 4


def test_db_mode_rejects_mcp(make_client, sample_db):
    client, _ = make_client([])
    sid = client.post("/api/sessions").json()["session_id"]
    client.post(f"/api/sessions/{sid}/sample_db")
    assert client.post(f"/api/sessions/{sid}/ask", json={"question": "x", "mode": "mcp"}).status_code == 400


def test_upload_sqlite_and_reject_fake_db(make_client, tmp_path, monkeypatch):
    import api.main as main
    from make_sample_db import build
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    client, _ = make_client([])
    sid = client.post("/api/sessions").json()["session_id"]
    content = build(tmp_path / "up.db", n_orders=50).read_bytes()
    info = client.post(f"/api/sessions/{sid}/upload",
                       files={"file": ("shop.db", io.BytesIO(content), "application/octet-stream")}).json()
    assert info["data_name"] == "shop.db" and len(info["database"]["tables"]) == 5
    bad = client.post(f"/api/sessions/{sid}/upload", files={"file": ("x.db", io.BytesIO(b"not sqlite"), "x")})
    assert bad.status_code == 400
