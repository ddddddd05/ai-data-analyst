"""网页界面（对话式）：上传表格后像聊天一样提问，可以连续追问。

每个回答都包含：分析报告、加载的 Skill、数字校验结果、每一步的代码和结果、下载按钮。
追问时会把前几轮的“问题 + 结论”交给 Agent，让它理解“那”“它”等指代。

启动：.venv\\Scripts\\streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from analyst.agent import DataAnalystAgent, RunResult, StepRecord, summarize_for_history
from analyst.config import LLMConfig, get_llm_config, get_reviewer_config
from analyst.llm import OpenAICompatLLM
from analyst.profiler import load_table, profile
from analyst.report import to_html, to_log_json
from analyst.tool_agent import ToolCallingAgent

SAMPLE = Path(__file__).resolve().parent / "sample_data" / "模拟电商订单.csv"
EXAMPLES = [
    "各品类的销售额和退货率分别是多少？哪个品类最值得重点投入？",
    "每个月的销售额趋势如何？有没有明显的旺季？",
    "哪个店铺的广告投入产出比最高？",
    "评分低的订单有什么共同特点？",
]

st.set_page_config(page_title="电商经营分析 Agent", page_icon="📊", layout="wide")
ss = st.session_state
ss.setdefault("messages", [])   # [{"role": "user"/"assistant", "content": 问题文字 或 RunResult 或 错误文字}]

# ---------- 侧边栏：模型设置 + 对话管理 ----------
cfg = get_llm_config()
with st.sidebar:
    st.header("模型设置")
    key_input = st.text_input("API Key", type="password", value="",
                              help="只保存在本次浏览器会话里。也可以写在项目根目录的 .env 文件中。")
    base_url = st.text_input("接口地址", value=cfg.base_url)
    model = st.text_input("模型名称", value=cfg.model)
    api_key = key_input or cfg.api_key
    st.write("✅ 已配置密钥" if api_key else "⚠️ 还没有配置 API Key")
    st.divider()
    mode = st.radio("工作模式", ["规划-执行", "工具调用"], horizontal=True,
                    help="规划-执行：程序先让模型列计划，再一步步执行，流程固定、稳定。\n\n"
                         "工具调用：模型每一轮自己决定调用哪个工具（看某列、预览数据、加载 Skill、运行代码、提交报告），更灵活。")
    if mode == "规划-执行":
        max_retries = st.slider("每步最多重试次数", 0, 3, 2)
    else:
        max_tool_calls = st.slider("工具调用次数上限", 5, 25, 15, help="防止模型反复调用工具、陷入死循环")
        st.caption("工具权限：只读工具直接执行；运行代码先过安全检查再在独立进程执行；"
                   "读写文件、联网的工具不提供。")
    use_review = st.checkbox("独立审查报告", value=True,
                             help="报告写完后，另起一次模型调用，专门检查结论和计算结果是否矛盾；"
                                  "发现严重矛盾时让 AI 改一次报告。")
    if st.button("清空对话", disabled=not ss["messages"]):
        ss["messages"] = []
        st.rerun()
    st.caption("发送给模型的只有数据概况、前 5 行样例和之前几轮的结论，不会上传整张表。")

# ---------- 数据 ----------
st.title("📊 电商经营分析 Agent")
st.caption("给电商运营用：上传订单表，用中文提问，自动清洗数据、按统一口径算指标、写出带数字出处的分析报告。")
col_up, col_sample = st.columns([3, 1])
uploaded = col_up.file_uploader("上传数据文件", type=["csv", "xlsx", "xls"])
if col_sample.button("使用示例数据", help="模拟的跨境电商渔具订单，含故意制造的数据问题"):
    ss["df"], ss["data_name"], ss["messages"] = load_table(SAMPLE), SAMPLE.name, []
if uploaded is not None and ss.get("data_name") != uploaded.name:
    try:
        ss["df"], ss["data_name"], ss["messages"] = load_table(uploaded, uploaded.name), uploaded.name, []
    except Exception as e:  # noqa: BLE001
        st.error(f"读取失败：{e}")

df: pd.DataFrame | None = ss.get("df")
if df is None:
    st.info("请上传文件，或点击“使用示例数据”。")
    st.stop()

p = profile(df)
with st.expander(f"数据：{ss.get('data_name', '')}　|　{p['rows']} 行 × {p['cols']} 列，"
                 f"重复 {p['duplicate_rows']} 行，"
                 f"{sum(1 for c in p['columns'] if c['missing'])} 列有缺失（点击查看）"):
    st.dataframe(df.head(50), width="stretch")
    overview = pd.DataFrame(p["columns"]).drop(columns=["top"], errors="ignore")
    st.dataframe(overview.astype(str).replace("nan", ""), width="stretch")


# ---------- 展示一条回答 ----------
def show_step(rec: StepRecord) -> None:
    icon = "✅" if rec.ok else "❌"
    with st.expander(f"{icon} 第{rec.id}步：{rec.goal}（使用 {rec.data_rows} 行，尝试 {rec.attempts} 次）"):
        st.caption(f"方法：{rec.method}")
        for i, err in enumerate(rec.errors, 1):
            st.warning(f"第 {i} 次失败，已把错误交给 AI 修改：\n\n{err[-600:]}")
        r = rec.result
        if r and r.ok:
            if r.notes:
                st.write(r.notes)
            if r.metrics:
                st.json(r.metrics)
            for name, t in r.tables.items():
                st.write(f"**{name}**")
                st.dataframe(t, width="stretch")
            for img in r.figures:
                st.image(img)
        for w in rec.warnings:
            st.info(w)
        st.code(rec.code, language="python")


def show_tool_calls(run: RunResult) -> None:
    with st.expander(f"查看工具调用记录（{len(run.tool_calls)} 次）"):
        for t in run.tool_calls:
            args = t["args"].get("purpose") or t["args"].get("name") or t["args"].get("column") or ""
            st.write(f"{'✅' if t['ok'] else '❌'} {t['n']}. `{t['name']}`（{t['level']}）{args}")
            if not t["ok"]:
                st.caption(t["result"][:300])


def show_review(run: RunResult) -> None:
    r = run.review
    if r is None:
        return
    if r.error:
        st.caption(f"🧑‍⚖️ 独立审查未完成（不影响报告）：{r.error}")
    elif r.revised:
        st.warning("🧑‍⚖️ 独立审查发现结论与计算结果矛盾，报告已按审查意见修改一次：\n\n"
                   + "\n".join(f"- {i['claim']}（结果显示：{i['evidence']}）" for i in r.serious))
    elif r.issues:
        st.info("🧑‍⚖️ 独立审查：没有严重矛盾，有几点表述提醒：\n\n"
                + "\n".join(f"- {i['claim']}（{i['evidence']}）" for i in r.issues))
    else:
        st.success("🧑‍⚖️ 独立审查通过：结论与计算结果没有发现矛盾。")


def show_answer(run: RunResult, idx: int) -> None:
    if run.skills:
        st.caption("🧩 加载的 Skill：" + "；".join(f"{n}（{w}）" for n, w in run.skills))
    if run.unverified:
        st.warning("⚠ 自动校验：报告中这些数字没有在计算结果里找到出处，请人工核对："
                   + "、".join(run.unverified))
    else:
        st.success("自动校验通过：报告中的数字都能在计算结果中找到出处。")
    show_review(run)
    st.markdown(run.report)
    if run.mode == "tools":
        show_tool_calls(run)
    with st.expander(f"查看分析过程（{len(run.steps)} 步，用时 {run.seconds} 秒）"):
        for rec in run.steps:
            show_step(rec)
    c1, c2 = st.columns(2)
    c1.download_button("下载 HTML 报告", to_html(run), file_name=f"分析报告_{idx}.html",
                       mime="text/html", key=f"html_{idx}")
    c2.download_button("下载运行日志（JSON）", to_log_json(run), file_name=f"运行日志_{idx}.json",
                       mime="application/json", key=f"log_{idx}")


# ---------- 对话记录 ----------
for i, msg in enumerate(ss["messages"]):
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], RunResult):
            show_answer(msg["content"], i)
        elif msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])

# 还没开始对话时，给几个示例问题
clicked = None
if not ss["messages"]:
    st.caption("试试这些问题：")
    cols = st.columns(len(EXAMPLES))
    for col, q in zip(cols, EXAMPLES):
        if col.button(q, disabled=not api_key, width="stretch"):
            clicked = q

typed = st.chat_input("输入问题，可以连续追问，例如“那钓竿为什么退货率高？”" if api_key
                      else "请先在左侧配置 API Key", disabled=not api_key)
question = (typed or clicked or "").strip()

# ---------- 处理新问题 ----------
if question:
    # 之前几轮的“问题 + 结论摘要”作为对话历史
    history = []
    for a, b in zip(ss["messages"][::2], ss["messages"][1::2]):
        if isinstance(b["content"], RunResult):
            history.append((a["content"], summarize_for_history(b["content"])))

    ss["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        llm = OpenAICompatLLM(LLMConfig(api_key, base_url, model, use_proxy=cfg.use_proxy))
        rcfg = get_reviewer_config(LLMConfig(api_key, base_url, model, use_proxy=cfg.use_proxy))
        reviewer = OpenAICompatLLM(rcfg) if rcfg else None
        if mode == "工具调用":
            agent = ToolCallingAgent(llm, max_tool_calls=max_tool_calls,
                                     reviewer=reviewer, review=use_review)
        else:
            agent = DataAnalystAgent(llm, max_retries=max_retries,
                                     reviewer=reviewer, review=use_review)
        with st.status("Agent 工作中……", expanded=True) as status:
            def on_event(kind, payload):
                if kind == "profile":
                    st.write("🔍 已读取数据概况")
                elif kind == "skills":
                    for name, why in payload or ([("（无）", "数据干净，不需要专项规则")] if mode == "规划-执行" else []):
                        st.write(f"🧩 加载 Skill：{name}（原因：{why}）")
                elif kind == "plan_trimmed":
                    st.write(f"✂️ 去掉汇总类步骤“{payload}”，改由最后的报告完成")
                elif kind == "plan":
                    st.write("🗺️ 分析计划：")
                    for s in payload:
                        st.write(f"　{s.get('id')}. {s.get('goal')}")
                elif kind == "step_start":
                    st.write(f"⚙️ 正在执行第{payload.id}步：{payload.goal}")
                elif kind == "step_error":
                    st.write(f"　↻ 第{payload['step']}步第{payload['attempt']}次出错，正在让 AI 修正")
                elif kind == "cleaned":
                    st.write(f"🧹 第{payload['step']}步清洗完成，后续步骤统一使用 {payload['rows']} 行数据")
                elif kind == "tool_call":
                    detail = payload["args"].get("purpose") or payload["args"].get("name") or payload["args"].get("column") or ""
                    st.write(f"🛠️ 第{payload['n']}次调用工具 `{payload['name']}`（{payload['level']}）{detail}")
                elif kind == "tool_result" and not payload["ok"]:
                    st.write(f"　↻ `{payload['name']}` 没成功：{payload['text'][:120]}")
                elif kind == "tool_limit":
                    st.write(f"⛔ 工具调用达到上限 {payload} 次，强制写报告")
                elif kind == "review_start":
                    st.write("🧑‍⚖️ 独立审查员正在检查结论是否与计算结果矛盾")
                elif kind == "review" and payload.revised:
                    st.write("　↻ 审查发现严重矛盾，已让 AI 修改报告")
                elif kind == "verify":
                    st.write("🔎 正在核对报告中的数字")
            try:
                run = agent.run(df, question, on_event=on_event, history=history)
                status.update(label=f"完成，用时 {run.seconds} 秒", state="complete", expanded=False)
                ss["messages"].append({"role": "assistant", "content": run})
            except Exception as e:  # noqa: BLE001
                status.update(label="出错了", state="error")
                ss["messages"].append({"role": "assistant", "error": True,
                                       "content": f"调用模型失败：{e}\n\n请检查 API Key、接口地址、模型名称和网络。"})
    st.rerun()
