// 主页面：左边设置栏，右边数据 + 对话。所有“状态”（数据、对话记录、设置）都放在这里，传给子组件。
import { useEffect, useRef, useState } from "react";
import { api, ask } from "./api.js";
import Sidebar from "./components/Sidebar.jsx";
import DataPanel from "./components/DataPanel.jsx";
import Answer from "./components/Answer.jsx";

const EXAMPLES = [
  "各品类的销售额和退货率分别是多少？哪个品类最值得重点投入？",
  "每个月的销售额趋势如何？有没有明显的旺季？",
  "哪个店铺的广告投入产出比最高？",
  "评分低的订单有什么共同特点？",
];

export default function App() {
  const [config, setConfig] = useState(null);       // 后端配置：模型名、是否配置了密钥
  const [sid, setSid] = useState(null);             // 当前对话编号
  const [info, setInfo] = useState(null);           // 数据概况 + 对话摘要
  const [messages, setMessages] = useState([]);     // 对话记录：{role, text} 或 {role, result} 或 {role, error}
  const [progress, setProgress] = useState([]);     // 正在分析时的进度
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [prefs, setPrefs] = useState([]);
  const [settings, setSettings] = useState({
    mode: "plan", maxRetries: 2, maxToolCalls: 15, toolSource: "builtin", review: true, knowledge: true,
  });
  const [notice, setNotice] = useState("");
  const bottomRef = useRef(null);

  // 页面打开时：读配置、建对话、读偏好
  useEffect(() => {
    api.config().then(setConfig).catch((e) => setNotice(e.message));
    api.newSession().then((r) => setSid(r.session_id)).catch((e) => setNotice(e.message));
    api.preferences().then(setPrefs).catch(() => {});
  }, []);

  // 有新消息或新进度时，滚到最下面
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, progress]);

  async function loadData(promise) {
    try {
      setNotice("");
      setInfo(await promise);
      setMessages([]);
    } catch (e) {
      setNotice(e.message);
    }
  }

  async function send(question) {
    const q = question.trim();
    if (!q || busy || !sid) return;
    setInput("");
    setBusy(true);
    setProgress([]);
    setMessages((m) => [...m, { role: "user", text: q }]);
    const mode = settings.mode === "plan" ? "plan" : settings.toolSource === "mcp" ? "mcp" : "tools";
    try {
      await ask(
        sid,
        { question: q, mode, max_retries: settings.maxRetries, max_tool_calls: settings.maxToolCalls,
          review: settings.review, knowledge: settings.knowledge },
        (kind, data) => {
          if (kind === "progress") setProgress((p) => [...p, data.text]);
          if (kind === "preferences_saved") {
            api.preferences().then(setPrefs);
            const text = "📌 已记住：" + data.items.map((x) => `${x.topic}——${x.content}`).join("；");
            if (!data.analysis) setMessages((m) => [...m, { role: "assistant", text: text + "。之后的分析都会按这个来。" }]);
            else setProgress((p) => [...p, text]);
          }
          if (kind === "result") {
            setMessages((m) => [...m, { role: "assistant", result: data }]);
            setInfo((old) => old && { ...old, memory_summary: data.memory_summary,
              compressed_turns: old.compressed_turns + (data.memory_compressed ? 1 : 0) });
          }
          if (kind === "error") setMessages((m) => [...m, { role: "assistant", error: data.message, retry: q }]);
        }
      );
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", error: e.message, retry: q }]);
    } finally {
      setBusy(false);
      setProgress([]);
    }
  }

  // 失败后“从断点继续”：去掉失败的那一问一答，用同一个问题重新提交（后端会找到存档）
  function retry(question) {
    setMessages((m) => m.slice(0, -2));
    send(question);
  }

  async function clearChat() {
    setMessages([]);
    setInfo(await api.clear(sid));
  }

  const ready = config?.ready && info?.data && !busy;

  return (
    <div className="layout">
      <Sidebar
        config={config} settings={settings} setSettings={setSettings} busy={busy}
        onClear={clearChat} canClear={messages.length > 0} info={info}
        prefs={prefs} setPrefs={setPrefs}
      />
      <main className="main">
        <header>
          <h1>📊 电商经营分析 Agent</h1>
          <p className="muted">给电商运营用：上传订单表，用中文提问，自动清洗数据、按统一口径算指标、写出带数字出处的分析报告。</p>
        </header>

        <DataPanel
          info={info} busy={busy}
          onSample={() => loadData(api.useSample(sid))}
          onUpload={(file) => loadData(api.upload(sid, file))}
        />
        {notice && <div className="banner error">{notice}</div>}
        {config && !config.ready && (
          <div className="banner warn">后端没有配置 API Key：请在项目根目录的 .env 里填写 LLM_API_KEY，然后重启后端。</div>
        )}

        <section className="chat">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="avatar">{m.role === "user" ? "🙋" : "🤖"}</div>
              <div className="bubble">
                {m.text && <p>{m.text}</p>}
                {m.result && <Answer result={m.result} />}
                {m.error && (
                  <>
                    <div className="banner error">{m.error}</div>
                    <p className="muted">已完成的步骤已经存档，检查网络或密钥后可以从断点继续，不用从头再来。</p>
                    {i === messages.length - 1 && (
                      <button className="primary" disabled={busy} onClick={() => retry(m.retry)}>
                        ↻ 从断点继续
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}

          {busy && (
            <div className="msg assistant">
              <div className="avatar">🤖</div>
              <div className="bubble progress">
                <p className="muted">Agent 工作中……</p>
                <ul>{progress.map((t, i) => <li key={i}>{t}</li>)}</ul>
              </div>
            </div>
          )}

          {info?.data && messages.length === 0 && !busy && (
            <div className="examples">
              <p className="muted">试试这些问题：</p>
              {EXAMPLES.map((q) => (
                <button key={q} disabled={!ready} onClick={() => send(q)}>{q}</button>
              ))}
            </div>
          )}
          <div ref={bottomRef} />
        </section>

        <form className="ask" onSubmit={(e) => { e.preventDefault(); send(input); }}>
          <input
            value={input} onChange={(e) => setInput(e.target.value)} disabled={!ready}
            placeholder={info?.data ? "输入问题，可以连续追问，例如“那钓竿为什么退货率高？”" : "请先上传数据或使用示例数据"}
          />
          <button className="primary" type="submit" disabled={!ready || !input.trim()}>发送</button>
        </form>
      </main>
    </div>
  );
}
