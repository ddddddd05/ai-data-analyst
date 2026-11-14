// 左侧设置栏：模型信息、工作模式、独立审查、对话摘要（短期记忆）、长期偏好
import { useState } from "react";
import { api } from "../api.js";

export default function Sidebar({ config, settings, setSettings, busy, onClear, canClear, info, prefs, setPrefs }) {
  const set = (key, value) => setSettings((s) => ({ ...s, [key]: value }));
  const [topic, setTopic] = useState("");
  const [content, setContent] = useState("");

  async function addPref(e) {
    e.preventDefault();
    if (!topic.trim() || !content.trim()) return;
    setPrefs(await api.addPreference(topic, content));
    setTopic("");
    setContent("");
  }

  return (
    <aside className="sidebar">
      <h2>模型</h2>
      <p>{config ? config.model : "读取中…"} {config && (config.ready ? "✅" : "⚠️ 未配置密钥")}</p>
      <p className="muted small">密钥只保存在后端的 .env 文件里，不会发到浏览器。</p>

      <h2>工作模式</h2>
      <div className="radio">
        <label><input type="radio" checked={settings.mode === "plan"} disabled={busy}
          onChange={() => set("mode", "plan")} /> 规划-执行</label>
        <label><input type="radio" checked={settings.mode === "tools"} disabled={busy}
          onChange={() => set("mode", "tools")} /> 工具调用</label>
      </div>
      <p className="muted small">
        {settings.mode === "plan"
          ? "程序先让模型列计划，再一步步执行：流程固定、稳定。"
          : "模型每一轮自己决定调用哪个工具：更灵活。"}
      </p>

      {settings.mode === "plan" ? (
        <label className="field">每步最多重试 {settings.maxRetries} 次
          <input type="range" min="0" max="3" value={settings.maxRetries} disabled={busy}
            onChange={(e) => set("maxRetries", Number(e.target.value))} />
        </label>
      ) : (
        <>
          <label className="field">工具调用上限 {settings.maxToolCalls} 次
            <input type="range" min="5" max="25" value={settings.maxToolCalls} disabled={busy}
              onChange={(e) => set("maxToolCalls", Number(e.target.value))} />
          </label>
          <div className="radio">
            <span className="small">工具来源：</span>
            <label><input type="radio" checked={settings.toolSource === "builtin"} disabled={busy}
              onChange={() => set("toolSource", "builtin")} /> 内置</label>
            <label><input type="radio" checked={settings.toolSource === "mcp"} disabled={busy}
              onChange={() => set("toolSource", "mcp")} /> MCP 服务器</label>
          </div>
          <p className="muted small">工具权限：只读工具直接执行；运行代码先过安全检查再在独立进程执行；读写文件、联网的工具不提供。</p>
        </>
      )}

      <label className="check">
        <input type="checkbox" checked={settings.review} disabled={busy}
          onChange={(e) => set("review", e.target.checked)} /> 独立审查报告
      </label>
      <label className="check">
        <input type="checkbox" checked={settings.knowledge} disabled={busy}
          onChange={(e) => set("knowledge", e.target.checked)} /> 检索业务知识库（RAG）
      </label>
      <button onClick={onClear} disabled={!canClear || busy}>清空对话</button>

      {info?.memory_summary && (
        <details className="memory">
          <summary>🗜️ 对话摘要（已压缩 {info.compressed_turns} 轮）</summary>
          <p className="muted small">最近 3 轮保留原文，更早的对话压缩成下面的摘要交给模型：</p>
          <pre>{info.memory_summary}</pre>
        </details>
      )}

      <h2>长期记忆（偏好）</h2>
      <p className="muted small">在对话里说“记住，以后退货率都按金额算”就会记下，也可以在这里手动添加。</p>
      {prefs.length === 0 && <p className="muted small">（暂无）</p>}
      <ul className="prefs">
        {prefs.map((p) => (
          <li key={p.topic}>
            <span>📌 <b>{p.topic}</b>：{p.content}</span>
            <button className="icon" title="删除这条偏好"
              onClick={async () => setPrefs(await api.deletePreference(p.topic))}>✕</button>
          </li>
        ))}
      </ul>
      <form className="pref-form" onSubmit={addPref}>
        <input placeholder="主题，如：退货率口径" value={topic} onChange={(e) => setTopic(e.target.value)} />
        <input placeholder="内容，如：按金额计算" value={content} onChange={(e) => setContent(e.target.value)} />
        <button type="submit" disabled={!topic.trim() || !content.trim()}>添加偏好</button>
      </form>
    </aside>
  );
}
