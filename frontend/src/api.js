// 和后端（FastAPI）打交道的函数都放在这里。组件只管显示，不直接写 fetch。

async function request(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = `请求失败（${res.status}）`;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* 响应不是 JSON，就用默认提示 */
    }
    throw new Error(message);
  }
  return res.json();
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  config: () => request("/api/config"),
  newSession: () => request("/api/sessions", { method: "POST" }),
  sessionInfo: (sid) => request(`/api/sessions/${sid}`),
  useSample: (sid) => request(`/api/sessions/${sid}/sample`, { method: "POST" }),
  upload: (sid, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/sessions/${sid}/upload`, { method: "POST", body: form });
  },
  clear: (sid) => request(`/api/sessions/${sid}/clear`, { method: "POST" }),
  preferences: () => request("/api/preferences"),
  addPreference: (topic, content) => request("/api/preferences", json({ topic, content })),
  deletePreference: (topic) =>
    request(`/api/preferences/${encodeURIComponent(topic)}`, { method: "DELETE" }),
};

/**
 * 提问。后端用 SSE（服务器推送事件）一条条推送进度，最后推送结果。
 * 普通的 EventSource 只支持 GET，这里要 POST 问题，所以用 fetch 读取“流”，自己按 SSE 格式拆开：
 *   event: progress
 *   data: {"text": "..."}
 *   （空行分隔下一条）
 * onEvent(事件名, 数据) 每收到一条就调用一次。
 */
export async function ask(sid, body, onEvent) {
  const res = await fetch(`/api/sessions/${sid}/ask`, json(body));
  if (!res.ok) {
    let message = `请求失败（${res.status}）`;
    try {
      message = (await res.json()).detail || message;
    } catch {
      /* 忽略 */
    }
    throw new Error(message);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let kind = "message";
      let data = null;
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) kind = line.slice(7);
        else if (line.startsWith("data: ")) data = JSON.parse(line.slice(6));
      }
      onEvent(kind, data);
    }
  }
}
