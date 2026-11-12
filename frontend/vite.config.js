import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发时：前端跑在 5173 端口，凡是 /api 开头的请求，转发给 8000 端口的后端（FastAPI）
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
