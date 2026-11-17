# 两阶段构建：第一阶段用 Node 打包前端，第二阶段只带 Python 运行环境和打包好的前端，镜像更小。
# 构建：docker build -t ecommerce-analyst .
# 运行：docker compose up（读取 .env 里的 LLM_API_KEY，打开 http://localhost:8000）

# ---------- 阶段 1：打包前端 ----------
FROM node:24-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- 阶段 2：Python 后端 ----------
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# 先只复制依赖清单再安装：代码改动时这一层可以用缓存，不用重新装依赖
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY analyst/ analyst/
COPY api/ api/
COPY skills/ skills/
COPY knowledge/ knowledge/
COPY sample_data/ sample_data/
COPY mcp_server.py mcp_servers.json make_sample_data.py make_sample_db.py ./
COPY --from=frontend /app/frontend/dist frontend/dist
# 不用 root 运行：即使代码被攻破，权限也有限
RUN useradd --create-home appuser && mkdir -p outputs memory && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/config')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
