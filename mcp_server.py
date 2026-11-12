"""MCP 服务器启动入口：python mcp_server.py

由 Claude 桌面软件 / Claude Code 按配置自动启动（stdio 方式），一般不需要手动运行。配置方法见 README。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 无论从哪个目录启动都能找到 analyst 包

from analyst.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
