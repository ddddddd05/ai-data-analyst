"""读取大模型接口配置。

密钥只放在本地 .env 文件或环境变量里，不写进代码、不上传 GitHub。
任何兼容 OpenAI 接口格式的模型都能用（Gemini、DeepSeek、通义千问等），
只需要改 LLM_BASE_URL 和 LLM_MODEL。
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini
DEFAULT_MODEL = "gemini-3.8-flash"


def load_env(path: Path = ROOT / ".env") -> None:
    """读取 .env 文件里的 KEY=VALUE（已存在的环境变量优先）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0]  # 去掉行尾注释
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    use_proxy: bool = True  # 国内模型（如 DeepSeek）可设为 False，直连不走系统代理

    @property
    def ready(self) -> bool:
        return bool(self.api_key)


def get_llm_config() -> LLMConfig:
    load_env()
    return LLMConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        use_proxy=os.getenv("LLM_USE_PROXY", "true").strip().lower() not in ("false", "0", "no"),
    )
