from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Local uvicorn runs from backend/, while Docker injects the same file via env_file.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    qwen_mode: str = os.getenv("QWEN_MODE", "local").lower()
    qwen_api_key: str = (
        os.getenv("QWEN_API_KEY")
        or os.getenv("CHENGQI_QWEN_API_KEY")
        or "not-needed-for-local-vllm"
    )
    qwen_base_url: str = os.getenv(
        "QWEN_BASE_URL", "http://172.21.25.98:8000/v1"
    )
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen3.6-27b")
    qwen_timeout_seconds: float = float(os.getenv("QWEN_TIMEOUT_SECONDS", "60"))
    qwen_max_tokens: int = int(os.getenv("QWEN_MAX_TOKENS", "512"))
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "voice_commerce")
    sbert_model: str = os.getenv(
        "SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    enable_sbert: bool = os.getenv("ENABLE_SBERT", "true").lower() in {"1", "true", "yes"}
    memory_min_score: float = float(os.getenv("MEMORY_MIN_SCORE", "0.10"))
    memory_limit_per_user: int = int(os.getenv("MEMORY_LIMIT_PER_USER", "100"))
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    piper_binary: str = os.getenv("PIPER_BINARY", "piper")
    piper_model: str = os.getenv("PIPER_MODEL", "")
    livetalking_url: str = os.getenv("LIVETALKING_URL", "http://localhost:8010")


settings = Settings()
