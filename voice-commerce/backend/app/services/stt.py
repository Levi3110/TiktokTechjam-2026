from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Lock

from app.config import settings


class WhisperService:
    def __init__(self) -> None:
        self._model = None
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Chưa cài faster-whisper. Hãy cài requirements-ai.txt.") from exc
            with self._lock:
                if self._model is None:
                    self._model = WhisperModel(
                        settings.whisper_model,
                        device=settings.whisper_device,
                        compute_type=settings.whisper_compute_type,
                    )
        return self._model

    def transcribe(self, content: bytes, suffix: str = ".webm") -> str:
        model = self._load()
        with tempfile.NamedTemporaryFile(suffix=suffix) as audio:
            audio.write(content)
            audio.flush()
            segments, _ = model.transcribe(audio.name, language="vi", vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()


whisper_service = WhisperService()

