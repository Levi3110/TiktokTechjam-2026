from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from app.config import settings


class PiperService:
    def synthesize(self, text: str) -> bytes:
        if not settings.piper_model:
            raise RuntimeError("Chưa cấu hình PIPER_MODEL; frontend sẽ dùng browser voice.")
        model = Path(settings.piper_model)
        if not model.is_file():
            raise RuntimeError(f"Không tìm thấy Piper model: {model}")
        with tempfile.NamedTemporaryFile(suffix=".wav") as output:
            try:
                subprocess.run(
                    [settings.piper_binary, "--model", str(model), "--output_file", output.name],
                    input=text.encode("utf-8"),
                    check=True,
                    timeout=30,
                    capture_output=True,
                )
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f"Piper không thể tổng hợp giọng nói: {exc}") from exc
            return Path(output.name).read_bytes()


piper_service = PiperService()
