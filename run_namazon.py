#!/usr/bin/env python3
"""Start and supervise LiveKit, LiveTalking, and the NAmazon web application."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
LIVETALKING_ROOT = WORKSPACE_ROOT / "LiveTalking"


class StartupError(RuntimeError):
    pass


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def web_is_ready() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise StartupError(f"Missing {description}: {path}")


def python_has_modules(python: Path, modules: list[str]) -> tuple[bool, str]:
    if not python.is_file():
        return False, f"missing Python environment: {python}"
    probe = "\n".join(f"import {module}" for module in modules)
    try:
        result = subprocess.run(
            [str(python), "-c", probe],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not inspect {python}: {exc}"
    if result.returncode == 0:
        return True, str(python)
    error = (result.stderr or result.stdout).strip().splitlines()
    return False, error[-1] if error else f"dependency probe exited with {result.returncode}"


def run_preflight(web_python: Path, talking_python: Path) -> int:
    livekit_binary = shutil.which("livekit-server")
    whisper_models = (
        PROJECT_ROOT
        / ".models/faster-whisper/models--Systran--faster-whisper-base/snapshots"
    )
    checks: list[tuple[str, bool, str]] = [
        (
            "LiveKit server",
            bool(livekit_binary),
            livekit_binary or "install with: brew install livekit",
        ),
        (
            "NAmazon web entry point",
            (PROJECT_ROOT / "web_demo.py").is_file(),
            str(PROJECT_ROOT / "web_demo.py"),
        ),
        (
            "Product catalog",
            (PROJECT_ROOT / "data/catalog.jsonl").is_file(),
            str(PROJECT_ROOT / "data/catalog.jsonl"),
        ),
        (
            "LiveTalking application",
            (LIVETALKING_ROOT / "app.py").is_file(),
            str(LIVETALKING_ROOT / "app.py"),
        ),
        (
            "Wav2Lip checkpoint",
            (LIVETALKING_ROOT / "models/wav2lip.pth").is_file(),
            str(LIVETALKING_ROOT / "models/wav2lip.pth"),
        ),
        (
            "NAmazon avatar",
            (LIVETALKING_ROOT / "data/avatars/namazon_ai_face").is_dir(),
            str(LIVETALKING_ROOT / "data/avatars/namazon_ai_face"),
        ),
        (
            "Faster-Whisper base model",
            any(whisper_models.glob("*/model.bin")),
            str(whisper_models),
        ),
    ]
    web_ok, web_detail = python_has_modules(
        web_python,
        [
            "numpy",
            "rank_bm25",
            "faiss",
            "sentence_transformers",
            "livekit",
            "faster_whisper",
        ],
    )
    talking_ok, talking_detail = python_has_modules(
        talking_python,
        ["aiohttp", "aiortc", "cv2", "torch", "numpy"],
    )
    checks.extend(
        [
            ("NAmazon Python dependencies", web_ok, web_detail),
            ("LiveTalking Python dependencies", talking_ok, talking_detail),
        ]
    )

    print(f"[NAmazon] Project root: {PROJECT_ROOT}")
    for label, passed, detail in checks:
        print(f"[{'OK' if passed else 'MISSING'}] {label}: {detail}")
    failed = sum(not passed for _, passed, _ in checks)
    if failed:
        print(
            f"\n[NAmazon] Preflight failed: {failed} required item(s) need attention.",
            file=sys.stderr,
        )
        print(
            "[NAmazon] Complete the README 'First-time setup' section, "
            "then run --check again.",
            file=sys.stderr,
        )
        return 1
    print("\n[NAmazon] Preflight passed. Start everything with: python3 run_namazon.py")
    return 0


def start_process(name: str, command: list[str], cwd: Path) -> ManagedProcess:
    print(f"[NAmazon] Starting {name}...", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        start_new_session=True,
    )
    return ManagedProcess(name, process)


def wait_until_ready(
    service: ManagedProcess,
    ready: Callable[[], bool],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = service.process.poll()
        if return_code is not None:
            raise StartupError(f"{service.name} exited during startup (code {return_code})")
        if ready():
            print(f"[NAmazon] {service.name} is ready.", flush=True)
            return
        time.sleep(0.25)
    raise StartupError(f"Timed out waiting for {service.name}")


def stop_processes(processes: list[ManagedProcess]) -> None:
    if not processes:
        return
    print("\n[NAmazon] Stopping services...", flush=True)
    for service in reversed(processes):
        if service.process.poll() is None:
            try:
                shutdown_signal = (
                    signal.SIGTERM if service.name == "LiveTalking" else signal.SIGINT
                )
                os.killpg(service.process.pid, shutdown_signal)
            except (ProcessLookupError, PermissionError):
                service.process.terminate()
    deadline = time.monotonic() + 8
    for service in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            service.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(service.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                service.process.kill()
    print("[NAmazon] All managed services stopped.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the web UI")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate dependencies and local assets, then exit",
    )
    args = parser.parse_args()

    web_python = PROJECT_ROOT / ".venv/bin/python"
    talking_python = LIVETALKING_ROOT / ".venv/bin/python"
    livekit_binary = shutil.which("livekit-server")
    launched: list[ManagedProcess] = []

    if args.check:
        return run_preflight(web_python, talking_python)

    try:
        if port_is_open(7880):
            print("[NAmazon] Reusing LiveKit already running on port 7880.", flush=True)
        else:
            if not livekit_binary:
                raise StartupError("livekit-server is missing. Install it with: brew install livekit")
            service = start_process(
                "LiveKit",
                [livekit_binary, "--dev", "--bind", "127.0.0.1"],
                PROJECT_ROOT,
            )
            launched.append(service)
            wait_until_ready(service, lambda: port_is_open(7880), 30)

        if port_is_open(8010):
            print("[NAmazon] Reusing LiveTalking already running on port 8010.", flush=True)
        else:
            require_file(talking_python, "LiveTalking virtual-environment Python")
            require_file(LIVETALKING_ROOT / "app.py", "LiveTalking application")
            require_file(LIVETALKING_ROOT / "models/wav2lip.pth", "Wav2Lip model")
            avatar = LIVETALKING_ROOT / "data/avatars/namazon_ai_face"
            if not avatar.is_dir():
                raise StartupError(f"Missing NAmazon avatar: {avatar}")
            service = start_process(
                "LiveTalking",
                [str(talking_python), "app.py"],
                LIVETALKING_ROOT,
            )
            launched.append(service)
            wait_until_ready(service, lambda: port_is_open(8010), 180)

        if web_is_ready():
            print("[NAmazon] Reusing web frontend/backend on port 8765.", flush=True)
        else:
            if port_is_open(8765):
                raise StartupError("Port 8765 is occupied by a non-NAmazon process")
            require_file(web_python, "NAmazon virtual-environment Python")
            require_file(PROJECT_ROOT / "data/catalog.jsonl", "product catalog")
            service = start_process(
                "NAmazon frontend/backend",
                [str(web_python), "web_demo.py", "--host", "127.0.0.1", "--port", "8765"],
                PROJECT_ROOT,
            )
            launched.append(service)
            wait_until_ready(service, web_is_ready, 180)

        url = "http://127.0.0.1:8765/"
        print(f"\n[NAmazon] Everything is ready: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

        if not launched:
            print("[NAmazon] All services were already running; nothing new to supervise.", flush=True)
            return 0

        print("[NAmazon] Press Ctrl+C once to stop all services started here.", flush=True)
        while True:
            for service in launched:
                return_code = service.process.poll()
                if return_code is not None:
                    raise StartupError(f"{service.name} stopped unexpectedly (code {return_code})")
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    except StartupError as exc:
        print(f"[NAmazon] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop_processes(launched)


if __name__ == "__main__":
    raise SystemExit(main())
