from __future__ import annotations

import asyncio
from pathlib import Path

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    class _HttpxFallbackError(Exception):
        pass

    class _HttpxFallbackModule:
        HTTPError = _HttpxFallbackError

    httpx = _HttpxFallbackModule()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.models import (
    AvatarSpeakRequest,
    ChatRequest,
    ChatResponse,
    SessionCreate,
    SessionResponse,
    TTSRequest,
)
from app.services.livetalking import livetalking_client
from app.services.llm import QwenServiceError
from app.services.memory import memory_store
from app.services.stt import whisper_service
from app.services.tts import piper_service
from app.supervisor import ShoppingSupervisor


app = FastAPI(title="Voice Commerce API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
supervisor = ShoppingSupervisor()


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": settings.qwen_mode,
        "model": settings.qwen_model if settings.qwen_mode != "demo" else None,
        "semantic": supervisor.retriever._semantic_backend,
        "catalog_vectors": len(supervisor.retriever._matrix),
        "memory": memory_store.semantic_backend,
    }


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(payload: SessionCreate) -> SessionResponse:
    session = memory_store.create_session(payload.user_id, payload.initial_intent)
    return SessionResponse(session_id=session.id, user_id=session.user_id, intent=session.intent)


@app.get("/api/users/{user_id}/memories")
async def user_memories(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "memories": memory_store.list_memories(user_id),
        "semantic_backend": memory_store.semantic_backend,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    memory_store.ensure_session(payload.session_id)
    try:
        return await supervisor.run(payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Qwen service error: {exc}") from exc
    except QwenServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict:
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file exceeds 20 MB")
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    try:
        text = await asyncio.to_thread(whisper_service.transcribe, content, suffix)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"text": text}


@app.post("/api/avatar/offer")
async def avatar_offer(payload: dict) -> JSONResponse:
    try:
        result = await livetalking_client.offer(payload)
        return JSONResponse(result)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LiveTalking unavailable: {exc}") from exc


@app.post("/api/tts")
async def synthesize(payload: TTSRequest) -> Response:
    try:
        audio = await asyncio.to_thread(piper_service.synthesize, payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/wav")


@app.post("/api/avatar/speak")
async def avatar_speak(payload: AvatarSpeakRequest) -> dict:
    try:
        return await livetalking_client.speak(payload.session_id, payload.text, payload.interrupt)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LiveTalking unavailable: {exc}") from exc
