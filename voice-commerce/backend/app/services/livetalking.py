from __future__ import annotations

import httpx

from app.config import settings


class LiveTalkingClient:
    async def offer(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{settings.livetalking_url.rstrip('/')}/offer", json=payload)
            response.raise_for_status()
            return response.json()

    async def speak(self, session_id: str, text: str, interrupt: bool = True) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{settings.livetalking_url.rstrip('/')}/human",
                json={"sessionid": session_id, "text": text, "type": "echo", "interrupt": interrupt},
            )
            response.raise_for_status()
            return response.json()


livetalking_client = LiveTalkingClient()
