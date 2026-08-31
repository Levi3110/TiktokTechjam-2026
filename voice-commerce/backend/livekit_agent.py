"""Optional LiveKit transport for the same LangGraph supervisor.

The lightweight React UI uses HTTP upload by default so it runs without LiveKit
credentials. Run this worker when the client is connected to a LiveKit room.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterable

import httpx
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, ModelSettings, inference, llm


API_URL = os.getenv("SUPERVISOR_API_URL", "http://localhost:8000")


class CommerceAgent(Agent):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            instructions="Bạn là Mây, trợ lý mua sắm tiếng Việt. Luôn trả lời ngắn gọn và tự nhiên."
        )
        self.session_id = session_id

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        del tools, model_settings
        message = next(
            (item.text_content for item in reversed(chat_ctx.items) if item.role == "user"),
            "",
        )
        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(
                f"{API_URL.rstrip('/')}/api/chat",
                json={"session_id": self.session_id, "message": message},
            )
            response.raise_for_status()
            yield response.json()["answer"]


server = AgentServer()


@server.rtc_session(agent_name="may-commerce")
async def commerce_session(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(
            model=os.getenv("LIVEKIT_STT_MODEL", "deepgram/nova-3"),
            language="vi",
        ),
        # This model only establishes the chained pipeline; CommerceAgent replaces its LLM node.
        llm=inference.LLM(
            model=os.getenv("LIVEKIT_PIPELINE_LLM", "google/gemma-4-31b-it")
        ),
        tts=inference.TTS(
            model=os.getenv("LIVEKIT_TTS_MODEL", "inworld/inworld-tts-2"),
            voice=os.getenv("LIVEKIT_TTS_VOICE", "Ashley"),
        ),
    )
    await session.start(room=ctx.room, agent=CommerceAgent(ctx.room.name))
    await ctx.connect()
    await session.say("Chào bạn, mình là Mây. Hôm nay bạn muốn mua hay chỉ đang tham khảo?")


if __name__ == "__main__":
    agents.cli.run_app(server)
