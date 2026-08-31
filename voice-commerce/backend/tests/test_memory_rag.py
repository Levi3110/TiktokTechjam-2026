import asyncio
from uuid import uuid4

from app.models import ChatRequest, Intent
from app.services.memory import MemoryStore, memory_store
from app.supervisor import ShoppingSupervisor


def test_user_habit_is_embedded_and_semantically_retrieved() -> None:
    store = MemoryStore()
    user_id = f"memory-{uuid4()}"
    assert store.remember(user_id, "Tôi thường mua laptop nhẹ để đi làm") is True

    facts, debug = store.semantic_search(user_id, "Tìm laptop cho công việc")

    assert facts == ["Tôi thường mua laptop nhẹ để đi làm"]
    assert debug["stored_count"] == 1
    assert debug["matched_count"] == 1


def test_frontend_chat_input_becomes_rag_memory_on_next_turn() -> None:
    user_id = f"supervisor-memory-{uuid4()}"
    session = memory_store.create_session(user_id, Intent.BROWSING)
    supervisor = ShoppingSupervisor()

    first = asyncio.run(
        supervisor.run(
            ChatRequest(
                session_id=session.id,
                message="Tôi thích laptop nhẹ để mang đi làm",
            )
        )
    )
    second = asyncio.run(
        supervisor.run(
            ChatRequest(session_id=session.id, message="Tìm laptop phù hợp cho tôi")
        )
    )

    assert first.debug["memory"]["saved_this_turn"] is True
    assert "Tôi thích laptop nhẹ để mang đi làm" in second.memory_used
    assert second.debug["memory"]["matched_count"] >= 1
