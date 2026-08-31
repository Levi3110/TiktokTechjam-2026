import asyncio

from app.models import ChatRequest, Intent
from app.services.memory import memory_store
from app.supervisor import ShoppingSupervisor


def test_supervisor_switches_workflow() -> None:
    session = memory_store.create_session("test-user", Intent.BROWSING)
    response = asyncio.run(
        ShoppingSupervisor().run(
            ChatRequest(session_id=session.id, message="Tôi muốn mua laptop dưới 18 triệu")
        )
    )
    assert response.intent == Intent.BUYING
    assert response.intent_changed is True
    assert response.extracted["max_price"] == 18_000_000
    assert all(product.price <= 18_000_000 for product in response.products)

