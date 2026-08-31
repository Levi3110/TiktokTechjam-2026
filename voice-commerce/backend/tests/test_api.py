from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_chat_round_trip() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    created = client.post(
        "/api/sessions",
        json={"user_id": "api-test", "initial_intent": "browsing"},
    )
    assert created.status_code == 200

    response = client.post(
        "/api/chat",
        json={
            "session_id": created.json()["session_id"],
            "message": "Tôi muốn mua tai nghe dưới 2 triệu",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "buying"
    assert body["intent_changed"] is True
    assert all(product["price"] <= 2_000_000 for product in body["products"])
