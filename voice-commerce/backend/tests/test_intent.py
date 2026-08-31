from app.models import Intent
from app.services.intent import detect_intent, extract_constraints


def test_buying_intent_overrides_browsing() -> None:
    result = detect_intent("Tôi muốn mua laptop dưới 20 triệu", Intent.BROWSING)
    assert result == Intent.BUYING


def test_browsing_intent_overrides_buying() -> None:
    result = detect_intent("Tôi chỉ xem và tham khảo tai nghe thôi", Intent.BUYING)
    assert result == Intent.BROWSING


def test_extracts_vietnamese_budget_and_category() -> None:
    constraints = extract_constraints("Tìm laptop dưới 18 triệu")
    assert constraints == {"category": "laptop", "max_price": 18_000_000}

