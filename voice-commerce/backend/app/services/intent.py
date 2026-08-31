from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.models import Intent


BUYING_MARKERS = {
    "mua", "dat hang", "chot", "gia", "bao nhieu", "ngan sach", "duoi", "toi da",
    "con hang", "giao hang", "thanh toan", "recommend", "goi y cho toi",
}
BROWSING_MARKERS = {
    "xem", "tham khao", "co gi", "xu huong", "so sanh", "kham pha", "browse",
    "chua muon mua", "chi xem",
}
CATEGORIES = {
    "dien thoai": "dien-thoai",
    "smartphone": "dien-thoai",
    "laptop": "laptop",
    "may tinh": "laptop",
    "tai nghe": "tai-nghe",
    "headphone": "tai-nghe",
}


def plain(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def detect_intent(text: str, current: Intent | None, selected: Intent | None = None) -> Intent:
    if selected is not None:
        return selected
    value = plain(text)
    buying_score = sum(marker in value for marker in BUYING_MARKERS)
    browsing_score = sum(marker in value for marker in BROWSING_MARKERS)
    if buying_score > browsing_score:
        return Intent.BUYING
    if browsing_score > buying_score:
        return Intent.BROWSING
    return current or Intent.BROWSING


def extract_constraints(text: str) -> dict[str, Any]:
    value = plain(text)
    constraints: dict[str, Any] = {}
    for marker, category in CATEGORIES.items():
        if marker in value:
            constraints["category"] = category
            break

    money = re.search(r"(?:duoi|toi da|khoang|ngan sach)?\s*(\d+(?:[.,]\d+)?)\s*(trieu|tr|k|nghin|ngan)?", value)
    if money:
        amount = float(money.group(1).replace(",", "."))
        unit = money.group(2)
        if unit in {"trieu", "tr"}:
            amount *= 1_000_000
        elif unit in {"k", "nghin", "ngan"}:
            amount *= 1_000
        if amount >= 100_000:
            constraints["max_price"] = int(amount)
    return constraints


def extract_memory_fact(text: str) -> str | None:
    value = plain(text)
    memory_markers = (
        "toi thich",
        "toi hay",
        "toi thuong",
        "toi luon",
        "uu tien",
        "khong thich",
        "khong dung",
        "mau yeu thich",
        "thuong mua",
        "ngan sach cua toi",
        "toi can",
    )
    if any(marker in value for marker in memory_markers):
        return text.strip()
    return None
