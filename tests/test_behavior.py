from __future__ import annotations

from pathlib import Path

from starter.agent import Agent
from starter.behavior import BehaviorStore, ProductImageCache


def test_confirmed_behavior_persists_and_builds_profile_summary(tmp_path: Path) -> None:
    path = tmp_path / "behavior.json"
    product = {
        "parent_asin": "BOOT-BLACK",
        "title": "Black leather winter boots",
        "categories": ["Clothing", "Shoes", "Boots"],
        "store": "North",
        "price": 99.0,
    }

    BehaviorStore(path).record("browser-client", product, "M")
    restored = BehaviorStore(path)

    summary = restored.summary("browser-client")
    assert "Black leather winter boots (size M)" in summary
    assert "Most selected category: Boots" in summary
    assert "Most selected size: M" in summary


def test_selection_event_enters_semantic_session_memory(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        '{"parent_asin":"BOOT-BLACK","title":"Black leather winter boots",'
        '"features":["warm"],"description":["snow boot"],'
        '"categories":["Clothing","Boots"],"details":{},"price":99}\n',
        encoding="utf-8",
    )
    agent = Agent(catalog)
    agent.reset("selection-session", {"summary": "Likes winter footwear"})

    agent.record_behavior(
        "selection-session",
        "The user confirmed Black leather winter boots in size M.",
        selected_product="BOOT-BLACK",
        selected_size="M",
        current_step="confirmed",
    )

    state = agent.memory.get("selection-session")
    assert state.selected_product == "BOOT-BLACK"
    assert state.selected_size == "M"
    assert state.current_step == "confirmed"
    assert state.revision == 1
    assert agent.memory.relevant(state, "black winter boots size M")


def test_product_image_search_has_offline_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NAMAZON_WEB_IMAGE_SEARCH", "false")
    cache = ProductImageCache(tmp_path / "images")

    assert cache.find({"parent_asin": "A1", "title": "Test product"}) is None
