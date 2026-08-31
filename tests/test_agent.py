from __future__ import annotations

import json
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from starter.agent import Agent
from starter.memory import ConversationMemory, SessionState, detect_intent, extract_constraints
from starter.qwen_ranker import QwenRanker


def test_required_agent_interface_is_exact() -> None:
    assert list(inspect.signature(Agent.reset).parameters) == [
        "self", "session_id", "user_profile",
    ]
    assert list(inspect.signature(Agent.respond).parameters) == [
        "self", "session_id", "user_message", "turn", "top_k",
    ]


def test_qwen_endpoint_has_no_source_default(monkeypatch) -> None:
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)

    ranker = QwenRanker()

    assert ranker.base_url == ""
    assert ranker.enabled is False
    assert ranker.last_error == "QWEN_BASE_URL is not configured"


def _write_catalog(path: Path) -> None:
    products = [
        {
            "parent_asin": "RUN-BLUE",
            "title": "Blue lightweight running shoes",
            "features": ["breathable mesh", "comfortable road running"],
            "details": {"department": "women", "color": "blue"},
            "description": ["Daily training shoe"],
            "categories": ["Clothing", "Shoes", "Running Shoes"],
            "store": "Stride",
            "average_rating": 4.7,
            "rating_number": 100,
            "price": 69.0,
        },
        {
            "parent_asin": "BOOT-BLACK",
            "title": "Black leather winter boots",
            "features": ["warm lining", "water resistant"],
            "details": {"department": "women", "material": "leather"},
            "description": ["Outdoor snow boot"],
            "categories": ["Clothing", "Shoes", "Boots"],
            "store": "North",
            "average_rating": 4.5,
            "rating_number": 80,
            "price": 99.0,
        },
        {
            "parent_asin": "SHIRT-WHITE",
            "title": "White cotton office shirt",
            "features": ["classic fit", "soft cotton"],
            "details": {"department": "men", "material": "cotton"},
            "description": ["Formal work shirt"],
            "categories": ["Clothing", "Shirts"],
            "store": "Office",
            "average_rating": 4.2,
            "rating_number": 45,
            "price": 39.0,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in products), encoding="utf-8")


def test_buying_flow_routes_retrieves_and_asks_attribute(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)
    agent.reset(
        "buying-session",
        {"summary": "Prefers comfort and fit", "preference_tags": ["comfort", "fit"]},
    )

    response = agent.respond(
        "buying-session",
        "I'm looking for running shoes. A key requirement is: color: blue.",
        1,
        10,
    )

    assert response["recommendations"][0]["parent_asin"] == "RUN-BLUE"
    assert response["ask_attribute"] is not None
    assert response["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}
    assert agent.memory.get("buying-session").intent == "buying"


def test_browsing_memory_rag_and_intent_override(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)
    agent.reset(
        "browse-session",
        {"summary": "Likes comfortable materials", "preference_tags": ["material", "comfort"]},
    )
    first = agent.respond(
        "browse-session",
        "I'm looking for shoes, but I'm still exploring.",
        1,
        10,
    )
    second = agent.respond(
        "browse-session",
        "Actually, ignore my earlier preference. What I need is: black leather winter boots.",
        3,
        10,
    )

    state = agent.memory.get("browse-session")
    assert first["ask_attribute"] is not None
    assert "explore" in first["message"].lower()
    assert state.intent == "buying"
    assert state.intent_changed is True
    assert second["recommendations"][0]["parent_asin"] == "BOOT-BLACK"
    assert agent.memory.relevant(state, "warm leather footwear")


def test_vietnamese_buying_intent_and_constraints() -> None:
    message = "Tôi muốn mua boots nữ màu đen đi mùa đông, ngân sách dưới 100 đô."

    intent, _ = detect_intent(message, "browsing")
    constraints = extract_constraints(message)

    assert intent == "buying"
    assert constraints["color"] == ["black"]
    assert constraints["use_case"] == ["winter"]
    assert constraints["budget"] == ["100"]
    assert constraints["category"] == ["boots"]


def test_broad_requirement_answer_extracts_multiple_slots() -> None:
    constraints = extract_constraints(
        "For that, what matters is: budget around $80; leather; winter hiking."
    )

    assert constraints["budget"] == ["80"]
    assert constraints["material"] == ["leather"]
    assert constraints["use_case"] == ["winter hiking"]


def test_early_question_collects_multiple_requirements() -> None:
    state = SessionState(
        session_id="multi-slot",
        user_profile={"preference_tags": ["fit"]},
        intent="buying",
        constraints={"category": ["running shoes"]},
    )

    assert ConversationMemory.choose_question(state, 1) == "other"


def test_vague_request_is_not_saved_as_a_category() -> None:
    assert "category" not in extract_constraints("I want something comfortable")


def test_budget_is_soft_and_does_not_delete_over_budget_candidate(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)

    ranking = agent.retriever.route_rankings(
        "winter boots under 50",
        {"category": ["boots"], "budget": ["50"]},
        "buying",
        3,
    )["metadata"]
    ranked_ids = {agent.retriever.ids[index] for index, _ in ranking}

    assert "BOOT-BLACK" in ranked_ids


def test_initial_mcq_intent_hint_wins_first_turn(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)
    agent.reset("mcq-session", {"summary": "Likes comfort"})
    agent.memory.set_initial_intent("mcq-session", "buying")

    response = agent.respond(
        "mcq-session",
        "Show me some winter boots",
        1,
        10,
    )

    assert agent.memory.get("mcq-session").intent == "buying"
    assert response["recommendations"][0]["parent_asin"] == "BOOT-BLACK"


def test_out_of_scope_turn_returns_boundary_without_calling_qwen(
    tmp_path: Path, monkeypatch,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)
    agent.reset("scope-session", {"summary": "Likes comfort"})
    monkeypatch.setattr(
        agent.ranker,
        "rank",
        lambda **_: (_ for _ in ()).throw(AssertionError("Qwen must not run")),
    )

    response = agent.respond("scope-session", "Write Python code for the weather", 1, 10)

    assert response == {
        "message": "I only help find and recommend products.",
        "ask_attribute": None,
        "recommendations": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    assert agent.memory.get("scope-session").messages == []


def test_weak_product_request_asks_before_generation(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)
    agent.reset("evidence-session", {"summary": "Likes comfort"})
    agent.memory.set_initial_intent("evidence-session", "buying")
    monkeypatch.setattr(
        agent.ranker,
        "rank",
        lambda **_: (_ for _ in ()).throw(AssertionError("Qwen must not run")),
    )

    response = agent.respond("evidence-session", "I want something comfortable", 1, 10)

    assert response["message"] == (
        "What type of clothing, shoes, or accessory are you most interested in?"
    )
    assert response["ask_attribute"] == "category"
    assert response["recommendations"] == []
    assert response["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}


def test_classifier_is_read_only_and_reducer_commits_once(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)
    state = agent.memory.reset("checkpoint-session", {"summary": "Likes comfort"})

    decision = agent.memory.plan_turn(state, "I need black winter boots", 1)

    assert state.revision == 0
    assert state.messages == []
    assert state.constraints == {}
    agent.memory.commit_turn(state, "I need black winter boots", decision)
    assert state.revision == 1
    assert state.current_step == "buying"
    assert state.messages == ["I need black winter boots"]


def test_concurrent_transcripts_do_not_lose_session_state(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog)
    agent = Agent(catalog)
    agent.reset("race-session", {"summary": "Likes comfort"})
    messages = ["I need winter boots", "My preferred color is black"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(
            lambda item: agent.respond("race-session", item[1], item[0], 3),
            enumerate(messages, start=1),
        ))

    state = agent.memory.get("race-session")
    assert state.revision == 2
    assert set(state.messages) == set(messages)
    assert len(state.asked_attributes) == 2
    assert all(response["recommendations"] for response in responses)
