from types import SimpleNamespace

import starter.qwen_ranker as ranker_module
from starter.qwen_ranker import QwenRanker
from starter.retrieval import Candidate


def test_qwen_reorders_candidates_and_reports_usage(monkeypatch) -> None:
    first = Candidate({"parent_asin": "A", "title": "First"}, 0.9, {"bm25": 2.0})
    second = Candidate({"parent_asin": "B", "title": "Second"}, 0.8, {"vector": 0.8})
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"ranked_ids":["B","A"],"message":"Best match first.",'
                    '"ask_attribute":"material","suggested_topics":[]}'
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
    )

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "qwen3.6-27b"
            assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
            return response

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(
        ranker_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(OpenAI=FakeOpenAI),
    )
    ranker = QwenRanker()
    ranker.enabled = True

    result = ranker.rank(
        intent="buying",
        query="I need the best option",
        constraints={"material": ["cotton"]},
        memory=["Prefers comfort"],
        candidates=[first, second],
        top_k=2,
        proposed_question="material",
    )

    assert [item.product["parent_asin"] for item in result.candidates] == ["B", "A"]
    assert result.provider == "qwen3.6-27b"
    assert result.ask_attribute == "material"
    assert result.usage == {"prompt_tokens": 120, "completion_tokens": 30}


def test_constraint_prerank_does_not_change_deterministic_fusion_fallback() -> None:
    fused_first = Candidate(
        {"parent_asin": "A", "title": "General boots"},
        0.9,
        {"bm25": 2.0},
    )
    exact_second = Candidate(
        {"parent_asin": "B", "title": "Black leather winter boots"},
        0.8,
        {"metadata": 3.0},
    )
    ranker = QwenRanker()
    ranker.enabled = False

    result = ranker.rank(
        intent="buying",
        query="black leather winter boots",
        constraints={"material": ["leather"]},
        memory=[],
        candidates=[fused_first, exact_second],
        top_k=2,
        proposed_question=None,
    )

    assert [item.product["parent_asin"] for item in result.candidates] == ["A", "B"]
