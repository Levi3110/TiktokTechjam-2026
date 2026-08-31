from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any

from starter.memory import ConversationMemory, localized_question
from starter.qwen_ranker import QwenRanker
from starter.retrieval import Candidate, Embedder, MultiRouteRetriever, product_document, tokens

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass


class Agent:
    """Conversational shopping supervisor with two explicit retrieval/RAG flows.

    BUYING:
      constraints -> BM25/FAISS/metadata -> candidate pool -> Qwen rerank -> products
    BROWSING:
      topic context -> BM25/FAISS/category -> candidate pool -> Qwen topics -> products
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.embedder = Embedder()
        self.retriever = MultiRouteRetriever(catalog_path, self.embedder)
        self.memory = ConversationMemory(self.embedder)
        self.ranker = QwenRanker()
        self._locks_guard = RLock()
        self._session_locks: dict[str, RLock] = {}

    def _session_lock(self, session_id: str) -> RLock:
        with self._locks_guard:
            return self._session_locks.setdefault(session_id, RLock())

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._session_lock(session_id):
            self.memory.reset(session_id, user_profile)

    def record_behavior(
        self,
        session_id: str,
        text: str,
        *,
        selected_product: str | None = None,
        selected_size: str | None = None,
        current_step: str = "selection",
    ) -> None:
        """Add a demo UI selection to semantic memory without changing Agent API."""
        with self._session_lock(session_id):
            self.memory.commit_behavior(
                self.memory.get(session_id),
                text,
                selected_product=selected_product,
                selected_size=selected_size,
                current_step=current_step,
            )

    @staticmethod
    def _query(
        message: str,
        intent: str,
        constraints: dict[str, list[str]],
        memories: list[str],
    ) -> str:
        constraint_text = " ".join(
            f"{attribute} {' '.join(values)}"
            for attribute, values in constraints.items()
        )
        if intent == "buying":
            return " ".join((message, constraint_text, *memories[:4])).strip()
        # Browsing favors semantic context and profile interests over hard filtering.
        return " ".join((message, *memories[:5], constraint_text)).strip()

    @staticmethod
    def _browsing_topics(candidates: list[Candidate]) -> list[str]:
        counts: Counter[str] = Counter()
        for candidate in candidates[:20]:
            categories = candidate.product.get("categories") or []
            if isinstance(categories, list):
                for value in categories[-2:]:
                    topic = str(value).strip()
                    if topic and topic.lower() not in {
                        "clothing", "clothing, shoes & jewelry", "clothing shoes & jewelry"
                    }:
                        counts[topic] += 1
        return [topic for topic, _ in counts.most_common(4)]

    PRODUCT_PATTERN = re.compile(
        r"\b(?:shirts?|t-?shirts?|tees?|blouses?|dresses?|skirts?|pants?|trousers?|"
        r"jeans?|shorts?|jackets?|coats?|sweaters?|hoodies?|suits?|bras?|underwear|"
        r"socks?|shoes?|sneakers?|boots?|sandals?|heels?|slippers?|loafers?|bags?|"
        r"backpacks?|wallets?|belts?|hats?|caps?|scarves?|gloves?|jewelry|accessories|"
        r"footwear|necklaces?|"
        r"bracelets?|rings?|earrings?|watches?|sunglasses?|quần|áo|váy|giày|dép|"
        r"túi|mũ|nhẫn|vòng|đồng hồ)\b",
        re.I,
    )
    SHOPPING_PATTERN = re.compile(
        r"\b(?:buy|buying|purchase|shop|shopping|product|recommend|recommendation|"
        r"gift|outfit|wear|size|fit|color|material|brand|budget|price|mua|sản phẩm|"
        r"gợi ý|kích cỡ|màu|chất liệu|thương hiệu|ngân sách|giá)\b",
        re.I,
    )
    OUT_OF_SCOPE_PATTERN = re.compile(
        r"\b(?:weather|forecast|news|politics|president|python|javascript|coding?|"
        r"homework|equation|translate|translation|recipe|poem|story|song|football|"
        r"thời tiết|tin tức|chính trị|lập trình|bài tập|dịch thuật|công thức nấu ăn)\b",
        re.I,
    )
    EVIDENCE_STOPWORDS = {
        "something", "anything", "item", "items", "product", "products",
        "comfortable", "comfort", "durable", "durability", "nice", "good",
        "best", "new", "help", "recommend", "recommendation", "gift",
        "black", "white", "blue", "red", "pink", "green", "brown", "gray",
        "grey", "purple", "yellow", "orange", "beige", "cotton", "polyester",
        "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
        "denim", "winter", "summer", "men", "women", "man", "woman", "wife",
        "husband", "kids", "kid", "children", "child",
    }

    @classmethod
    def _is_shopping_scope(cls, message: str, state: Any) -> bool:
        """Deterministic scope gate; it never calls the LLM."""
        if cls.PRODUCT_PATTERN.search(message):
            return True
        if cls.OUT_OF_SCOPE_PATTERN.search(message):
            return False
        if cls.SHOPPING_PATTERN.search(message):
            return True
        if state.pending_initial_intent or state.constraints or state.selected_product:
            return True
        return any(
            cls.PRODUCT_PATTERN.search(previous) or cls.SHOPPING_PATTERN.search(previous)
            for previous in state.messages[-3:]
        )

    @classmethod
    def _has_catalog_evidence(
        cls,
        evidence_text: str,
        constraints: dict[str, list[str]],
        candidates: list[Candidate],
    ) -> bool:
        """Require a product/category signal that is present in retrieved rows."""
        if not candidates:
            return False
        query_terms = set(tokens(evidence_text)).difference(cls.EVIDENCE_STOPWORDS)
        best_overlap = max(
            (
                len(query_terms.intersection(tokens(product_document(candidate.product))))
                for candidate in candidates[:12]
            ),
            default=0,
        )
        category_text = " ".join(constraints.get("category", []))
        category_terms = set(tokens(category_text)).difference(cls.EVIDENCE_STOPWORDS)
        category_overlap = max(
            (
                len(
                    category_terms.intersection(
                        tokens(" ".join(str(value) for value in candidate.product.get("categories", [])))
                    )
                )
                for candidate in candidates[:12]
            ),
            default=0,
        )
        has_product_signal = bool(
            cls.PRODUCT_PATTERN.search(evidence_text)
            or constraints.get("category")
            or category_overlap >= 1
        )
        if has_product_signal and best_overlap >= 1:
            return True
        return False

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        # A session lock makes the checkpointer/reducer the only state writer,
        # while unrelated sessions can still retrieve and rank concurrently.
        with self._session_lock(session_id):
            return self._respond_locked(session_id, user_message, turn, top_k)

    def _respond_locked(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self.memory.get(session_id)
        if not self._is_shopping_scope(user_message, state):
            return {
                "message": "I only help find and recommend products.",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        decision = self.memory.plan_turn(state, user_message, turn)
        relevant_memory = self.memory.relevant(state, user_message, limit=5)
        retrieval_query = self._query(
            user_message,
            decision.intent,
            decision.constraints,
            relevant_memory,
        )
        candidate_pool, _retrieval_debug = self.retriever.search(
            retrieval_query,
            decision.constraints,
            decision.intent,
            candidate_limit=max(80, top_k * 8),
        )

        evidence_text = " ".join(
            (
                *state.messages[-3:],
                user_message,
                *(value for values in decision.constraints.values() for value in values),
            )
        )
        if not self._has_catalog_evidence(
            evidence_text, decision.constraints, candidate_pool
        ):
            ask_attribute = (
                decision.proposed_question
                if self.PRODUCT_PATTERN.search(evidence_text)
                else "category"
            ) or "other"
            decision = replace(decision, proposed_question=ask_attribute)
            self.memory.commit_turn(state, user_message, decision)
            return {
                "message": localized_question(ask_attribute, user_message),
                "ask_attribute": ask_attribute,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        self.memory.commit_turn(state, user_message, decision)
        relevant_memory = self.memory.relevant(state, user_message, limit=5)

        proposed_question = decision.proposed_question
        ranked = self.ranker.rank(
            intent=state.intent,
            query=user_message,
            constraints=state.constraints,
            memory=relevant_memory,
            candidates=candidate_pool,
            top_k=min(top_k, 10),
            proposed_question=proposed_question,
        )
        message = ranked.message
        if state.intent == "browsing":
            topics = ranked.suggested_topics or self._browsing_topics(candidate_pool)
            if topics:
                message += " You could explore: " + ", ".join(topics) + "."
        if ranked.ask_attribute:
            question = localized_question(ranked.ask_attribute, user_message)
            if question.lower() not in message.lower():
                message += " " + question
        if state.intent_changed:
            message = "I updated the search to match your new intent. " + message
        if not candidate_pool:
            message = "I could not find a strong match yet. " + (
                localized_question(ranked.ask_attribute, user_message)
                if ranked.ask_attribute
                else "Could you share one more requirement?"
            )

        recommendations = [
            {
                "parent_asin": str(candidate.product["parent_asin"]),
                "score": round(float(candidate.score), 8),
            }
            for candidate in ranked.candidates[: min(top_k, 10)]
        ]
        return {
            "message": message.strip(),
            "ask_attribute": ranked.ask_attribute,
            "recommendations": recommendations,
            "usage": ranked.usage,
        }
