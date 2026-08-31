from __future__ import annotations

import importlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from starter.retrieval import Candidate, flatten


DEFAULT_MODEL = "qwen3.6-27b"
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}


@dataclass
class RankResult:
    candidates: list[Candidate]
    message: str
    ask_attribute: str | None
    suggested_topics: list[str]
    usage: dict[str, int]
    provider: str


class QwenRanker:
    """Optional Qwen reranker backed by a private OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        self.base_url = os.getenv("QWEN_BASE_URL", "").strip()
        self.model = os.getenv("QWEN_MODEL", DEFAULT_MODEL)
        self.api_key = (
            os.getenv("QWEN_API_KEY")
            or os.getenv("CHENGQI_QWEN_API_KEY")
            or "not-needed-for-local-vllm"
        )
        self.timeout = float(os.getenv("QWEN_RERANK_TIMEOUT", "60"))
        requested = os.getenv("QWEN_RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.enabled = requested and bool(self.base_url)
        self._disabled_until = 0.0
        self.last_error = "" if self.base_url else "QWEN_BASE_URL is not configured"

    @staticmethod
    def _candidate_payload(candidates: list[Candidate]) -> list[dict[str, Any]]:
        result = []
        for candidate in candidates[:30]:
            product = candidate.product
            result.append(
                {
                    "parent_asin": str(product["parent_asin"]),
                    "title": str(product.get("title", ""))[:180],
                    "categories": product.get("categories", []),
                    "features": flatten(product.get("features"))[:260],
                    "details": flatten(product.get("details"))[:220],
                    "price": product.get("price"),
                    "rating": product.get("average_rating"),
                    "retrieval_score": round(candidate.score, 6),
                }
            )
        return result

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise

    def rank(
        self,
        *,
        intent: str,
        query: str,
        constraints: dict[str, list[str]],
        memory: list[str],
        candidates: list[Candidate],
        top_k: int,
        proposed_question: str | None,
    ) -> RankResult:
        fallback_message = (
            "I ranked the products that best satisfy your requirements."
            if intent == "buying"
            else "Here are several relevant directions to explore."
        )
        fallback = RankResult(
            candidates=candidates[:top_k],
            message=fallback_message,
            ask_attribute=proposed_question,
            suggested_topics=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            provider="weighted_rrf",
        )
        if not self.enabled or not candidates or time.monotonic() < self._disabled_until:
            return fallback
        try:
            openai = importlib.import_module("openai")
            client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
            system = (
                "You are a product reranking component. Return one valid JSON object only. "
                "Never invent product IDs and only use IDs from candidates. "
                "For buying, prioritize hard-constraint satisfaction before semantic relevance. "
                "For browsing, prioritize relevance plus category/style diversity. "
                "Always write the customer-facing message and suggested topics in clear, "
                "friendly English, regardless of the language used in the query."
            )
            prompt = json.dumps(
                {
                    "task": "rerank_candidates_and_generate_response",
                    "intent": intent,
                    "user_query": query,
                    "semantic_memory": memory,
                    "constraints": constraints,
                    "candidate_products": self._candidate_payload(candidates),
                    "output_schema": {
                        "ranked_ids": ["parent_asin"],
                        "message": "short, friendly English response",
                        "ask_attribute": "one allowed attribute or null",
                        "suggested_topics": ["browsing topic; empty for buying"],
                    },
                    "proposed_question": proposed_question,
                    "max_results": top_k,
                },
                ensure_ascii=False,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=700,
                extra_body={"top_k": -1, "chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
            payload = self._parse_json(content)
            by_id = {str(candidate.product["parent_asin"]): candidate for candidate in candidates}
            ranked: list[Candidate] = []
            seen: set[str] = set()
            for value in payload.get("ranked_ids", []):
                product_id = str(value)
                if product_id in by_id and product_id not in seen:
                    ranked.append(by_id[product_id])
                    seen.add(product_id)
            ranked.extend(
                candidate
                for candidate in candidates
                if str(candidate.product["parent_asin"]) not in seen
            )
            ask_attribute = payload.get("ask_attribute")
            if ask_attribute not in ALLOWED_ATTRIBUTES or ask_attribute != proposed_question:
                ask_attribute = proposed_question
            topics = [str(item)[:80] for item in payload.get("suggested_topics", [])[:5]]
            usage = getattr(response, "usage", None)
            return RankResult(
                candidates=ranked[:top_k],
                message=str(payload.get("message") or fallback_message)[:600],
                ask_attribute=ask_attribute,
                suggested_topics=topics if intent == "browsing" else [],
                usage={
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                },
                provider=self.model,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._disabled_until = time.monotonic() + float(
                os.getenv("QWEN_FAILURE_COOLDOWN", "60")
            )
            return fallback
