from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import settings
from app.models import Product
from app.services.embeddings import embedding_engine, tokenize


class HybridProductRetriever:
    """BM25 + semantic + metadata retrieval with reciprocal-rank fusion."""

    def __init__(self, products_path: Path | None = None) -> None:
        path = products_path or Path(__file__).parents[1] / "data" / "products.json"
        seed_rows = json.loads(path.read_text())
        rows = self._load_from_mongodb(seed_rows)
        self.products = [Product.model_validate(row) for row in rows]
        self.documents = [self._document(product) for product in self.products]
        self._semantic_backend = embedding_engine.backend
        self._matrix = embedding_engine.embed_many(self.documents)
        self._setup_bm25()

    @staticmethod
    def _load_from_mongodb(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not settings.mongodb_uri:
            return seed_rows
        try:
            from pymongo import MongoClient

            client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=1500)
            collection = client[settings.mongodb_database]["products"]
            if collection.estimated_document_count() == 0:
                collection.insert_many(seed_rows)
            return list(collection.find({}, {"_id": 0}))
        except (ImportError, OSError):
            return seed_rows
        except Exception:
            # The application remains usable if MongoDB is temporarily offline.
            return seed_rows

    @staticmethod
    def _document(product: Product) -> str:
        attributes = " ".join(
            f"{key} {' '.join(value) if isinstance(value, list) else value}"
            for key, value in product.attributes.items()
        )
        return f"{product.name} {product.category} {product.description} {attributes}"

    def _setup_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([tokenize(document) for document in self.documents])
        except ImportError:
            self._bm25 = None

    def _lexical_ranking(self, query: str) -> list[int]:
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query))
        else:
            query_counts = Counter(tokenize(query))
            scores = []
            for document in self.documents:
                document_counts = Counter(tokenize(document))
                scores.append(sum(min(count, document_counts[token]) for token, count in query_counts.items()))
        return sorted(range(len(self.products)), key=lambda index: float(scores[index]), reverse=True)

    @staticmethod
    def _matches(product: Product, filters: dict[str, Any]) -> bool:
        if filters.get("category") and product.category != filters["category"]:
            return False
        if filters.get("max_price") and product.price > int(filters["max_price"]):
            return False
        if filters.get("min_price") and product.price < int(filters["min_price"]):
            return False
        return product.stock > 0

    def search(self, query: str, filters: dict[str, Any] | None = None, limit: int = 4) -> tuple[list[Product], dict[str, Any]]:
        filters = filters or {}
        eligible = {
            index for index, product in enumerate(self.products) if self._matches(product, filters)
        }
        if not eligible:
            return [], {
                "semantic_backend": self._semantic_backend,
                "candidate_count": 0,
                "fusion": "rrf",
            }

        semantic_results = embedding_engine.rank(query, self._matrix, len(self.products))
        rankings = [self._lexical_ranking(query), [index for index, _ in semantic_results]]
        fused: dict[int, float] = {index: 0.0 for index in eligible}
        for ranking in rankings:
            for rank, index in enumerate(ranking):
                if index in eligible:
                    fused[index] += 1 / (60 + rank + 1)

        ordered = sorted(fused, key=fused.get, reverse=True)[:limit]
        return [self.products[index] for index in ordered], {
            "semantic_backend": self._semantic_backend,
            "embedded_documents": len(self._matrix),
            "semantic_top_score": semantic_results[0][1] if semantic_results else None,
            "candidate_count": len(eligible),
            "fusion": "rrf",
        }
