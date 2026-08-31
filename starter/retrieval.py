from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    return str(value)


def product_document(product: dict[str, Any]) -> str:
    weighted_title = " ".join([flatten(product.get("title"))] * 3)
    return " ".join(
        (
            weighted_title,
            flatten(product.get("categories")),
            flatten(product.get("features")),
            flatten(product.get("details")),
            flatten(product.get("store")),
            flatten(product.get("description")),
        )
    ).strip()


class Embedder:
    """Shared multilingual SBERT encoder with a deterministic offline fallback."""

    def __init__(self) -> None:
        self.model_name = os.getenv(
            "SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.backend = "hash"
        self.dimensions = 384
        self._model: Any = None
        self._np: Any = None
        self._lock = RLock()
        self._setup()

    def _setup(self) -> None:
        enabled = os.getenv("TECHJAM_ENABLE_SBERT", "true").lower() in {"1", "true", "yes"}
        try:
            import numpy as np

            self._np = np
        except ImportError:
            return
        if not enabled:
            return
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            # Prefer an already downloaded model.  Hugging Face otherwise performs
            # several network HEAD requests on every startup, which can stall the
            # evaluator on an offline competition machine.  A genuinely new setup
            # still gets one normal download attempt.
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except (OSError, ValueError):
                self._model = SentenceTransformer(self.model_name)
            getter = (
                self._model.get_embedding_dimension
                if hasattr(self._model, "get_embedding_dimension")
                else self._model.get_sentence_embedding_dimension
            )
            self.dimensions = int(getter())
            self.backend = "sbert"
        except (ImportError, OSError, ValueError):
            self._model = None

    def _hash_vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            position = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[position] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def encode(self, texts: list[str]) -> Any:
        if self._model is not None:
            with self._lock:
                return self._model.encode(
                    texts,
                    batch_size=int(os.getenv("SBERT_BATCH_SIZE", "64")),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 1000,
                ).astype("float32")
        rows = [self._hash_vector(text) for text in texts]
        return self._np.asarray(rows, dtype="float32") if self._np is not None else rows


@dataclass
class Candidate:
    product: dict[str, Any]
    score: float
    route_scores: dict[str, float]


class MultiRouteRetriever:
    """BM25 + FAISS + metadata/category routes fused into one candidate pool."""

    def __init__(self, catalog_path: str | Path, embedder: Embedder | None = None) -> None:
        self.catalog_path = Path(catalog_path)
        if not self.catalog_path.is_file():
            raise FileNotFoundError(
                f"Catalog not found: {self.catalog_path}. Download data/catalog.jsonl.gz first."
            )
        self.embedder = embedder or Embedder()
        self.products: list[dict[str, Any]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            self.products = [json.loads(line) for line in handle if line.strip()]
        self.documents = [product_document(product) for product in self.products]
        self.documents_lower = [document.lower() for document in self.documents]
        self.ids = [str(product["parent_asin"]) for product in self.products]
        self._setup_bm25()
        self.vectors = self._load_or_build_vectors()
        self._setup_faiss()

    def _setup_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi([tokens(document) for document in self.documents])
        except ImportError:
            self.bm25 = None

    def _cache_path(self) -> Path:
        cache_root = Path(
            os.getenv("TECHJAM_CACHE_DIR", str(self.catalog_path.parents[1] / ".cache"))
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        stat = self.catalog_path.stat()
        signature = hashlib.sha256(
            f"{stat.st_size}:{stat.st_mtime_ns}:{self.embedder.backend}:{self.embedder.model_name}".encode()
        ).hexdigest()[:16]
        return cache_root / f"catalog-vectors-{signature}.npy"

    def _load_or_build_vectors(self) -> Any:
        if self.embedder._np is None:
            return self.embedder.encode(self.documents)
        path = self._cache_path()
        if path.is_file():
            matrix = self.embedder._np.load(path, mmap_mode="r")
            if matrix.shape[0] == len(self.products):
                return matrix
        matrix = self.embedder.encode(self.documents)
        self.embedder._np.save(path, matrix)
        return matrix

    def _setup_faiss(self) -> None:
        self.faiss_index: Any = None
        try:
            import faiss

            matrix = self.embedder._np.asarray(self.vectors, dtype="float32")
            self.faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            self.faiss_index.add(matrix)
        except (ImportError, AttributeError, ValueError):
            self.faiss_index = None

    @staticmethod
    def _top_indexes(scores: Any, limit: int) -> list[int]:
        limit = min(limit, len(scores))
        try:
            import numpy as np

            if limit == len(scores):
                return np.argsort(-scores).tolist()
            selected = np.argpartition(scores, -limit)[-limit:]
            return selected[np.argsort(-scores[selected])].tolist()
        except ImportError:
            return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]

    def _bm25_route(self, query: str, limit: int) -> list[tuple[int, float]]:
        query_tokens = tokens(query)
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            query_counts = Counter(query_tokens)
            scores = [
                float(sum(min(count, Counter(tokens(doc))[token]) for token, count in query_counts.items()))
                for doc in self.documents
            ]
        return [(index, float(scores[index])) for index in self._top_indexes(scores, limit)]

    def _vector_route(self, query: str, limit: int) -> list[tuple[int, float]]:
        query_vector = self.embedder.encode([query])
        if self.faiss_index is not None:
            scores, indexes = self.faiss_index.search(query_vector, min(limit, len(self.products)))
            return [
                (int(index), float(score))
                for index, score in zip(indexes[0], scores[0], strict=True)
                if index >= 0
            ]
        if self.embedder._np is not None:
            scores = self.embedder._np.asarray(self.vectors) @ query_vector[0]
            return [(index, float(scores[index])) for index in self._top_indexes(scores, limit)]
        scores = [sum(a * b for a, b in zip(vector, query_vector[0], strict=False)) for vector in self.vectors]
        return [(index, float(scores[index])) for index in self._top_indexes(scores, limit)]

    def _metadata_route(
        self,
        constraints: dict[str, list[str]],
        intent: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        terms = [
            value.lower()
            for attribute, values in constraints.items()
            if attribute != "budget"
            for value in values
            if value
        ]
        budget_values = constraints.get("budget", [])
        budget = None
        if budget_values:
            numeric = re.search(r"\d+(?:\.\d+)?", str(budget_values[-1]).replace(",", ""))
            budget = float(numeric.group()) if numeric else None
        scored: list[tuple[int, float]] = []
        for index, (product, document) in enumerate(zip(self.products, self.documents_lower, strict=True)):
            if budget is not None and intent == "buying":
                try:
                    if float(product.get("price")) > budget:
                        continue
                except (TypeError, ValueError):
                    pass
            matched = sum(1 for term in terms if term in document)
            category = " ".join(str(item) for item in product.get("categories") or []).lower()
            category_bonus = sum(1.5 for term in constraints.get("category", []) if term in category)
            score = matched + category_bonus
            if score > 0 or budget is not None:
                scored.append((index, float(score)))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def search(
        self,
        query: str,
        constraints: dict[str, list[str]],
        intent: str,
        candidate_limit: int = 80,
    ) -> tuple[list[Candidate], dict[str, Any]]:
        route_limit = max(candidate_limit * 2, 100)
        routes = {
            "bm25": self._bm25_route(query, route_limit),
            "vector": self._vector_route(query, route_limit),
            "metadata": self._metadata_route(constraints, intent, route_limit),
        }
        weights = (
            {"bm25": 1.15, "vector": 1.0, "metadata": 1.35}
            if intent == "buying"
            else {"bm25": 0.9, "vector": 1.35, "metadata": 1.1}
        )
        fused: dict[int, float] = {}
        per_route: dict[int, dict[str, float]] = {}
        for route_name, ranking in routes.items():
            for rank, (index, raw_score) in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + weights[route_name] / (60 + rank + 1)
                per_route.setdefault(index, {})[route_name] = raw_score
        ordered = sorted(fused, key=fused.get, reverse=True)[:candidate_limit]
        candidates = [Candidate(self.products[index], fused[index], per_route[index]) for index in ordered]
        return candidates, {
            "intent": intent,
            "embedding_backend": self.embedder.backend,
            "faiss": self.faiss_index is not None,
            "routes": {name: len(values) for name, values in routes.items()},
            "candidate_pool": len(candidates),
            "fusion": "weighted_rrf",
        }
