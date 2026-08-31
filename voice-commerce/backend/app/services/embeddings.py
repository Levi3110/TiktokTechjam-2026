from __future__ import annotations

import hashlib
import math
import os
import re
from threading import RLock
from typing import Any

from app.config import settings


# Avoid native OpenMP crashes/oversubscription in web workers, especially on macOS ARM.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Deterministic embedding fallback for environments without ML packages."""
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        position = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1 if digest[4] & 1 else -1
        vector[position] += float(sign)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


class EmbeddingEngine:
    """Shared SBERT encoder and FAISS search service used by catalog and memory RAG."""

    def __init__(self) -> None:
        self._encoder: Any = None
        self._faiss: Any = None
        self._numpy: Any = None
        self._lock = RLock()
        self.backend = "hash+cosine"
        self.model_id = "hash-v1"
        self.dimensions = 384
        self._setup()

    def _setup(self) -> None:
        if not settings.enable_sbert:
            return
        try:
            import faiss
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            self._encoder = SentenceTransformer(settings.sbert_model)
            self._faiss = faiss
            self._numpy = np
            dimension_getter = (
                self._encoder.get_embedding_dimension
                if hasattr(self._encoder, "get_embedding_dimension")
                else self._encoder.get_sentence_embedding_dimension
            )
            self.dimensions = int(dimension_getter())
            self.backend = "sbert+faiss"
            self.model_id = settings.sbert_model
        except (ImportError, OSError, ValueError):
            # API and tests remain operational until optional AI dependencies are installed.
            self._encoder = None
            self._faiss = None
            self._numpy = None

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._encoder is None:
            return [hash_embedding(text, self.dimensions) for text in texts]
        with self._lock:
            matrix = self._encoder.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
        return matrix.tolist()

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def rank(
        self,
        query: str,
        vectors: list[list[float]],
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        if not vectors:
            return []
        limit = min(limit or len(vectors), len(vectors))
        query_vector = self.embed(query)
        if self._faiss is not None and self._numpy is not None:
            matrix = self._numpy.asarray(vectors, dtype="float32")
            index = self._faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            scores, indexes = index.search(
                self._numpy.asarray([query_vector], dtype="float32"), limit
            )
            return [
                (int(index_value), float(score))
                for index_value, score in zip(indexes[0], scores[0], strict=True)
                if index_value >= 0
            ]
        scores = [cosine(query_vector, vector) for vector in vectors]
        indexes = sorted(range(len(vectors)), key=scores.__getitem__, reverse=True)[:limit]
        return [(index, scores[index]) for index in indexes]


embedding_engine = EmbeddingEngine()
