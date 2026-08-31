from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from app.config import settings
from app.models import Intent
from app.services.embeddings import embedding_engine


@dataclass
class Session:
    id: str
    user_id: str
    intent: Intent | None = None
    turns: list[dict[str, str]] = field(default_factory=list)


@dataclass
class UserMemory:
    text: str
    embedding: list[float]
    created_at: str


class MemoryStore:
    """Session state plus per-user semantic memory for RAG."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._user_memories: dict[str, list[UserMemory]] = {}
        self._loaded_users: set[str] = set()
        self._lock = RLock()
        self._mongo_collection: Any = None
        self._setup_mongodb()

    def _setup_mongodb(self) -> None:
        if not settings.mongodb_uri:
            return
        try:
            from pymongo import ASCENDING, MongoClient

            client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
            collection = client[settings.mongodb_database]["user_memories"]
            collection.create_index(
                [("user_id", ASCENDING), ("text", ASCENDING)],
                unique=True,
            )
            self._mongo_collection = collection
        except Exception:
            self._mongo_collection = None

    @property
    def semantic_backend(self) -> str:
        persistence = "mongodb" if self._mongo_collection is not None else "ram"
        return f"{embedding_engine.backend}+{persistence}"

    def create_session(self, user_id: str, intent: Intent | None = None) -> Session:
        with self._lock:
            session = Session(id=str(uuid4()), user_id=user_id, intent=intent)
            self._sessions[session.id] = session
            return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def ensure_session(self, session_id: str, user_id: str = "demo-user") -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(id=session_id, user_id=user_id)
            return self._sessions[session_id]

    def set_intent(self, session_id: str, intent: Intent) -> None:
        with self._lock:
            self.ensure_session(session_id).intent = intent

    def add_turn(self, session_id: str, role: str, text: str) -> None:
        with self._lock:
            session = self.ensure_session(session_id)
            session.turns.append({"role": role, "text": text})
            session.turns[:] = session.turns[-20:]

    def _load_user(self, user_id: str) -> None:
        if user_id in self._loaded_users:
            return
        memories: list[UserMemory] = []
        if self._mongo_collection is not None:
            rows = self._mongo_collection.find(
                {"user_id": user_id},
                {
                    "_id": 0,
                    "text": 1,
                    "embedding": 1,
                    "embedding_model": 1,
                    "created_at": 1,
                },
            )
            for row in rows:
                vector = row.get("embedding") or []
                if (
                    len(vector) != embedding_engine.dimensions
                    or row.get("embedding_model") != embedding_engine.model_id
                ):
                    vector = embedding_engine.embed(row["text"])
                memories.append(
                    UserMemory(
                        text=row["text"],
                        embedding=[float(value) for value in vector],
                        created_at=str(row.get("created_at", "")),
                    )
                )
        self._user_memories[user_id] = memories[-settings.memory_limit_per_user :]
        self._loaded_users.add(user_id)

    def remember(self, user_id: str, fact: str) -> bool:
        clean_fact = " ".join(fact.split())
        if not clean_fact:
            return False
        vector = embedding_engine.embed(clean_fact)
        created_at = datetime.now(UTC).isoformat()
        with self._lock:
            self._load_user(user_id)
            memories = self._user_memories.setdefault(user_id, [])
            if any(memory.text.casefold() == clean_fact.casefold() for memory in memories):
                return False
            memory = UserMemory(clean_fact, vector, created_at)
            memories.append(memory)
            memories[:] = memories[-settings.memory_limit_per_user :]

        if self._mongo_collection is not None:
            self._mongo_collection.update_one(
                {"user_id": user_id, "text": clean_fact},
                {
                    "$set": {
                        "embedding": vector,
                        "embedding_model": embedding_engine.model_id,
                        "created_at": created_at,
                    }
                },
                upsert=True,
            )
        return True

    def semantic_search(
        self,
        user_id: str,
        query: str,
        limit: int = 4,
    ) -> tuple[list[str], dict[str, Any]]:
        with self._lock:
            self._load_user(user_id)
            memories = list(self._user_memories.get(user_id, []))
        ranked = embedding_engine.rank(
            query,
            [memory.embedding for memory in memories],
            limit=len(memories),
        )
        accepted = [
            (memories[index].text, score)
            for index, score in ranked
            if score >= settings.memory_min_score
        ][:limit]
        return [text for text, _ in accepted], {
            "backend": self.semantic_backend,
            "stored_count": len(memories),
            "matched_count": len(accepted),
            "top_score": accepted[0][1] if accepted else None,
        }

    def relevant_facts(self, user_id: str, query: str, limit: int = 4) -> list[str]:
        facts, _ = self.semantic_search(user_id, query, limit)
        return facts

    def list_memories(self, user_id: str) -> list[str]:
        with self._lock:
            self._load_user(user_id)
            return [memory.text for memory in self._user_memories.get(user_id, [])]


memory_store = MemoryStore()
