from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Intent(str, Enum):
    BUYING = "buying"
    BROWSING = "browsing"


class Product(BaseModel):
    id: str
    name: str
    category: str
    price: int
    currency: str = "VND"
    description: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    image: str = ""
    stock: int = 0


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)
    selected_intent: Intent | None = None


class ChatResponse(BaseModel):
    session_id: str
    intent: Intent
    intent_changed: bool
    answer: str
    products: list[Product] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    memory_used: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class SessionCreate(BaseModel):
    user_id: str = "demo-user"
    initial_intent: Intent | None = None


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    intent: Intent | None = None


class AvatarSpeakRequest(BaseModel):
    session_id: str
    text: str
    interrupt: bool = True


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
