from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from starter.retrieval import Embedder, tokens


ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)
MATERIALS = "cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|denim"
COLORS = "black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|beige"
BUYING_MARKERS = (
    "key requirement", "need", "must", "buy", "purchase", "budget", "under $",
    "around $", "exactly", "actually", "prioritize", "muốn mua", "cần mua",
    "mua", "cần", "ngân sách", "dưới", "tối đa", "bắt buộc", "ưu tiên",
)
BROWSING_MARKERS = (
    "still exploring", "just browsing", "browse", "ideas", "not sure", "show me",
    "what options", "considering", "xem thử", "tham khảo", "khám phá", "gợi ý",
    "chưa chắc", "chưa biết", "đang cân nhắc",
)

VI_MATERIALS = {
    "da": "leather", "len": "wool", "lụa": "silk", "cotton": "cotton",
    "vải": "fabric", "denim": "denim", "nylon": "nylon", "polyester": "polyester",
}
VI_COLORS = {
    "đen": "black", "trắng": "white", "xanh dương": "blue", "xanh lá": "green",
    "đỏ": "red", "hồng": "pink", "nâu": "brown", "xám": "gray",
    "tím": "purple", "vàng": "yellow", "cam": "orange", "be": "beige",
}

VAGUE_CATEGORY_TERMS = {
    "something", "anything", "item", "product", "comfortable", "durable",
    "nice", "good", "best", "new", "gift", "option", "options",
}


def _is_specific_category(value: str) -> bool:
    category_tokens = set(tokens(value))
    return bool(category_tokens) and bool(category_tokens - VAGUE_CATEGORY_TERMS)


def detect_intent(message: str, current: str | None) -> tuple[str, bool]:
    text = message.lower()
    buying = sum(marker in text for marker in BUYING_MARKERS)
    browsing = sum(marker in text for marker in BROWSING_MARKERS)
    intent = "buying" if buying > browsing else "browsing" if browsing > buying else current or "browsing"
    return intent, current is not None and current != intent


def classify_constraint(value: str) -> str:
    text = value.lower()
    if re.search(r"(?:\$|budget|under|less than|around)\s*\$?\d", text):
        return "budget"
    if re.search(rf"\b(?:{MATERIALS})\b", text):
        return "material"
    if re.search(rf"\b(?:{COLORS})\b", text):
        return "color"
    if re.search(r"\b(?:size|sizing|small|medium|large|wide|narrow|\d{1,2}(?:\.5)?)\b", text):
        return "size"
    if re.search(r"\b(?:casual|formal|classic|modern|style|fit|sleeve|neck|dressy)\b", text):
        return "style"
    if re.search(r"\b(?:running|hiking|walking|gym|winter|summer|outdoor|work|office|travel)\b", text):
        return "use_case"
    if re.search(r"\b(?:brand|made by|store)\b", text):
        return "brand"
    return "feature"


def extract_constraints(message: str) -> dict[str, list[str]]:
    text = " ".join(message.strip().split())
    lowered = text.lower()
    found: dict[str, list[str]] = {}

    category_match = re.search(
        r"(?:looking for|interested in|show me|need|want)\s+(?:an?\s+|some\s+)?(.+?)(?:[.;]|,\s*(?:but|and)|\s+with\s+|$)",
        text,
        re.I,
    )
    if category_match:
        category = category_match.group(1).strip(" .,")
        if category and len(category.split()) <= 12 and _is_specific_category(category):
            found["category"] = [category]

    vietnamese_categories = (
        (r"\b(?:boots?|bốt)\b", "boots"),
        (r"\bgiày\s+chạy(?:\s+bộ)?\b", "running shoes"),
        (r"\bgiày\s+thể\s+thao\b", "sports shoes"),
        (r"\báo\s+khoác\b", "jacket"),
        (r"\báo\s+sơ\s+mi\b", "shirt"),
        (r"\bváy\b", "dress"),
        (r"\bquần\s+jeans?\b", "jeans"),
        (r"\btúi\s+xách\b", "handbag"),
    )
    if "category" not in found:
        for pattern, normalized in vietnamese_categories:
            if re.search(pattern, lowered):
                found["category"] = [normalized]
                break

    for material in re.findall(rf"\b(?:{MATERIALS})\b", lowered):
        found.setdefault("material", []).append(material)
    for color in re.findall(rf"\b(?:{COLORS})\b", lowered):
        found.setdefault("color", []).append(color)

    for vietnamese, normalized in VI_MATERIALS.items():
        if re.search(rf"(?<!\w){re.escape(vietnamese)}(?!\w)", lowered):
            found.setdefault("material", []).append(normalized)
    for vietnamese, normalized in VI_COLORS.items():
        if re.search(rf"(?<!\w){re.escape(vietnamese)}(?!\w)", lowered):
            found.setdefault("color", []).append(normalized)

    budget = re.search(
        r"(?:budget(?:\s+is)?|under|less than|around|up to|"
        r"ngân sách(?:\s+(?:là|dưới|tối đa|khoảng))?|dưới|tối đa|khoảng)"
        r"\s*\$?\s*(\d+(?:[.,]\d+)?)",
        lowered,
    )
    if budget:
        found["budget"] = [budget.group(1).replace(",", ".")]

    vietnamese_use_cases = {
        "mùa đông": "winter", "mùa hè": "summer", "chạy bộ": "running",
        "đi bộ": "walking", "leo núi": "hiking", "tập gym": "gym",
        "văn phòng": "office", "du lịch": "travel",
    }
    for vietnamese, normalized in vietnamese_use_cases.items():
        if vietnamese in lowered:
            found.setdefault("use_case", []).append(normalized)

    size = re.search(r"\b(?:size|sizing)\s*[:=]?\s*([a-z0-9. -]{1,16})", lowered)
    if size:
        found["size"] = [size.group(1).strip()]

    requirement = re.search(r"(?:key requirement is|what matters is|what i need is|prioritize)\s*:\s*(.+)", text, re.I)
    if requirement:
        for value in re.split(r";|,\s+(?:and\s+)?", requirement.group(1)):
            clean = value.strip(" .,")
            if clean:
                attribute = classify_constraint(clean)
                # The dedicated parsers above already normalize numeric budgets
                # and exact material/color values. Avoid retaining a second,
                # differently worded copy of the same requirement.
                if attribute == "budget" and "budget" in found:
                    continue
                found.setdefault(attribute, []).append(clean)
    return {key: list(dict.fromkeys(values)) for key, values in found.items()}


@dataclass
class MemoryItem:
    text: str
    vector: Any


@dataclass
class SessionState:
    session_id: str
    user_profile: dict[str, Any]
    intent: str = "browsing"
    constraints: dict[str, list[str]] = field(default_factory=dict)
    asked_attributes: set[str] = field(default_factory=set)
    no_preference: set[str] = field(default_factory=set)
    messages: list[str] = field(default_factory=list)
    memories: list[MemoryItem] = field(default_factory=list)
    intent_changed: bool = False
    pending_initial_intent: str | None = None
    current_step: str = "browsing"
    revision: int = 0
    last_turn: int = 0
    selected_product: str | None = None
    selected_size: str | None = None


@dataclass(frozen=True)
class TurnDecision:
    """Read-only classifier output consumed by the single state writer."""

    intent: str
    intent_changed: bool
    constraints: dict[str, list[str]]
    no_preference: set[str]
    messages: list[str]
    proposed_question: str | None
    turn: int


class ConversationMemory:
    """Per-session semantic memory using the same vectors as catalog retrieval."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict[str, Any]) -> SessionState:
        state = SessionState(session_id=session_id, user_profile=dict(user_profile))
        profile_text = " ".join(
            (
                str(user_profile.get("summary", "")),
                "preferences " + " ".join(str(item) for item in user_profile.get("preference_tags", [])),
                "rating style " + str(user_profile.get("rating_style", "")),
            )
        ).strip()
        if profile_text:
            state.memories.append(MemoryItem(profile_text, self.embedder.encode([profile_text])[0]))
        self.sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        return self.sessions[session_id]

    def set_initial_intent(self, session_id: str, intent: str) -> None:
        """Apply an explicit UI choice to the next turn without changing Agent API."""
        if intent not in {"buying", "browsing"}:
            raise ValueError("intent must be buying or browsing")
        self.get(session_id).pending_initial_intent = intent

    def plan_turn(self, state: SessionState, message: str, turn: int) -> TurnDecision:
        """Classify a transcript without mutating checkpointed session state."""
        lowered = message.lower()
        previous_intent = state.intent
        intent, intent_changed = detect_intent(message, state.intent)
        if state.pending_initial_intent:
            intent = state.pending_initial_intent
            intent_changed = False
        # The first message establishes intent; it is not an intent override.
        intent_changed = intent_changed and bool(state.messages)
        is_override = any(marker in lowered for marker in ("actually", "ignore my earlier", "instead", "changed my mind"))
        extracted = extract_constraints(message)
        constraints = {key: list(values) for key, values in state.constraints.items()}
        messages = list(state.messages)
        no_preference_values = set(state.no_preference)
        if is_override:
            # Preserve product category but replace conflicting preference values.
            category = constraints.get("category")
            constraints = {"category": category} if category else {}
            intent = "buying"
            intent_changed = previous_intent != "buying"
            messages = messages[:1]
        for attribute, values in extracted.items():
            if is_override and attribute != "category":
                constraints[attribute] = values
            else:
                constraints.setdefault(attribute, [])
                constraints[attribute] = list(dict.fromkeys([*constraints[attribute], *values]))

        no_preference = re.search(r"no (?:additional )?preference for ([a-z_]+)", lowered)
        if no_preference:
            no_preference_values.add(no_preference.group(1))
        messages.append(message)
        messages = messages[-8:]

        projected = SessionState(
            session_id=state.session_id,
            user_profile=state.user_profile,
            intent=intent,
            constraints=constraints,
            asked_attributes=set(state.asked_attributes),
            no_preference=no_preference_values,
            messages=messages,
        )
        return TurnDecision(
            intent=intent,
            intent_changed=intent_changed,
            constraints=constraints,
            no_preference=no_preference_values,
            messages=messages,
            proposed_question=self.choose_question(projected, turn),
            turn=turn,
        )

    def commit_turn(self, state: SessionState, message: str, decision: TurnDecision) -> None:
        """Atomically apply one classifier decision; this is the sole turn-state writer."""
        state.intent = decision.intent
        state.intent_changed = decision.intent_changed
        state.constraints = decision.constraints
        state.no_preference = decision.no_preference
        state.messages = decision.messages
        state.pending_initial_intent = None
        state.current_step = decision.intent
        state.last_turn = max(state.last_turn, decision.turn)
        if decision.proposed_question:
            state.asked_attributes.add(decision.proposed_question)
        state.memories.append(MemoryItem(message, self.embedder.encode([message])[0]))
        state.memories[:] = state.memories[-20:]
        state.revision += 1

    def commit_behavior(
        self,
        state: SessionState,
        text: str,
        *,
        selected_product: str | None = None,
        selected_size: str | None = None,
        current_step: str = "selection",
    ) -> None:
        """Commit a UI behavior event into semantic memory without consuming a turn."""
        if selected_product is not None:
            state.selected_product = selected_product
        if selected_size is not None:
            state.selected_size = selected_size
        state.current_step = current_step
        state.memories.append(MemoryItem(text, self.embedder.encode([text])[0]))
        state.memories[:] = state.memories[-20:]
        state.revision += 1

    def relevant(self, state: SessionState, query: str, limit: int = 5) -> list[str]:
        if not state.memories:
            return []
        query_vector = self.embedder.encode([query])[0]
        scores: list[tuple[int, float]] = []
        for index, memory in enumerate(state.memories):
            score = float(sum(a * b for a, b in zip(memory.vector, query_vector, strict=False)))
            scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [state.memories[index].text for index, score in scores[:limit] if score > 0.08]

    @staticmethod
    def choose_question(state: SessionState, turn: int) -> str | None:
        if turn >= 7:
            return None
        # One broad early question lets the customer state several requirements
        # in a single voice turn. extract_constraints() classifies every value in
        # the resulting semicolon-separated answer, reducing rigid MCQ turns.
        known_preferences = {
            attribute for attribute in state.constraints if attribute != "category"
        }
        if (
            "category" in state.constraints
            and len(known_preferences) < 2
            and "other" not in state.asked_attributes
            and "other" not in state.no_preference
            and turn <= 2
        ):
            return "other"
        profile_tags = [str(item).lower() for item in state.user_profile.get("preference_tags", [])]
        profile_map = {
            "material": "material", "fit": "size", "style": "style",
            "comfort": "feature", "weather": "use_case", "warmth": "material",
            "performance": "use_case", "durability": "material",
        }
        preferred = [profile_map[tag] for tag in profile_tags if tag in profile_map]
        defaults = (
            ["category", "size", "material", "budget", "color", "style", "use_case", "feature"]
            if state.intent == "buying"
            else ["category", "use_case", "style", "material", "color", "feature", "budget"]
        )
        for attribute in [*preferred, *defaults]:
            if (
                attribute in ALLOWED_ATTRIBUTES
                and attribute not in state.constraints
                and attribute not in state.asked_attributes
                and attribute not in state.no_preference
            ):
                return attribute
        return None


QUESTION_TEXT = {
    "category": "What type of clothing, shoes, or accessory are you most interested in?",
    "material": "Do you have a preferred material?",
    "color": "Which color would you prefer?",
    "size": "What size or fit should I prioritize?",
    "style": "What style are you looking for?",
    "brand": "Do you have a preferred brand?",
    "budget": "What budget should I stay within?",
    "feature": "Which product feature matters most to you?",
    "use_case": "What activity or occasion will you use it for?",
    "other": "What are your most important requirements, such as budget, material, fit, or use case?",
}

def localized_question(attribute: str | None, user_message: str) -> str:
    """Return English UI copy while keeping the public agent API unchanged."""
    key = attribute if attribute in QUESTION_TEXT else "other"
    return QUESTION_TEXT[key]
