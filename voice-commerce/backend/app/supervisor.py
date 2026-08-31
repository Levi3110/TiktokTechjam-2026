from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models import ChatRequest, ChatResponse, Intent, Product
from app.services.intent import detect_intent, extract_constraints, extract_memory_fact
from app.services.llm import qwen_client
from app.services.memory import memory_store
from app.services.retrieval import HybridProductRetriever


class AgentState(TypedDict, total=False):
    session_id: str
    message: str
    selected_intent: Intent | None
    previous_intent: Intent | None
    intent: Intent
    intent_changed: bool
    user_id: str
    constraints: dict[str, Any]
    memory: list[str]
    memory_saved: bool
    products: list[Product]
    answer: str
    debug: dict[str, Any]


class ShoppingSupervisor:
    def __init__(self, retriever: HybridProductRetriever | None = None) -> None:
        self.retriever = retriever or HybridProductRetriever()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("detect_intent", self._detect_intent)
        graph.add_node("buying_flow", self._buying_flow)
        graph.add_node("browsing_flow", self._browsing_flow)
        graph.add_node("capture_memory", self._capture_memory)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("respond", self._respond)
        graph.add_node("persist", self._persist)
        graph.add_edge(START, "detect_intent")
        graph.add_conditional_edges(
            "detect_intent",
            lambda state: state["intent"].value,
            {Intent.BUYING.value: "buying_flow", Intent.BROWSING.value: "browsing_flow"},
        )
        graph.add_edge("buying_flow", "capture_memory")
        graph.add_edge("browsing_flow", "capture_memory")
        graph.add_edge("capture_memory", "load_memory")
        graph.add_edge("load_memory", "retrieve")
        graph.add_edge("retrieve", "respond")
        graph.add_edge("respond", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    @staticmethod
    def _detect_intent(state: AgentState) -> dict[str, Any]:
        session = memory_store.ensure_session(state["session_id"])
        intent = detect_intent(state["message"], session.intent, state.get("selected_intent"))
        return {
            "intent": intent,
            "previous_intent": session.intent,
            "intent_changed": session.intent is not None and session.intent != intent,
            "user_id": session.user_id,
        }

    @staticmethod
    def _buying_flow(state: AgentState) -> dict[str, Any]:
        constraints = extract_constraints(state["message"])
        return {"constraints": constraints, "debug": {"workflow": "buying"}}

    @staticmethod
    def _browsing_flow(state: AgentState) -> dict[str, Any]:
        constraints = extract_constraints(state["message"])
        constraints.pop("max_price", None)
        return {"constraints": constraints, "debug": {"workflow": "browsing"}}

    @staticmethod
    def _capture_memory(state: AgentState) -> dict[str, Any]:
        fact = extract_memory_fact(state["message"])
        saved = bool(fact and memory_store.remember(state["user_id"], fact))
        return {"memory_saved": saved}

    @staticmethod
    def _load_memory(state: AgentState) -> dict[str, Any]:
        facts, memory_debug = memory_store.semantic_search(
            state["user_id"], state["message"]
        )
        debug = {
            **state.get("debug", {}),
            "memory": {**memory_debug, "saved_this_turn": state.get("memory_saved", False)},
        }
        return {"memory": facts, "debug": debug}

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        enriched_query = " ".join([state["message"], *state.get("memory", [])])
        products, retrieval_debug = self.retriever.search(
            enriched_query, state.get("constraints", {}), limit=4
        )
        debug = {**state.get("debug", {}), "retrieval": retrieval_debug}
        return {"products": products, "debug": debug}

    @staticmethod
    async def _respond(state: AgentState) -> dict[str, Any]:
        answer, provider = await qwen_client.answer(
            query=state["message"],
            intent=state["intent"],
            products=state.get("products", []),
            memory=state.get("memory", []),
            constraints=state.get("constraints", {}),
        )
        debug = {**state.get("debug", {}), "llm": provider}
        return {"answer": answer, "debug": debug}

    @staticmethod
    def _persist(state: AgentState) -> dict[str, Any]:
        memory_store.set_intent(state["session_id"], state["intent"])
        memory_store.add_turn(state["session_id"], "user", state["message"])
        memory_store.add_turn(state["session_id"], "assistant", state["answer"])
        return {}

    async def run(self, request: ChatRequest) -> ChatResponse:
        result = await self.graph.ainvoke(
            {
                "session_id": request.session_id,
                "message": request.message,
                "selected_intent": request.selected_intent,
            }
        )
        return ChatResponse(
            session_id=request.session_id,
            intent=result["intent"],
            intent_changed=result.get("intent_changed", False),
            answer=result["answer"],
            products=result.get("products", []),
            extracted=result.get("constraints", {}),
            memory_used=result.get("memory", []),
            debug=result.get("debug", {}),
        )
