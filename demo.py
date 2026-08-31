#!/usr/bin/env python3
"""Interactive terminal demo for the TechJam conversational search agent."""

from __future__ import annotations

import argparse
import json
import uuid

from starter.agent import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--message", help="Run one message and exit instead of interactive mode")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def print_response(agent: Agent, response: dict) -> None:
    products = {str(item["parent_asin"]): item for item in agent.retriever.products}
    print(f"\nQwen: {response['message']}")
    print(f"ask_attribute: {response.get('ask_attribute')}")
    for rank, item in enumerate(response.get("recommendations", []), start=1):
        parent_asin = str(item["parent_asin"])
        product = products.get(parent_asin, {})
        price = product.get("price")
        price_text = f" | ${price}" if price is not None else ""
        print(f"  {rank}. {product.get('title', parent_asin)}{price_text} | {parent_asin}")
    print("usage:", json.dumps(response.get("usage", {}), ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if not 1 <= args.top_k <= 10:
        raise SystemExit("--top-k must be between 1 and 10")

    print("Loading the catalog, SBERT, and FAISS...")
    agent = Agent(args.catalog)
    session_id = f"demo_{uuid.uuid4().hex}"
    profile = {
        "summary": "Prioritizes comfort, durability, and products that fit the intended use.",
        "preference_tags": ["comfort", "durability"],
        "rating_style": "usually positive",
    }
    agent.reset(session_id, profile)

    if args.message:
        print_response(agent, agent.respond(session_id, args.message, 1, args.top_k))
        return

    print("Ready. Type /reset to clear conversation memory or /quit to exit.")
    turn = 1
    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExited.")
            return
        if not message:
            continue
        if message.lower() in {"/quit", "/exit"}:
            print("Exited.")
            return
        if message.lower() == "/reset":
            session_id = f"demo_{uuid.uuid4().hex}"
            agent.reset(session_id, profile)
            turn = 1
            print("Conversation memory cleared.")
            continue
        print_response(agent, agent.respond(session_id, message, turn, args.top_k))
        turn += 1


if __name__ == "__main__":
    main()
