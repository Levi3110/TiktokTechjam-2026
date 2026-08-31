#!/usr/bin/env python3
"""Measure public-set recall for every retrieval route and fused candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent
from starter.memory import detect_intent, extract_constraints


def profile_memory(profile: dict) -> str:
    return " ".join(
        (
            str(profile.get("summary", "")),
            "preferences " + " ".join(map(str, profile.get("preference_tags", []))),
        )
    ).strip()


def evaluate_routes(catalog_path: Path, dataset_path: Path, route_limit: int) -> dict:
    samples = load_jsonl(dataset_path)
    _, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path)
    retriever = agent.retriever
    id_to_index = {parent_asin: index for index, parent_asin in enumerate(retriever.ids)}
    cutoffs = tuple(value for value in (10, 30, 50, 100, 160) if value <= route_limit)
    route_names = ("bm25", "vector", "metadata", "fusion")
    hits = {name: {cutoff: 0 for cutoff in cutoffs} for name in route_names}
    misses = {name: set() for name in route_names}

    for number, sample in enumerate(samples, start=1):
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        message = initial_message(
            effective_sample,
            coarse_category(categories[target]),
            set(),
        )
        constraints = extract_constraints(message)
        intent, _ = detect_intent(message, None)
        query = agent._query(  # Same production query construction, without state mutation.
            message,
            intent,
            constraints,
            [profile_memory(sample["user_profile"])],
        )
        rankings = retriever.route_rankings(query, constraints, intent, route_limit)
        weights, rrf_constant = retriever.fusion_parameters(intent)
        fused: dict[int, float] = {}
        for route_name, ranking in rankings.items():
            for rank, (index, _) in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + weights[route_name] / (
                    rrf_constant + rank + 1
                )
        rankings["fusion"] = [
            (index, fused[index])
            for index in sorted(fused, key=fused.get, reverse=True)[:route_limit]
        ]

        target_index = id_to_index[target]
        for route_name, ranking in rankings.items():
            rank = next(
                (position for position, (index, _) in enumerate(ranking, start=1) if index == target_index),
                None,
            )
            for cutoff in cutoffs:
                hits[route_name][cutoff] += int(rank is not None and rank <= cutoff)
            if rank is None:
                misses[route_name].add(str(sample["sample_id"]))
        if number % 25 == 0:
            print(f"Processed {number}/{len(samples)} sessions", flush=True)

    sample_count = len(samples)
    return {
        "sample_count": sample_count,
        "embedding_backend": retriever.embedder.backend,
        "faiss": retriever.faiss_index is not None,
        "route_limit": route_limit,
        "hit_rate": {
            route_name: {
                f"@{cutoff}": round(hits[route_name][cutoff] / sample_count, 6)
                for cutoff in cutoffs
            }
            for route_name in route_names
        },
        "misses_at_route_limit": {
            route_name: len(misses[route_name]) for route_name in route_names
        },
        "bm25_and_vector_shared_misses": len(misses["bm25"] & misses["vector"]),
        "notes": {
            "scope": "First user turn only; use local_evaluator.py for end-to-end metrics.",
            "budget_filter": "Soft preference; it never deletes a candidate globally.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--route-limit", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("docs/retrieval_diagnostics.json"))
    args = parser.parse_args()
    result = evaluate_routes(args.catalog, args.dataset, max(10, args.route_limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
