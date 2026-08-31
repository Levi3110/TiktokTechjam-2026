from app.services.retrieval import HybridProductRetriever


def test_metadata_filter_applies_budget_and_category() -> None:
    retriever = HybridProductRetriever()
    products, debug = retriever.search(
        "laptop nhẹ cho sinh viên", {"category": "laptop", "max_price": 18_000_000}
    )
    assert products
    assert all(product.category == "laptop" for product in products)
    assert all(product.price <= 18_000_000 for product in products)
    assert debug["fusion"] == "rrf"


def test_hard_budget_never_returns_over_budget_fallback() -> None:
    retriever = HybridProductRetriever()
    products, _ = retriever.search("điện thoại", {"max_price": 1})
    assert products == []
