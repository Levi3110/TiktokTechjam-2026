import asyncio
from types import SimpleNamespace

import httpx

import app.services.llm as llm_module
from app.models import Intent, Product


def test_qwen_uses_existing_vllm_model_and_options(monkeypatch) -> None:
    captured: dict = {}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url: str, **kwargs) -> httpx.Response:
            captured["url"] = url
            captured.update(kwargs)
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"choices": [{"message": {"content": "Đây là câu trả lời từ Qwen."}}]},
            )

    monkeypatch.setattr(
        llm_module,
        "settings",
        SimpleNamespace(
            qwen_mode="local",
            qwen_model="qwen3.6-27b",
            qwen_base_url="http://172.21.25.98:8000/v1",
            qwen_api_key="not-needed-for-local-vllm",
            qwen_timeout_seconds=60,
            qwen_max_tokens=512,
        ),
    )
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeClient)
    product = Product(
        id="test",
        name="Laptop test",
        category="laptop",
        price=10_000_000,
        description="Sản phẩm test",
        stock=1,
    )

    answer, provider = asyncio.run(
        llm_module.QwenClient().answer(
            "Tôi muốn mua laptop",
            Intent.BUYING,
            [product],
            [],
            {"max_price": 12_000_000},
        )
    )

    assert answer == "Đây là câu trả lời từ Qwen."
    assert provider == "qwen3.6-27b"
    assert captured["url"] == "http://172.21.25.98:8000/v1/chat/completions"
    assert captured["json"]["model"] == "qwen3.6-27b"
    assert captured["json"]["top_k"] == -1
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["headers"]["Authorization"] == "Bearer not-needed-for-local-vllm"
