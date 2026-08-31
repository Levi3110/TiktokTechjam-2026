from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import settings
from app.models import Intent, Product


class QwenServiceError(RuntimeError):
    """The configured Qwen endpoint returned an unusable response."""


class QwenClient:
    async def answer(
        self,
        query: str,
        intent: Intent,
        products: list[Product],
        memory: list[str],
        constraints: dict[str, Any],
    ) -> tuple[str, str]:
        if settings.qwen_mode == "demo":
            return self._demo_answer(intent, products, constraints), "demo"

        product_context = [product.model_dump() for product in products]
        system = (
            "Bạn là Mây, trợ lý mua sắm bằng giọng nói. "
            "Luôn trả lời bằng tiếng Việt tự nhiên và ngắn gọn để có thể đọc thành tiếng. "
            "Chỉ dùng dữ liệu sản phẩm được cung cấp, không tự tạo giá, tồn kho hay tính năng. "
            "Nếu intent là buying, hãy tôn trọng ràng buộc cứng và giúp người dùng chốt lựa chọn. "
            "Nếu intent là browsing, hãy giúp khám phá và kết thúc bằng một câu hỏi gợi mở. "
            "Không dùng markdown, bảng hoặc emoji."
        )
        prompt = json.dumps(
            {
                "intent": intent.value,
                "user_query": query,
                "relevant_user_memory": memory,
                "constraints": constraints,
                "retrieved_products": product_context,
            },
            ensure_ascii=False,
        )
        payload = {
            "model": settings.qwen_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": settings.qwen_max_tokens,
            "top_k": -1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=settings.qwen_timeout_seconds) as client:
            response = await client.post(
                f"{settings.qwen_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.qwen_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise QwenServiceError("Qwen trả về response không đúng định dạng chat completions.") from exc
        if not isinstance(content, str) or not content.strip():
            raise QwenServiceError("Qwen trả về nội dung rỗng.")
        return content.strip(), settings.qwen_model

    @staticmethod
    def _demo_answer(intent: Intent, products: list[Product], constraints: dict[str, Any]) -> str:
        if not products:
            return "Mình chưa tìm thấy sản phẩm phù hợp. Bạn có thể nới ngân sách hoặc đổi danh mục nhé."
        names = ", ".join(product.name for product in products[:3])
        if intent == Intent.BUYING:
            budget = constraints.get("max_price")
            budget_text = f" trong ngân sách {budget:,.0f}đ" if budget else ""
            return f"Mình đã lọc các lựa chọn phù hợp{budget_text}. Nổi bật là {names}. Bạn muốn mình so sánh hai mẫu nào để chốt?"
        return f"Bạn có thể bắt đầu với {names}. Mình đang cho bạn xem nhiều hướng khác nhau; bạn ưu tiên giá, thiết kế hay hiệu năng?"


qwen_client = QwenClient()
