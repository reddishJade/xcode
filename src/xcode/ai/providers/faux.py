from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

"""模拟 provider：无真实 API 调用，用于测试。"""


class FauxProvider:
    """模拟 provider，返回固定的 mock 响应。"""

    def __init__(
        self,
        response_text: str = "Mock response",
        delay_seconds: float = 0.0,
    ) -> None:
        self.response_text = response_text
        self.delay_seconds = delay_seconds
        self.model = "mock-model"

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        import asyncio

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        yield {"type": "text_delta", "delta": self.response_text}
        yield {
            "type": "final_message",
            "text": self.response_text,
            "stop_reason": "end_turn",
        }
