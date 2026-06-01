from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from xcode.harness.agent_runtime.events import FinalMessage, ProviderEvent, TextDelta

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
    ) -> AsyncIterator[ProviderEvent]:
        import asyncio

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        yield TextDelta(self.response_text)
        yield FinalMessage(self.response_text, "end_turn")
