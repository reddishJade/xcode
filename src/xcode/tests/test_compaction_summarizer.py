"""模型压缩摘要同步桥接测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from xcode.ai.events import FinalMessage, ProviderEvent, TextDelta, UsageUpdate
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_runtime.compaction import build_compact_summarize_fn


class _SummaryProvider:
    async def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        assert messages
        assert tools == []
        assert options is not None
        assert options.max_tokens == 4096
        yield TextDelta(
            "## Goal\nFinish task\n\n## Progress\nWork continues.\n\n"
            "## Next Steps\nRun tests and report the result."
        )
        yield UsageUpdate(input_tokens=10, output_tokens=10)
        yield FinalMessage(content="", stop_reason="end_turn")


async def test_summarizer_completes_inside_active_event_loop() -> None:
    summarize = build_compact_summarize_fn(_SummaryProvider())

    result = summarize([{"role": "user", "content": "continue"}])

    assert "## Goal" in result
    assert "## Next Steps" in result
