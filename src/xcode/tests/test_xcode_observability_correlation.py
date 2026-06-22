"""Hook、结构化事件和最终指标的关联字段测试。"""

from __future__ import annotations

import time
from datetime import datetime

from xcode.ai.events import FinalMessage, TextDelta, ToolCall, ToolCallEvent
from xcode.cli.repl_tools import event_to_dict
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import GateConfig
from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    ToolResultStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.harness.observability import (
    AuditRecord,
    HookManager,
    HookRecord,
    PreToolEvent,
)
from xcode.harness.skills import ToolSpec
from xcode.tests.fixtures import FakeProvider
import pytest

EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class ObservabilityCorrelationTests:
    """验证同一轮 hook、事件和结果可以稳定关联。"""

    def test_typed_hook_subscriber_receives_correlation(self) -> None:
        """typed hook event 保留 HookRecord 的关联字段。"""
        events: list[object] = []
        manager = HookManager()
        manager.subscribe("pre_tool", events.append)

        manager.emit(
            HookRecord(
                "pre_tool",
                session_id="session-1",
                turn_id="turn-1",
                request_id="request-1",
                tool_call_id="call-1",
            )
        )

        event = events[0]
        assert isinstance(event, PreToolEvent)
        assert isinstance(event, PreToolEvent)
        assert event.correlation.session_id == "session-1"
        assert event.correlation.tool_call_id == "call-1"

    def test_tool_run_shares_correlation_and_reports_timing_totals(self) -> None:
        """provider/tool hook 与结构化事件共享 session、turn、request、call id。"""
        records: list[HookRecord] = []
        audits: list[AuditRecord] = []
        manager = HookManager()
        for event in (
            "pre_tool",
            "post_tool",
            "before_agent_start",
            "before_provider_request",
        ):
            manager.register(event, records.append)

        def handler(_data: dict[str, object]) -> str:
            time.sleep(0.01)
            return "ok"

        provider = FakeProvider(
            [
                [
                    ToolCallEvent(calls=[ToolCall(id="call-1", name="read", input={})]),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec(
                    "read",
                    "Read.",
                    "{}",
                    handler,
                    read_only=True,
                    concurrency_safe=True,
                    schema=EMPTY_SCHEMA,
                ),
            ),
            gate=GateConfig(
                session_id="session-123",
                hook_manager=manager,
                audit_logger=audits.append,
            ),
        )

        events = list(agent.run_stream("go"))

        for record in records:
            assert record.session_id == "session-123"
            assert datetime.fromisoformat(record.timestamp) is not None
        provider_record = next(
            record for record in records if record.event == "before_provider_request"
        )
        pre_record = next(record for record in records if record.event == "pre_tool")
        post_record = next(record for record in records if record.event == "post_tool")
        assert pre_record.turn_id == provider_record.turn_id
        assert pre_record.request_id == provider_record.request_id
        assert pre_record.tool_call_id == "call-1"
        assert post_record.tool_call_id == "call-1"
        assert audits[0].turn_id == pre_record.turn_id
        assert audits[0].request_id == pre_record.request_id
        assert audits[0].tool_call_id == "call-1"

        tool_use = next(
            event for event in events if isinstance(event, ToolUseStructuredEvent)
        )
        tool_result = next(
            event for event in events if isinstance(event, ToolResultStructuredEvent)
        )
        assert tool_use.correlation.turn_id == pre_record.turn_id
        assert tool_use.correlation.request_id == pre_record.request_id
        assert tool_use.correlation.tool_call_id == "call-1"
        assert tool_result.correlation.request_id == tool_use.correlation.request_id

        final = events[-1]
        assert isinstance(final, FinalStructuredEvent)
        assert isinstance(final, FinalStructuredEvent)
        metrics = final.data.metrics
        assert metrics is not None
        assert metrics["model_time_ms"] > 0
        assert metrics["tool_time_ms"] > 0
        assert final.correlation.session_id == "session-123"

        serialized = event_to_dict(tool_use)
        assert serialized["correlation"]["tool_call_id"] == "call-1"


if __name__ == "__main__":
    pytest.main()
