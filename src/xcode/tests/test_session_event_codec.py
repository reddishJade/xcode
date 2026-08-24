"""持久化 session 事件编码测试。"""

from xcode.agent.types import DiffRenderIntent
from xcode.ai.events import ToolCall
from xcode.harness.agent_runtime.events import (
    AssistantStructuredEvent,
    AssistantTextBlock,
    FinalStructuredEvent,
    ToolResultBlock,
    ToolResultStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.harness.agent_runtime.result import AgentHarnessResult
from xcode.harness.session.event_codec import (
    SESSION_EVENT_SCHEMA_VERSION,
    encode_session_event,
)


def test_encode_tool_use_event_has_versioned_envelope() -> None:
    event = ToolUseStructuredEvent(
        "tool_use",
        2,
        ToolCall(id="call-1", name="read_file", input={"path": "README.md"}),
    )

    encoded = encode_session_event(event)

    assert encoded["schema_version"] == SESSION_EVENT_SCHEMA_VERSION
    assert encoded["type"] == "tool_use"
    assert encoded["step"] == 2
    assert encoded["data"] == {
        "id": "call-1",
        "name": "read_file",
        "input": {"path": "README.md"},
    }


def test_encode_assistant_and_final_payloads() -> None:
    assistant = encode_session_event(
        AssistantStructuredEvent(
            "assistant",
            1,
            (AssistantTextBlock("done"),),
        )
    )
    final = encode_session_event(
        FinalStructuredEvent(
            "final",
            1,
            AgentHarnessResult(answer="done", messages=[], steps=1, tool_calls=[]),
        )
    )

    assert assistant["data"] == [{"type": "text", "text": "done"}]
    assert final["data"]["answer"] == "done"
    assert final["data"]["termination_reason"] == "completed"


def test_encode_tool_result_preserves_render_intent() -> None:
    encoded = encode_session_event(
        ToolResultStructuredEvent(
            "tool_result",
            2,
            ToolResultBlock(
                tool_use_id="call-1",
                content="edited",
                render_intent=DiffRenderIntent(
                    patch="@@ -1 +1 @@\n-old\n+new",
                    files=("src/app.py",),
                    first_changed_line=1,
                ),
            ),
        )
    )

    assert encoded["data"]["render_intent"] == {
        "kind": "diff",
        "patch": "@@ -1 +1 @@\n-old\n+new",
        "files": ("src/app.py",),
        "first_changed_line": 1,
    }
