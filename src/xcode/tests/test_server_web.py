"""Web 序列化与运行控制器的纯逻辑测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel

from xcode.agent.messages import AssistantMessage
from xcode.agent.types import TextContent, ToolCallContent
from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    MessageStartStructuredEvent,
    TextDeltaStructuredEvent,
    ToolUpdateData,
    ToolUpdateStructuredEvent,
)
from xcode.harness.agent_runtime.result import AgentHarnessResult
from xcode.server.serialize import event_to_dict, to_jsonable
from xcode.server.runner import WebRunHub


class _Color(StrEnum):
    RED = "red"


class _PayloadModel(BaseModel):
    name: str = "m"
    count: int = 1


@dataclass(frozen=True)
class _Nested:
    value: str = "v"
    tags: tuple[str, ...] = ("a", "b")
    model: _PayloadModel = field(default_factory=_PayloadModel)
    color: _Color = _Color.RED


def test_to_jsonable_handles_enum_dataclass_and_pydantic() -> None:
    payload = to_jsonable(_Nested())
    assert payload == {
        "value": "v",
        "tags": ["a", "b"],
        "model": {"name": "m", "count": 1},
        "color": "red",
    }


def test_text_delta_event_to_dict() -> None:
    payload = event_to_dict(TextDeltaStructuredEvent("text_delta", 2, "你好"))
    assert payload["type"] == "text_delta"
    assert payload["step"] == 2
    assert payload["data"] == "你好"


def test_tool_update_event_to_dict() -> None:
    event = ToolUpdateStructuredEvent(
        "tool_update",
        3,
        ToolUpdateData(tool_call_id="c1", tool_name="read_file", partial_result="…"),
    )
    payload = event_to_dict(event)
    assert payload["data"]["tool_call_id"] == "c1"
    assert payload["data"]["tool_name"] == "read_file"
    assert payload["data"]["partial_result"] == "…"


def test_final_event_keeps_metrics() -> None:
    result = AgentHarnessResult(
        answer="ok",
        messages=[{"role": "assistant", "content": "ok"}],
        steps=1,
        tool_calls=[],
        metrics={"llm_calls": 1},
    )
    payload = event_to_dict(FinalStructuredEvent("final", 1, result))
    assert payload["type"] == "final"
    assert payload["data"]["answer"] == "ok"
    assert payload["data"]["metrics"] == {"llm_calls": 1}


def test_message_start_event_with_pydantic_message() -> None:
    message = AssistantMessage(
        content=[TextContent(text="hi"), ToolCallContent(id="t1", name="grep_search")]
    )
    payload = event_to_dict(MessageStartStructuredEvent("message_start", 1, message))
    blocks = payload["data"]["content"]
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1]["type"] == "tool_call"
    assert blocks[1]["name"] == "grep_search"


class _FakeAgent:
    def __init__(self) -> None:
        self.user_approval_callback = None


class _FakeStore:
    session_id = "s1"


class _FakeApp:
    def __init__(self) -> None:
        self.agent = _FakeAgent()
        self.session_store = _FakeStore()

    def get_model_info(self) -> dict:
        return {"model": "m"}

    def mcp_status(self) -> tuple:
        return ()


def test_hub_rejects_submit_while_running() -> None:
    outgoing: list[dict] = []
    hub = WebRunHub(_FakeApp())

    def _sink(payload: dict) -> None:
        outgoing.append(payload)

    hub.attach(_sink)

    class _FakeTask:
        def done(self) -> bool:
            return False

    hub._run_task = _FakeTask()  # type: ignore[assignment]  # 模拟运行中的任务
    hub.submit("hi", None)
    assert outgoing and outgoing[-1]["type"] == "run_error"


def test_hub_install_user_approval_callback() -> None:
    app = _FakeApp()
    hub = WebRunHub(app)
    assert app.agent.user_approval_callback is not None
    assert hub._pending is None
