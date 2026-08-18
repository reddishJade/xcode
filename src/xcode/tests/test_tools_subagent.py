"""Session-backed subagent 与 continuation 契约测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from xcode.ai.events import FinalMessage, Message, ProviderEvent, TextDelta
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.agent.request import DefaultRequestAssembler
from xcode.agent.types import ToolOutput, ToolSpec
from xcode.coding_agent.tools.subagent import (
    BUILD_SUBAGENT_PROMPTS,
    _bounded_prompt,
    _max_concurrent,
    _parse_tasks,
)
from xcode.harness.agent_runtime.composition import AgentComposition
from xcode.harness.agent_runtime.cancellation import CancellationToken
from xcode.harness.agent_runtime.config import GateConfig
from xcode.harness.agent_runtime.subagents import SubagentSessionManager
from xcode.harness.agent_runtime.subagents import _ChildCancellationToken
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.config import AgentConfig
from xcode.harness.session import SessionStore


class _Provider:
    model = "child-model"
    base_url = "https://child.invalid"
    transport = "test"
    thinking = False
    reasoning_effort = None

    def __init__(self) -> None:
        self.requests: list[list[Message]] = []
        self.tool_requests: list[list[str]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ):
        del options, kwargs
        self.requests.append(deepcopy(messages))
        self.tool_requests.append([tool.name for tool in tools])
        answer = f"child-answer-{len(self.requests)}"
        yield cast(ProviderEvent, TextDelta(answer))
        yield cast(
            ProviderEvent,
            FinalMessage(content=answer, stop_reason="end_turn"),
        )


class _AllowMode:
    current_mode = "act"

    def check_call(self, _call: object) -> str:
        return "allow"


def _gate() -> ToolGate:
    return ToolGate(
        mode_state=cast(Any, _AllowMode()),
        approval_callback=None,
        permission_policy=None,
        hook_manager=None,
        audit_logger=None,
        session_id="parent",
        mode_fallbacks={"act": "allow"},
    )


def _manager(
    tmp_path: Path,
    provider: _Provider,
    tools: tuple[ToolSpec, ...] = (),
) -> SubagentSessionManager:
    store = SessionStore(tmp_path / "sessions", project_root=tmp_path)
    store.ensure_metadata("parent task")
    store.append("event", {"type": "parent/test"})
    manager = SubagentSessionManager(
        provider=cast(Any, provider),
        coding_tools=tools,
        research_tools=tools,
        system_prompts=BUILD_SUBAGENT_PROMPTS,
        parent_store=store,
    )
    composition = AgentComposition.create(
        primary_provider=cast(Any, provider),
        fallback_provider=None,
        registry=(),
        config=AgentConfig(),
        gate=GateConfig(),
        request_assembler=DefaultRequestAssembler(),
        runtime_context_provider=None,
    )
    manager.bind_parent(lambda: composition, _gate(), CancellationToken())
    return manager


def _tool(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        input_hint="JSON",
        handler=lambda _data, _update: ToolOutput("ok"),
        schema={"type": "object", "properties": {}},
    )


def test_parse_single_requires_explicit_mode() -> None:
    assert _parse_tasks({"prompt": "Inspect auth"}) == (
        "Error: mode must be one_shot or continuable"
    )
    assert _parse_tasks(
        {
            "description": "scan auth",
            "prompt": "Inspect auth",
            "mode": "continuable",
        }
    ) == [
        {
            "description": "scan auth",
            "prompt": "Inspect auth",
            "subagent_type": "coding",
            "mode": "continuable",
        }
    ]


def test_parallel_tasks_are_always_one_shot() -> None:
    parsed = _parse_tasks(
        {
            "subagent_type": "research",
            "tasks": [
                {"description": "auth", "prompt": "Inspect auth"},
                {"description": "db", "prompt": "Inspect db"},
            ],
        }
    )
    assert not isinstance(parsed, str)
    assert [task["mode"] for task in parsed] == ["one_shot", "one_shot"]
    assert [task["subagent_type"] for task in parsed] == ["research", "research"]


def test_batch_limit_and_bounded_prompt() -> None:
    assert _max_concurrent(0) == 1
    assert _max_concurrent(99) == 16
    assert "return a concise summary" in _bounded_prompt("Inspect auth")


@pytest.mark.asyncio
async def test_one_shot_child_has_independent_durable_session(tmp_path: Path) -> None:
    provider = _Provider()
    manager = _manager(tmp_path, provider)

    result = await manager.execute(
        description="inspect runtime",
        prompt="child task only",
        subagent_type="coding",
        mode="one_shot",
        run_id="run-1",
        batch_id="batch-1",
        task_index=1,
    )

    assert result.status == "completed"
    assert result.answer == "child-answer-1"
    descriptors = manager.list_children()
    assert [item.child_session_id for item in descriptors] == [result.child_session_id]
    child_info = manager._parent_store.find_by_id(result.child_session_id)
    assert child_info is not None
    assert child_info.parent_id == manager._parent_store.session_id
    child_events = _event_types(child_info.path)
    assert child_events[:3] == [
        "subagent/descriptor",
        "subagent/activation",
        "inbox/inserted",
    ]
    assert "provider_request" in child_events
    assert "final" in child_events
    assert child_events[-1] == "subagent/activation"
    assert descriptors[0].tool_names == ()
    assert any(
        message.get("role") == "user" and "child task only" in str(message)
        for message in provider.requests[0]
    )


@pytest.mark.asyncio
async def test_continuable_child_cold_resumes_same_session(tmp_path: Path) -> None:
    provider = _Provider()
    first_manager = _manager(tmp_path, provider)
    created = await first_manager.execute(
        description="continue parser",
        prompt="first child turn",
        subagent_type="coding",
        mode="continuable",
        run_id="run-1",
        batch_id="batch-1",
        task_index=1,
    )
    first_manager.release(created.child_session_id)

    recovered_manager = SubagentSessionManager(
        provider=cast(Any, provider),
        coding_tools=(),
        research_tools=(),
        system_prompts=BUILD_SUBAGENT_PROMPTS,
        parent_store=first_manager._parent_store,
    )
    recovered_manager.bind_parent(
        first_manager._composition_provider or cast(Any, lambda: None),
        _gate(),
        CancellationToken(),
    )
    continued = await recovered_manager.send(
        created.child_session_id,
        "second child turn",
    )

    assert continued.child_session_id == created.child_session_id
    assert continued.answer == "child-answer-2"
    second_request = provider.requests[1]
    assert any("first child turn" in str(message) for message in second_request)
    assert any("child-answer-1" in str(message) for message in second_request)
    assert any("second child turn" in str(message) for message in second_request)
    child_info = recovered_manager._parent_store.find_by_id(created.child_session_id)
    assert child_info is not None
    store = SessionStore(child_info.path.parent, project_root=tmp_path)
    store.resume(child_info.path)
    activation_states = [
        entry.content["data"]["status"]
        for entry in store.build_branch()
        if entry.type == "event"
        and isinstance(entry.content, dict)
        and entry.content.get("type") == "subagent/activation"
    ]
    assert activation_states == ["materialized", "released", "materialized"]


def test_child_cancellation_observes_parent_without_resetting_it() -> None:
    parent = CancellationToken()
    child = _ChildCancellationToken(parent)

    parent.cancel("parent stopped")
    child.reset()

    assert child.is_cancelled()
    assert child.reason == "parent stopped"


@pytest.mark.asyncio
async def test_control_requires_current_direct_parent(tmp_path: Path) -> None:
    provider = _Provider()
    manager = _manager(tmp_path, provider)
    created = await manager.execute(
        description="owned child",
        prompt="first turn",
        subagent_type="coding",
        mode="continuable",
        run_id="run-1",
        batch_id="batch-1",
        task_index=1,
    )
    manager._parent_store.clear()

    with pytest.raises(PermissionError):
        manager.release(created.child_session_id)


@pytest.mark.asyncio
async def test_cold_activation_cannot_gain_new_tools(tmp_path: Path) -> None:
    provider = _Provider()
    read_tool = _tool("read_file")
    created_manager = _manager(tmp_path, provider, (read_tool,))
    created = await created_manager.execute(
        description="bounded authority",
        prompt="first turn",
        subagent_type="coding",
        mode="continuable",
        run_id="run-1",
        batch_id="batch-1",
        task_index=1,
    )
    created_manager.release(created.child_session_id)

    expanded_manager = SubagentSessionManager(
        provider=cast(Any, provider),
        coding_tools=(read_tool, _tool("bash")),
        research_tools=(),
        system_prompts=BUILD_SUBAGENT_PROMPTS,
        parent_store=created_manager._parent_store,
    )
    expanded_manager.bind_parent(
        created_manager._composition_provider or cast(Any, lambda: None),
        _gate(),
        CancellationToken(),
    )
    await expanded_manager.send(created.child_session_id, "second turn")

    assert provider.tool_requests == [["read_file"], ["read_file"]]


@pytest.mark.asyncio
async def test_close_releases_children_without_deleting_sessions(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    manager = _manager(tmp_path, provider)
    created = await manager.execute(
        description="durable after close",
        prompt="first turn",
        subagent_type="coding",
        mode="continuable",
        run_id="run-1",
        batch_id="batch-1",
        task_index=1,
    )

    manager.close(timeout_seconds=0.1)

    child_info = manager._parent_store.find_by_id(created.child_session_id)
    assert child_info is not None
    store = SessionStore(child_info.path.parent, project_root=tmp_path)
    store.resume(child_info.path)
    activation_states = [
        entry.content["data"]["status"]
        for entry in store.build_branch()
        if entry.type == "event"
        and isinstance(entry.content, dict)
        and entry.content.get("type") == "subagent/activation"
    ]
    assert activation_states == ["materialized", "released"]
    with pytest.raises(RuntimeError, match="closed"):
        await manager.send(created.child_session_id, "must not run")


def _event_types(path: Path) -> list[str]:
    store = SessionStore(path.parent, project_root=path.parent.parent)
    store.resume(path)
    return [
        str(entry.content.get("type"))
        for entry in store.build_branch()
        if entry.type == "event" and isinstance(entry.content, dict)
    ]
