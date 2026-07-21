"""subagent 工具输入解析单元测试。"""

from __future__ import annotations

from typing import Any, cast

import pytest

from xcode.agent.agent import Agent
from xcode.agent.config import AgentContext, AfterToolCallContext, BeforeToolCallContext
from xcode.agent.messages import AssistantMessage
from xcode.agent.types import (
    AgentToolResult,
    ApprovalRequest,
    TextContent,
    ToolCallContent,
    ToolSpec,
)
from xcode.coding_agent.tools.subagent import (
    _max_concurrent,
    _parse_tasks,
    _run_one,
    build_subagent_tool,
)
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.observability import AuditRecord
from xcode.harness.security import HITLResult
from xcode.harness.security.permission_model import InMemoryGrantStore, Rule


class _AllowMode:
    current_mode = "act"

    def check_call(self, _call: object) -> str:
        return "allow"


def _tool(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        input_hint="JSON",
        handler=lambda _data, _update=None: "executed",
        schema={"type": "object"},
    )


def _gate(**kwargs: Any) -> ToolGate:
    return ToolGate(
        mode_state=cast(Any, _AllowMode()),
        approval_callback=kwargs.get("approval_callback"),
        permission_policy=None,
        hook_manager=None,
        audit_logger=kwargs.get("audit_logger"),
        session_id="test-session",
        project_root=kwargs.get("project_root"),
        session_grant_store=kwargs.get("session_grant_store"),
        user_ruleset=kwargs.get("user_ruleset", ()),
        mode_fallbacks={"act": "allow"},
    )


def test_parse_single_prompt() -> None:
    parsed = _parse_tasks({"description": "scan auth", "prompt": "Inspect auth"})
    assert not isinstance(parsed, str)
    assert parsed == [
        {
            "description": "scan auth",
            "prompt": "Inspect auth",
            "subagent_type": "coding",
        }
    ]


def test_parse_parallel_tasks() -> None:
    parsed = _parse_tasks(
        {
            "subagent_type": "research",
            "tasks": [
                {"description": "auth", "prompt": "Inspect auth"},
                {
                    "description": "db",
                    "prompt": "Inspect db",
                    "subagent_type": "coding",
                },
            ],
        }
    )
    assert not isinstance(parsed, str)
    assert [task["subagent_type"] for task in parsed] == ["research", "coding"]


def test_parse_tasks_requires_prompt() -> None:
    assert (
        _parse_tasks({"tasks": [{"description": "missing"}]})
        == "Error: task 1 prompt is required"
    )


def test_max_concurrent_clamps() -> None:
    assert _max_concurrent(0) == 1
    assert _max_concurrent(99) == 16
    assert _max_concurrent("nope") == 4


def test_bounded_prompt_adds_summary_constraint() -> None:
    from xcode.coding_agent.tools.subagent import _bounded_prompt

    prompt = _bounded_prompt("Inspect auth")
    assert "Inspect auth" in prompt
    assert "return a concise summary" in prompt


def test_subagent_fails_closed_without_permission_gate() -> None:
    tool = build_subagent_tool(
        model=cast(Any, object()),
        coding_tools=[],
        research_tools=[],
    )

    result = tool.handler({"prompt": "inspect"}, None)

    assert result == "Error: subagent permission gate is not configured"


@pytest.mark.asyncio
async def test_subagent_edit_obeys_parent_deny(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    edit_tool = _tool("edit_file")
    gate = _gate(
        project_root=tmp_path,
        user_ruleset=(Rule(action="edit_file", effect="deny"),),
    )

    async def fake_prompt(_self: Agent, _text: str, **kwargs: object) -> str:
        config = kwargs["loop_config"]
        before = cast(Any, config).before_tool_call
        assert before is not None
        tool_call = ToolCallContent(
            id="edit-1",
            name="edit_file",
            arguments={"path": str(tmp_path / "blocked.py")},
        )
        result = before(
            BeforeToolCallContext(
                assistant_message=AssistantMessage(content=[tool_call]),
                tool_call=tool_call,
                args=tool_call.arguments or {},
                context=AgentContext(),
            ),
            None,
        )
        assert result is not None and result.block
        return "blocked"

    monkeypatch.setattr(Agent, "prompt", fake_prompt)

    result = await _run_one(
        {"description": "edit", "prompt": "edit file", "subagent_type": "coding"},
        cast(Any, object()),
        [edit_tool],
        [],
        None,
        None,
        gate,
    )

    assert result == "blocked"


@pytest.mark.asyncio
async def test_subagent_reuses_session_grant_and_audits_child_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    approvals: list[dict[str, Any]] = []
    audits: list[AuditRecord] = []
    store = InMemoryGrantStore()

    def approve(request: ApprovalRequest) -> HITLResult:
        approvals.append(request.action_input)
        return HITLResult("allow", "session")

    gate = _gate(
        project_root=tmp_path,
        approval_callback=approve,
        audit_logger=audits.append,
        session_grant_store=store,
        user_ruleset=(Rule(action="bash", effect="ask"),),
    )
    bash_tool = _tool("bash")
    call_index = 0
    tool_results: list[AgentToolResult] = []

    async def fake_prompt(_self: Agent, _text: str, **kwargs: object) -> str:
        nonlocal call_index
        call_index += 1
        config = cast(Any, kwargs["loop_config"])
        tool_call = ToolCallContent(
            id=f"bash-{call_index}",
            name="bash",
            arguments={"command": "git status"},
        )
        assistant = AssistantMessage(content=[tool_call])
        before_ctx = BeforeToolCallContext(
            assistant_message=assistant,
            tool_call=tool_call,
            args=tool_call.arguments or {},
            context=AgentContext(),
        )
        assert config.before_tool_call(before_ctx, None) is None
        tool_result = AgentToolResult(content=[TextContent(text="ok")])
        config.after_tool_call(
            AfterToolCallContext(
                assistant_message=assistant,
                tool_call=tool_call,
                args=before_ctx.args,
                result=tool_result,
                is_error=False,
                context=AgentContext(),
            ),
            None,
        )
        tool_results.append(tool_result)
        return "ok"

    monkeypatch.setattr(Agent, "prompt", fake_prompt)
    task = {"description": "status", "prompt": "run status", "subagent_type": "coding"}

    await _run_one(task, cast(Any, object()), [bash_tool], [], None, None, gate)
    await _run_one(task, cast(Any, object()), [bash_tool], [], None, None, gate)

    assert len(approvals) == 1
    assert [record.tool for record in audits] == ["bash", "bash"]
    assert audits[0].approval_scope == "session"
    assert audits[1].matched_rule == "session_grant"
    assert audits[1].approval_grant_id is not None
    assert tool_results[1].details == {"permission_notice": "Allowed by session grant"}


def test_subagent_updates_keep_numbered_slots() -> None:
    import io

    from rich.console import Console

    from xcode.cli.commands import ReplState
    from xcode.cli.repl_turn_handler import ToolCallHandler
    from xcode.harness.agent_runtime.events import ToolUpdateData

    output = io.StringIO()
    handler = ToolCallHandler(
        ReplState(), Console(file=output, force_terminal=False, width=120)
    )

    handler.handle_tool_update(
        ToolUpdateData(
            tool_call_id="sub-1",
            tool_name="subagent",
            partial_result="[1] → tools\n[2] → runtime\n[3] → cli",
        )
    )
    handler.handle_tool_update(
        ToolUpdateData(
            tool_call_id="sub-1",
            tool_name="subagent",
            partial_result="[2]   → read src/xcode/coding_agent/tools/subagent.py",
        )
    )
    assert (
        handler._subagent_slots[2]["tool"]
        == "→ read src/xcode/coding_agent/tools/subagent.py"
    )
    handler.handle_tool_update(
        ToolUpdateData(
            tool_call_id="sub-1",
            tool_name="subagent",
            partial_result="[2] ✓ runtime",
        )
    )
    assert handler._subagent_slots == {
        1: {"task": "→ tools", "tool": ""},
        2: {"task": "✓ runtime", "tool": ""},
        3: {"task": "→ cli", "tool": ""},
    }
