from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from rich.console import Console
from rich.text import Text

from .file_refs import FileReference
from .repl_rendering import (
    CLI_COLOR_ERROR,
    CLI_COLOR_INFO,
    CLI_COLOR_SUCCESS,
    CLI_COLOR_TOOL,
    CLI_COLOR_WARNING,
    DEBUG_TOOL_RESULT_PREVIEW_LIMIT,
    NORMAL_TOOL_RESULT_PREVIEW_LIMIT,
    VERBOSE_TOOL_RESULT_PREVIEW_LIMIT,
    single_line_preview,
)
from .tool_catalog import build_tool_catalog
from xcode.harness.agent_runtime.events import (
    AssistantEventBlock,
    AssistantStructuredEvent,
    AssistantTextBlock,
    CompactionStructuredEvent,
    MessageStartStructuredEvent,
    ReasoningDeltaStructuredEvent,
    StructuredAgentEvent,
    TextDeltaStructuredEvent,
    ToolResultBlock,
    ToolResultStructuredEvent,
    ToolUpdateStructuredEvent,
    ToolUseStructuredEvent,
    TurnEndStructuredEvent,
)
from xcode.harness.agent_runtime.result import StructuredAgentResult
from xcode.harness.agent_runtime.execution_modes import ExecutionModeState
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.skills import ToolInput, ToolSpec
from xcode.agent.config import AgentContext, BeforeToolCallContext
from xcode.agent.messages import AssistantMessage
from xcode.agent.types import TextContent, ToolCallContent


def run_tool_command(command: str, app: object) -> str:
    parts = command.split(maxsplit=2)
    if len(parts) < 2:
        return "usage: /tool NAME INPUT\n/tool list - show enabled tools by group"
    tool_name = parts[1]
    registry: tuple[ToolSpec, ...] = tuple(getattr(app, "registry", ()) or ())
    if tool_name == "list":
        catalog = build_tool_catalog()
        enabled_names = {t.name for t in registry}

        lines = ["<visible tools>"]
        core_names = sorted(t.name for t in registry if t.group == "core")
        if core_names:
            lines.append("  core:")
            lines.extend(f"    {n}" for n in core_names)

        noncore_groups = sorted({t.group for t in registry if t.group != "core"})
        for group in noncore_groups:
            lines.append(f"  {group}:")
            tools_in_group = sorted(
                (t for t in registry if t.group == group), key=lambda item: item.name
            )
            for tool in tools_in_group:
                suffix = ""
                if tool.group == "mcp" and "[mcp: " in tool.description:
                    server_name = tool.description.split("[mcp: ")[-1].split("]")[0]
                    suffix = f" [mcp: {server_name}]"
                lines.append(f"    {tool.name}{suffix}")
        lines.append("</visible tools>")

        all_known = set()
        for group_names in catalog.values():
            all_known.update(group_names)
        hidden = sorted(all_known - enabled_names)
        if hidden:
            lines.append("<hidden tools (enable via tools.enabled_groups)>")
            for group in sorted(catalog):
                group_hidden = sorted(catalog[group] & set(hidden))
                if group_hidden:
                    lines.append(f"  {group}:")
                    lines.extend(f"    {name}" for name in group_hidden)
            lines.append("</hidden tools>")

        available_groups = sorted(catalog.keys() - {tool.group for tool in registry})
        if available_groups:
            lines.append("<available groups>")
            lines.extend(f"  {group}" for group in available_groups)
            lines.append("</available groups>")
        return "\n".join(lines)

    tool_map: dict[str, ToolSpec] = {tool.name: tool for tool in registry}
    selected_tool = tool_map.get(tool_name)
    if selected_tool is None:
        return f"unknown tool: {tool_name}"
    raw_input = parts[2] if len(parts) == 3 else ""
    try:
        action_input = parse_tool_input(selected_tool, raw_input)
    except ValueError as exc:
        return str(exc)
    agent = getattr(app, "agent", None)
    return _execute_tool_via_gate(selected_tool, action_input, agent)


def run_shell_shortcut(command: str, app: object) -> str:
    shell_command = command[1:].strip()
    if not shell_command:
        return "usage: !COMMAND"
    return run_tool_command(f"/tool bash {shell_command}", app)


def _execute_tool_via_gate(
    tool: ToolSpec,
    tool_input: ToolInput,
    agent: object,
) -> str:
    """通过 ToolGate 门控 + ToolSpecAdapter 执行 REPL 工具命令。

    保持与 canonical agent loop 一致的权限门控路径：
    ToolGate._precheck_permission → PermissionEngine.decide()（唯一生产调用点）
    ToolSpecAdapter.execute() → handler（纯适配器，不自检权限）

    build_after_tool_hook 不在此处调用，REPL 手动工具命令不经过 agent 轮次，
    无 session/audit 上下文，且为用户显式输入而非 LLM 决策，不写入审计日志。
    """
    import asyncio

    if agent is None:
        return str(tool.handler(tool_input))

    mode_state = ExecutionModeState()
    gate = ToolGate(
        mode_state=mode_state,
        approval_callback=getattr(agent, "approval_callback", None),
        permission_policy=getattr(agent, "permission_policy", None),
        hook_manager=None,
        audit_logger=None,
        session_id="repl",
        restricted_dirs=getattr(agent, "restricted_dirs", ()),
        hook_constraint_providers=getattr(agent, "hook_constraint_providers", ()),
        project_root=getattr(agent, "project_root", None),
    )
    snapshot = gate.snapshot_for((tool,))
    before_hook = gate.build_before_tool_hook(snapshot)

    ctx = BeforeToolCallContext(
        assistant_message=AssistantMessage(content=[]),
        tool_call=ToolCallContent(
            id="repl", name=tool.name, arguments=dict(tool_input)
        ),
        args=tool_input,
        context=AgentContext(),
    )
    before_result = before_hook(ctx, None)
    if before_result is not None:
        return before_result.reason or f"tool {tool.name} was blocked"

    adapted = gate.adapt_tools((tool,))
    result = asyncio.run(adapted[0].execute("repl", tool_input))
    return "".join(
        item.text for item in result.content if isinstance(item, TextContent)
    )


def parse_tool_input(tool: ToolSpec, raw_input: str) -> ToolInput:
    """解析 `/tool` 命令的人类输入；核心工具协议只接收 dict。"""
    text = raw_input.strip()
    if text.startswith(("{", "[")):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON input: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON input must be an object")
        return data
    key = cli_shorthand_key(tool)
    return {key: text} if key else {}


def cli_shorthand_key(tool: ToolSpec) -> str:
    schema = tool.schema or {}
    required = schema.get("required")
    if (
        isinstance(required, list)
        and len(required) == 1
        and isinstance(required[0], str)
    ):
        return required[0]
    return "input"


def brief_input(name: str, raw_input: ToolInput | str) -> str:
    """从工具输入中提取简短的人类可读摘要。"""
    if isinstance(raw_input, dict):
        if name == "bash":
            command = raw_input.get("command") or raw_input.get("input")
            return single_line_preview(f"bash: {command}") if command else name
        parts = []
        for key, value in raw_input.items():
            if value in (None, "", [], {}):
                continue
            parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
        if parts:
            return single_line_preview(f"{name}: {', '.join(parts)}")
        if raw_input:
            key, val = next(iter(raw_input.items()))
            return single_line_preview(f"{name}: {key}={val}")
        return name
    if raw_input:
        return single_line_preview(f"{name}: {raw_input}")
    return name


def tool_intent(name: str, raw_input: ToolInput | str) -> str:
    if not isinstance(raw_input, dict):
        return single_line_preview(f"Run {name}")
    if name == "grep_search":
        pattern = (
            raw_input.get("pattern") or raw_input.get("query") or raw_input.get("input")
        )
        path = raw_input.get("path") or raw_input.get("include") or "workspace"
        if pattern:
            return single_line_preview(f"Search {path} for {pattern}")
    if name == "glob_files":
        pattern = (
            raw_input.get("pattern") or raw_input.get("path") or raw_input.get("input")
        )
        path = raw_input.get("path") if raw_input.get("pattern") else "workspace"
        if pattern:
            return single_line_preview(f"Find {pattern} in {path}")
    if name == "read_file":
        path = raw_input.get("path") or raw_input.get("input")
        if path:
            return single_line_preview(f"Read {path}")
    if name in {"write_file", "edit_file"}:
        path = raw_input.get("path") or raw_input.get("input")
        if path:
            return single_line_preview(f"Edit {path}")
    if name == "bash":
        command = raw_input.get("command") or raw_input.get("input")
        if command:
            return single_line_preview(f"Run {command}")
    return single_line_preview(f"Run {name}")


def summarize_intents(intents: list[str]) -> str:
    if not intents:
        return "workspace"
    if len(intents) == 1:
        return intents[0]
    first = intents[0]
    return single_line_preview(f"{first} and {len(intents) - 1} more")


def event_to_dict(event: StructuredAgentEvent) -> dict[str, Any]:
    return {"type": event.type, "step": event.step, "data": _event_payload(event)}


def _event_payload(event: StructuredAgentEvent) -> object:
    if isinstance(event, (TextDeltaStructuredEvent, ReasoningDeltaStructuredEvent)):
        return event.data
    if isinstance(event, MessageStartStructuredEvent):
        return asdict(event.data) if event.data is not None else None
    if isinstance(event, TurnEndStructuredEvent):
        return {
            "tool_results": [
                {"tool_call_id": r.tool_call_id, "content": r.content}
                for r in event.data.tool_results
            ]
        }
    if isinstance(event, AssistantStructuredEvent):
        return [_assistant_block_payload(block) for block in event.data]
    if isinstance(event, ToolUseStructuredEvent):
        return {"id": event.data.id, "name": event.data.name, "input": event.data.input}
    if isinstance(event, ToolUpdateStructuredEvent):
        return {
            "tool_call_id": event.data.tool_call_id,
            "tool_name": event.data.tool_name,
            "partial_result": event.data.partial_result,
        }
    if isinstance(event, ToolResultStructuredEvent):
        return {
            "tool_use_id": event.data.tool_use_id,
            "content": event.data.content,
            "status": event.data.status,
            "type": "tool_result",
        }
    if isinstance(event, CompactionStructuredEvent):
        return {
            "messages_removed": event.data.messages_removed,
            "messages_after": event.data.messages_after,
            "summary_token_estimate": event.data.summary_token_estimate,
            "trigger": event.data.trigger,
        }
    return {
        "answer": event.data.answer,
        "steps": event.data.steps,
        "tool_calls": [
            {"id": c.id, "name": c.name, "input": c.input}
            for c in event.data.tool_calls
        ],
        "stopped_by_limit": event.data.stopped_by_limit,
        "metrics": event.data.metrics,
        "stopped_by_watchdog": event.data.stopped_by_watchdog,
        "watchdog_reason": event.data.watchdog_reason,
        "needs_follow_up": event.data.needs_follow_up,
        "last_agent": event.data.last_agent,
        "run_state": event.data.run_state.to_dict()
        if event.data.run_state is not None
        else None,
    }


def _assistant_block_payload(block: AssistantEventBlock) -> dict[str, object]:
    if isinstance(block, AssistantTextBlock):
        return {"type": "text", "text": block.text}
    return {
        "type": "tool_use",
        "id": block.id,
        "name": block.name,
        "input": block.input,
    }


def print_tool_call_rich(label: str, console: Console) -> None:
    console.print(Text(f"  • {label}", style=CLI_COLOR_TOOL))


def print_tool_result_rich(
    data: ToolResultBlock,
    verbosity: str,
    console: Console,
) -> None:
    if data.status == "ok" and verbosity == "normal":
        return
    border = {
        "error": CLI_COLOR_ERROR,
        "denied": CLI_COLOR_ERROR,
        "approval_required": CLI_COLOR_WARNING,
    }.get(data.status, CLI_COLOR_SUCCESS if data.status == "ok" else CLI_COLOR_INFO)
    mark = {"error": "✘", "denied": "⊘", "approval_required": "?"}.get(
        data.status, data.status
    )
    limit = {
        "debug": DEBUG_TOOL_RESULT_PREVIEW_LIMIT,
        "verbose": VERBOSE_TOOL_RESULT_PREVIEW_LIMIT,
    }.get(verbosity, NORMAL_TOOL_RESULT_PREVIEW_LIMIT)
    summary = single_line_preview(str(data.content), width=limit)
    console.print(Text(f"  ← {mark} {summary}", style=border))


def final_stop_reason(data: StructuredAgentResult) -> str | None:
    if data.stopped_by_limit:
        return "[stopped] step limit reached"
    if data.stopped_by_watchdog:
        reason = data.watchdog_reason or "repeated tool calls detected"
        return f"[stopped] {reason}"
    return None


def file_reference_event(references: list[FileReference]) -> dict[str, Any]:
    return {
        "type": "file_references",
        "data": [
            {
                "path": reference.path,
                "status": reference.status,
                "error": reference.error,
            }
            for reference in references
        ],
    }
