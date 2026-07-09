from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel

from rich.console import Console
from rich.text import Text

from xcode.agent.types import ToolInput

from .file_refs import FileReference
from .repl_rendering import (
    CLI_COLOR_DIM,
    CLI_COLOR_ERROR,
    CLI_COLOR_INFO,
    CLI_COLOR_SUCCESS,
    CLI_COLOR_TOOL,
    CLI_COLOR_WARNING,
    DEBUG_TOOL_RESULT_PREVIEW_LIMIT,
    NORMAL_TOOL_RESULT_PREVIEW_LIMIT,
    VERBOSE_TOOL_RESULT_PREVIEW_LIMIT,
)
from .shared.thinking import single_line_preview

from xcode.harness.agent_runtime.events import (
    AssistantEventBlock,
    AssistantStructuredEvent,
    AssistantTextBlock,
    CompactionStructuredEvent,
    MessageStartStructuredEvent,
    ReasoningDeltaStructuredEvent,
    CodingAgentHarnessEvent,
    TextDeltaStructuredEvent,
    ToolResultBlock,
    ToolResultStructuredEvent,
    ToolUpdateStructuredEvent,
    ToolUseStructuredEvent,
    TurnEndStructuredEvent,
)
from xcode.harness.agent_runtime.result import CodingAgentHarnessResult
from xcode.coding_agent.execution_modes import ExecutionModeState
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.agent.types import ToolSpec
from xcode.agent.config import AgentContext, BeforeToolCallContext
from xcode.agent.messages import AssistantMessage
from xcode.agent.types import TextContent, ToolCallContent


def _registry(app: object) -> tuple[ToolSpec, ...]:
    raw = getattr(app, "registry", ())
    return tuple(raw) if raw else ()


def run_tool_command(command: str, app: object) -> str:
    parts = command.split(maxsplit=2)
    if len(parts) < 2:
        return "usage: /tool NAME INPUT\n/tool list - show enabled tools"
    tool_name = parts[1]
    registry = _registry(app)

    if tool_name == "list":
        return _tool_list_legacy(registry)

    # ── Direct execution ──
    selected = _resolve_tool_legacy(tool_name, registry)

    if selected is None:
        return f"unknown tool: {tool_name}"
    raw_input = parts[2] if len(parts) == 3 else ""
    try:
        action_input = parse_tool_input(selected, raw_input)
    except ValueError as exc:
        return str(exc)
    return _execute_tool_via_gate(selected, action_input, getattr(app, "agent", None))


def _tool_list_legacy(registry: tuple[ToolSpec, ...]) -> str:
    lines = [f"## Available Tools ({len(registry)})", ""]
    for t in sorted(registry, key=lambda x: x.name):
        lines.append(f"  - `{t.name}`: {t.description[:80]}")
    return "\n".join(lines)


def _resolve_tool_legacy(name: str, registry: tuple[ToolSpec, ...]) -> ToolSpec | None:
    return next((t for t in registry if t.name == name), None)


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
    if agent is None:
        return str(tool.handler(tool_input, None))

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


def _shorten_path(p: str) -> str:
    """缩短路径，保留尾部和关键部分。"""
    if not p or p == ".":
        return "."
    p = p.replace("\\", "/")
    parts = p.split("/")
    if len(parts) <= 3:
        return p
    # src/xcode/cli/repl.py → src/xcode/.../repl.py
    return f"{parts[0]}/{parts[1]}/.../{parts[-1]}"


def brief_input(name: str, raw_input: ToolInput | str) -> str:
    """从工具输入中提取简短的人类可读摘要。"""
    if not isinstance(raw_input, dict):
        return single_line_preview(f"{name}: {raw_input}") if raw_input else name

    # ── 各工具类型特化格式化 ──
    if name in ("bash", "hypa_shell"):
        cmd = raw_input.get("command") or raw_input.get("input") or ""
        return single_line_preview(f"$ {cmd}") if cmd else name

    if name in ("read_file", "read", "hypa_read"):
        path = _shorten_path(
            str(
                raw_input.get(
                    "file_path", raw_input.get("path", raw_input.get("input", ""))
                )
            )
        )
        off = raw_input.get("offset")
        lim = raw_input.get("limit")
        suffix = f":{off}" if off else ""
        if lim:
            suffix += f"-{off + lim - 1}" if off else f" ({lim} lines)"
        return single_line_preview(f"read {path}{suffix}")

    if name in ("write_file", "write"):
        path = _shorten_path(str(raw_input.get("file_path", raw_input.get("path", ""))))
        content = str(raw_input.get("content", ""))
        lines = content.count("\n") + 1 if content else 0
        return single_line_preview(
            f"write {path}" + (f" ({lines} lines)" if lines else "")
        )

    if name in ("edit_file", "edit"):
        path = _shorten_path(str(raw_input.get("file_path", raw_input.get("path", ""))))
        return single_line_preview(f"edit {path}")

    if name in ("list_dir", "ls", "hypa_ls"):
        path = _shorten_path(str(raw_input.get("path", ".")))
        return single_line_preview(f"ls {path}")

    if name in ("glob_files", "find", "hypa_find"):
        pattern = str(raw_input.get("pattern", raw_input.get("path", "*")))
        path = _shorten_path(str(raw_input.get("path", ".")))
        return single_line_preview(
            f"glob {pattern}" + (f" in {path}" if path != "." else "")
        )

    if name in ("grep_search", "grep", "hypa_grep"):
        pattern = str(raw_input.get("pattern", ""))
        path = _shorten_path(str(raw_input.get("path", raw_input.get("include", "."))))
        return single_line_preview(
            f"grep /{pattern}/" + (f" in {path}" if path != "." else "")
        )

    if name == "todowrite":
        todos = _todo_items(raw_input)
        return f"todo list ({len(todos)})" if todos else name

    if name == "subagent":
        tasks = _subagent_tasks(raw_input)
        if tasks:
            return f"subagent tasks ({len(tasks)})"
        desc = raw_input.get("description", "")
        return single_line_preview(f"subagent: {desc}") if desc else name

    if name == "websearch":
        query = raw_input.get("query", raw_input.get("input", ""))
        return single_line_preview(f"web: {query}") if query else name

    if name == "webfetch":
        url = raw_input.get("url", raw_input.get("input", ""))
        return single_line_preview(f"fetch: {url}") if url else name

    # ── 兜底：key=value 列表 ──
    parts = [
        f"{k}={json.dumps(v, ensure_ascii=False)}"
        for k, v in raw_input.items()
        if v not in (None, "", [], {})
    ]
    if parts:
        return single_line_preview(f"{name}: {', '.join(parts)}")
    if raw_input:
        k, v = next(iter(raw_input.items()))
        return single_line_preview(f"{name}: {k}={v}")
    return name


def tool_call_text(name: str, label: str, raw_input: ToolInput | str) -> Text:
    """渲染工具调用摘要。"""
    if isinstance(raw_input, dict):
        if name == "todowrite":
            rendered = _todo_list_text(raw_input)
            if rendered is not None:
                return rendered
        if name == "subagent":
            rendered = _subagent_list_text(raw_input)
            if rendered is not None:
                return rendered
    return Text(f"  → {label}", style=CLI_COLOR_TOOL)


def _subagent_tasks(raw_input: ToolInput) -> list[dict[str, Any]]:
    tasks = raw_input.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    return [item for item in tasks if isinstance(item, dict)]


def _subagent_list_text(raw_input: ToolInput) -> Text | None:
    tasks = _subagent_tasks(raw_input)
    if not tasks:
        return None
    text = Text(f"  → Subagent tasks ({len(tasks)})", style=CLI_COLOR_TOOL)
    for index, task in enumerate(tasks, start=1):
        label = str(task.get("description", "")).strip() or f"task-{index}"
        agent_type = str(task.get("subagent_type", "coding")).strip() or "coding"
        text.append("\n")
        text.append(f"    [{index}] ", style=CLI_COLOR_DIM)
        text.append(label, style=CLI_COLOR_TOOL)
        text.append(f" [{agent_type}]", style=CLI_COLOR_DIM)
    return text


def _todo_items(raw_input: ToolInput) -> list[dict[str, Any]]:
    todos = raw_input.get("todos", [])
    if not isinstance(todos, list):
        return []
    return [item for item in todos if isinstance(item, dict)]


def _todo_list_text(raw_input: ToolInput) -> Text | None:
    todos = _todo_items(raw_input)
    if not todos:
        return None
    icons = {
        "completed": "✓",
        "in_progress": "◌",
        "pending": "·",
        "cancelled": "✕",
    }
    styles = {
        "completed": CLI_COLOR_SUCCESS,
        "in_progress": CLI_COLOR_WARNING,
        "pending": CLI_COLOR_TOOL,
        "cancelled": CLI_COLOR_ERROR,
    }
    text = Text(f"  → Todo list ({len(todos)})", style=CLI_COLOR_TOOL)
    for item in todos:
        status = str(item.get("status", ""))
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        priority = str(item.get("priority", "")).strip()
        suffix = f" [{priority}]" if priority else ""
        item_style = styles.get(status, CLI_COLOR_INFO)
        text.append("\n")
        text.append(f"    {icons.get(status, '?')} ", style=item_style)
        text.append(content, style=item_style)
        text.append(suffix, style=CLI_COLOR_DIM)
    return text


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
    if name == "websearch":
        query = raw_input.get("query") or raw_input.get("input")
        if query:
            return single_line_preview(f"Search web for {query}")
    if name == "webfetch":
        url = raw_input.get("url") or raw_input.get("input")
        if url:
            return single_line_preview(f"Fetch {url}")
    return single_line_preview(f"Run {name}")


def summarize_intents(intents: list[str]) -> str:
    if not intents:
        return "workspace"
    if len(intents) == 1:
        return intents[0]
    first = intents[0]
    return single_line_preview(f"{first} and {len(intents) - 1} more")


def event_to_dict(event: CodingAgentHarnessEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "step": event.step,
        "data": _event_payload(event),
        "correlation": asdict(event.correlation),
    }


def _event_payload(event: CodingAgentHarnessEvent) -> object:
    if isinstance(event, (TextDeltaStructuredEvent, ReasoningDeltaStructuredEvent)):
        return event.data
    if isinstance(event, MessageStartStructuredEvent):
        if isinstance(event.data, BaseModel):
            return event.data.model_dump()
        return None
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
        "termination_reason": event.data.termination_reason.value,
        "metrics": event.data.metrics,
        "watchdog_reason": event.data.watchdog_reason,
        "error_detail": event.data.error_detail,
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


def final_stop_reason(data: CodingAgentHarnessResult) -> str | None:
    if data.termination_reason.value == "step_limit":
        return "[stopped] step limit reached"
    if data.termination_reason.value == "watchdog":
        reason = data.watchdog_reason or "repeated tool calls detected"
        return f"[stopped] {reason}"
    if data.termination_reason.value == "cancelled":
        return "[stopped] cancelled"
    if data.termination_reason.value == "provider_error":
        reason = data.error_detail or "provider error"
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
