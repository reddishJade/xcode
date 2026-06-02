from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import sys
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
    single_line_preview,
)
from .tool_catalog import build_tool_catalog
from xcode.harness.skills import ToolInput, ToolSpec, run_tool_result


_console = Console(file=sys.stdout)


def run_tool_command(command: str, app: Any) -> str:
    parts = command.split(maxsplit=2)
    if len(parts) < 2:
        return "usage: /tool NAME INPUT\n/tool list - show enabled tools by group"
    name = parts[1]
    registry: tuple[ToolSpec, ...] = tuple(getattr(app, "registry", ()) or ())
    if name == "list":
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

    tool_map = {tool.name: tool for tool in registry}
    selected_tool = tool_map.get(name)
    if selected_tool is None:
        return f"unknown tool: {name}"
    raw_input = parts[2] if len(parts) == 3 else ""
    try:
        action_input = parse_tool_input(selected_tool, raw_input)
    except ValueError as exc:
        return str(exc)
    result = run_tool_result(tool_map, name, action_input)
    return result.content


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


def brief_input(name: str, raw_input: Any) -> str:
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
    if isinstance(raw_input, str) and raw_input:
        return single_line_preview(f"{name}: {raw_input}")
    return name


def tool_intent(name: str, raw_input: Any) -> str:
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


def event_to_dict(event: Any) -> dict[str, Any]:
    data = event.data
    if is_dataclass(data) and not isinstance(data, type):
        payload = asdict(data)
    else:
        payload = data
    return {"type": event.type, "step": event.step, "data": payload}


def print_tool_call_rich(label: str, console: Console | None = None) -> None:
    target = console or _console
    target.print(Text(f"  • {label}", style=CLI_COLOR_TOOL))


def print_tool_result_rich(
    data: Any,
    verbose: bool,
    console: Console | None = None,
) -> None:
    if data.status == "ok" and not verbose:
        return
    target = console or _console
    border = {
        "error": CLI_COLOR_ERROR,
        "denied": CLI_COLOR_ERROR,
        "approval_required": CLI_COLOR_WARNING,
    }.get(data.status, CLI_COLOR_SUCCESS if data.status == "ok" else CLI_COLOR_INFO)
    mark = {"error": "✘", "denied": "⊘", "approval_required": "?"}.get(
        data.status, data.status
    )
    limit = 600 if verbose else 200
    summary = single_line_preview(str(data.content), width=limit)
    target.print(Text(f"  ← {mark} {summary}", style=border))


def final_stop_reason(data: Any) -> str | None:
    if getattr(data, "stopped_by_limit", False):
        return "[stopped] step limit reached"
    if getattr(data, "stopped_by_watchdog", False):
        reason = getattr(data, "watchdog_reason", "repeated tool calls detected")
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
