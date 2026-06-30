from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Protocol, TypeGuard

import questionary

from .setup_wizard import CONFIG_FILENAME, _load_existing_config, _save_config
from xcode.ai.model_modes import parse_model_mode
from xcode.harness.observability import (
    FileGrantStore,
    InMemoryGrantStore,
    PermissionPolicy,
)
from xcode.harness.observability.permission_model import StaticPermission
from .reasoning_effort import (
    reasoning_effort_levels_for_transport,
)


class ModelControlApp(Protocol):
    def get_model_info(self) -> dict[str, str]: ...

    def set_model(
        self,
        *,
        model: str,
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str: ...


def _is_model_control_app(app: object) -> TypeGuard[ModelControlApp]:
    return hasattr(app, "get_model_info") and hasattr(app, "set_model")


def _model_info(app: object) -> dict[str, str]:
    return app.get_model_info() if _is_model_control_app(app) else {}


PERMISSION_TABS = (
    "Recently denied",
    "Allow",
    "Ask",
    "Deny",
    "Workspace",
)


def handle_permissions(
    command: str,
    session_grant_store: InMemoryGrantStore | None,
    permanent_grant_store: FileGrantStore | None,
    static_policy: PermissionPolicy | None = None,
    restricted_dirs: tuple[str, ...] = (),
    project_root: Path | None = None,
    app: object | None = None,
    store: object | None = None,
) -> None:
    parts = command.split(maxsplit=2)
    sub = parts[1] if len(parts) >= 2 else ""
    if sub == "clear" and session_grant_store is not None:
        session_grant_store.clear()
        print("Session permissions cleared.")
        return
    if sub and sub != "list":
        print("Usage: /permissions [list|clear]")
        return
    if sub != "list" and project_root is not None and sys.stdin.isatty():
        manage_permissions(
            project_root,
            session_grant_store,
            permanent_grant_store,
            static_policy,
            restricted_dirs,
            app=app,
            store=store,
        )
        return
    list_permissions(
        session_grant_store,
        permanent_grant_store,
        static_policy,
        restricted_dirs,
    )


def _format_rules(rules: tuple) -> list[str]:
    lines = []
    for rule in rules:
        input_part = _format_rule_input(rule)
        target_part = _format_rule_target(rule)
        lines.append(
            f"  - tool `{rule.tool}` -> {rule.decision}{input_part}{target_part}"
        )
    return lines


def _format_rule_input(rule: object) -> str:
    parts = []
    ic = getattr(rule, "input_contains", None)
    ip = getattr(rule, "input_prefix", None)
    ir = getattr(rule, "input_regex", None)
    if ip:
        parts.append(f"prefix={ip}")
    elif ic:
        parts.append(f"contains={ic}")
    if ir:
        parts.append(f"regex={ir}")
    return f" ({'; '.join(parts)})" if parts else ""


def _format_rule_target(rule: object) -> str:
    target = getattr(rule, "target", None)
    target_type = getattr(rule, "target_type", None)
    if target and target_type:
        return f" [{target_type}:{target}]"
    if target:
        return f" [{target}]"
    if target_type:
        return f" [{target_type}]"
    return ""


def list_permissions(
    session_grant_store: InMemoryGrantStore | None,
    permanent_grant_store: FileGrantStore | None,
    static_policy: PermissionPolicy | None = None,
    restricted_dirs: tuple[str, ...] = (),
) -> None:
    lines = [
        "Permission Status",
        "",
        "Decision order: restricted directories -> static deny/ask rules -> "
        "saved grants -> static allow/default -> execution-mode and safety checks.",
        "",
    ]

    if static_policy is not None:
        if static_policy.rules:
            lines.append(f"Static rules ({len(static_policy.rules)})")
            lines.extend(_format_rules(static_policy.rules))
            lines.append("")
        else:
            lines.append("Static rules: none")
        if static_policy.global_default is not None:
            lines.append(f"Default from config: {static_policy.global_default}")
        else:
            lines.append("Default from config: not set")
    else:
        lines.append("Static rules: none")
        lines.append("Default from config: not set")
    lines.append("Implicit fallback: allow unless another policy layer blocks the tool")
    lines.append("")

    if restricted_dirs:
        lines.append(f"Restricted directories ({len(restricted_dirs)})")
        for d in restricted_dirs:
            lines.append(f"  - `{d}`")
        lines.append("")
    else:
        lines.append("Restricted directories: none")
        lines.append("")

    if session_grant_store is not None:
        records = session_grant_store.records()
        if records:
            lines.append(f"Session grants ({len(records)})")
            for rec in records:
                lines.append(_format_grant(rec))
            lines.append("")
        else:
            lines.append("Session grants: none")
            lines.append("")

    if permanent_grant_store is not None:
        records = permanent_grant_store.records()
        if records:
            lines.append(f"Persistent grants ({len(records)})")
            for rec in records:
                lines.append(_format_grant(rec))
            lines.append("")
        else:
            lines.append("Persistent grants: none")
            lines.append("")

    lines.append("Use `/permissions clear` to remove session grants.")
    print("\n".join(lines))


def manage_permissions(
    project_root: Path,
    session_grant_store: InMemoryGrantStore | None,
    permanent_grant_store: FileGrantStore | None,
    static_policy: PermissionPolicy | None = None,
    restricted_dirs: tuple[str, ...] = (),
    *,
    app: object | None = None,
    store: object | None = None,
) -> None:
    """交互式管理权限规则。"""
    config_path = project_root / CONFIG_FILENAME
    config = _load_existing_config(config_path)
    print_permission_overview(config, session_grant_store, permanent_grant_store, store)

    while True:
        tab = questionary.select(
            "Permissions:",
            choices=[*PERMISSION_TABS, "Done"],
        ).ask()
        if tab in (None, "Done"):
            return
        if tab == "Recently denied":
            _render_recently_denied_tab(store)
            _pause_permission_tab()
            continue
        if tab == "Workspace":
            if _handle_workspace_tab(config, config_path, project_root, restricted_dirs):
                print(f"Saved to {config_path.name}. Restart Xcode to reload workspace roots.")
            continue
        decision = str(tab).lower()
        changed = _handle_rule_tab(config, config_path, decision)
        if changed:
            _apply_runtime_permission_policy(app, config)
            print(
                f"Saved to {config_path.name}. The current REPL policy was refreshed when possible."
            )


def print_permission_overview(
    config: dict[str, Any],
    session_grant_store: InMemoryGrantStore | None,
    permanent_grant_store: FileGrantStore | None,
    store: object | None,
) -> None:
    """打印类似权限面板的简短概览。"""
    rules = _security_rules(config)
    counts = {
        decision: sum(1 for rule in rules if rule.get("decision") == decision)
        for decision in ("allow", "ask", "deny")
    }
    recent_denied = _recently_denied(store)
    session_count = len(session_grant_store.records()) if session_grant_store else 0
    permanent_count = (
        len(permanent_grant_store.records()) if permanent_grant_store else 0
    )
    print("Permissions")
    print(
        "  "
        f"Recently denied {len(recent_denied)}   "
        f"Allow {counts['allow']}   "
        f"Ask {counts['ask']}   "
        f"Deny {counts['deny']}   "
        "Workspace"
    )
    print()
    print("  Xcode will not ask before using tools matched by Allow rules.")
    print(
        f"  Saved grants: session {session_count}, persistent {permanent_count}."
    )
    print()


def _permission_header(active_tab: str) -> str:
    labels = [
        f"[{tab}]" if tab == active_tab else tab
        for tab in PERMISSION_TABS
    ]
    return "Permissions  " + "   ".join(labels)


def _render_recently_denied_tab(store: object | None) -> None:
    print(_permission_header("Recently denied"))
    print()
    denied = _recently_denied(store)
    if not denied:
        print(
            "   No recent denials. Commands denied by the auto mode classifier will appear here."
        )
        print()
        return
    for index, item in enumerate(denied, start=1):
        print(f"   {index}. {item}")
    print()


def _handle_rule_tab(
    config: dict[str, Any],
    config_path: Path,
    decision: str,
) -> bool:
    rules = [rule for rule in _security_rules(config) if rule.get("decision") == decision]
    _render_rule_tab(decision, rules)
    choices = ["Add a new rule…", "Back"]
    action = questionary.select(
        "Select action:",
        choices=choices,
        default="Add a new rule…",
    ).ask()
    if action != "Add a new rule…":
        return False
    return add_permission_rule_interactive(
        config,
        config_path,
        default_decision=decision,
    )


def _render_rule_tab(decision: str, rules: list[dict[str, Any]]) -> None:
    active = decision.capitalize()
    print(_permission_header(active))
    print()
    guidance = {
        "allow": "   Xcode won't ask before using allowed tools.",
        "ask": "   Xcode will always ask for confirmation before using these tools.",
        "deny": "   Xcode will always reject requests to use denied tools.",
    }[decision]
    print(guidance)
    print("     1. Add a new rule…")
    for index, rule in enumerate(rules, start=2):
        print(f"     {index}. {format_permission_rule(rule)}")
    print()


def _handle_workspace_tab(
    config: dict[str, Any],
    config_path: Path,
    project_root: Path,
    restricted_dirs: tuple[str, ...],
) -> bool:
    _render_workspace_tab(config, project_root, restricted_dirs)
    action = questionary.select(
        "Select action:",
        choices=["Add directory…", "Back"],
        default="Add directory…",
    ).ask()
    if action != "Add directory…":
        return False
    directory = questionary.text("Directory:", default=str(project_root)).ask()
    if directory is None or not directory.strip():
        return False
    security = config.setdefault("security", {})
    roots = list(security.get("writable_roots", ()))
    roots.append(directory.strip())
    security["writable_roots"] = roots
    _save_config(config, config_path)
    return True


def _render_workspace_tab(
    config: dict[str, Any],
    project_root: Path,
    restricted_dirs: tuple[str, ...],
) -> None:
    print(_permission_header("Workspace"))
    print()
    print(
        "   Xcode can read files in the workspace, and make edits when auto-accept edits is on."
    )
    print()
    print(f"     -  {project_root} (Original working directory)")
    security = config.get("security", {})
    writable_roots: object = ()
    if isinstance(security, dict):
        writable_roots = security.get("writable_roots", ())
    if isinstance(writable_roots, list | tuple):
        for root in writable_roots:
            print(f"     -  {root}")
    for directory in restricted_dirs:
        print(f"     -  {directory} (restricted)")
    print("     1. Add directory…")
    print()


def _pause_permission_tab() -> None:
    questionary.select(
        "Select action:",
        choices=["Back"],
        default="Back",
    ).ask()


def add_permission_rule_interactive(
    config: dict[str, Any],
    config_path: Path,
    *,
    default_decision: str = "ask",
) -> bool:
    decision = questionary.select(
        "Rule decision:",
        choices=["allow", "ask", "deny"],
        default=default_decision,
    ).ask()
    if decision is None:
        return False

    rule = _prompt_custom_rule(str(decision))
    if rule is None:
        return False

    security = config.setdefault("security", {})
    rules = list(security.get("rules", ()))
    rules.append(rule)
    security["rules"] = rules
    _save_config(config, config_path)
    print(f"Added {format_permission_rule(rule)}")
    return True


def _prompt_custom_rule(decision: str) -> dict[str, Any] | None:
    tool = questionary.text("Tool name or pattern:", default="bash").ask()
    if tool is None or not tool.strip():
        return None
    rule: dict[str, Any] = {"tool": tool.strip(), "decision": decision}
    target_type = questionary.select(
        "Target type:",
        choices=["none", "command", "path", "mcp", "subagent", "skill"],
        default="none",
    ).ask()
    if target_type is None:
        return None
    if target_type != "none":
        rule["target_type"] = target_type
        target = questionary.text("Target pattern (optional):").ask()
        if target is None:
            return None
        if target.strip():
            rule["target"] = target.strip()
    input_kind = questionary.select(
        "Input match:",
        choices=["none", "prefix", "contains", "regex"],
        default="none",
    ).ask()
    if input_kind is None:
        return None
    if input_kind != "none":
        value = questionary.text(f"Input {input_kind}:").ask()
        if value is None or not value.strip():
            return None
        rule[f"input_{input_kind}"] = value.strip()
    return rule


def _apply_runtime_permission_policy(app: object | None, config: dict[str, Any]) -> None:
    if app is None:
        return
    agent = getattr(app, "agent", None)
    if agent is None:
        return
    policy = _policy_from_config(config)
    if hasattr(agent, "permission_policy"):
        agent.permission_policy = policy
    gate = getattr(agent, "_gate", None)
    if gate is not None and hasattr(gate, "_permission_policy"):
        gate._permission_policy = policy


def _policy_from_config(config: dict[str, Any]) -> PermissionPolicy | None:
    security = config.get("security", {})
    if not isinstance(security, dict):
        return None
    rules: list[StaticPermission] = []
    for raw in security.get("rules", ()):
        if not isinstance(raw, dict):
            continue
        try:
            rules.append(StaticPermission.model_validate(raw))
        except ValueError:
            continue
    global_default = security.get("global_default")
    if not isinstance(global_default, str):
        global_default = None
    if not rules and global_default is None:
        return None
    return PermissionPolicy(tuple(rules), global_default=global_default)


def _security_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    security = config.get("security", {})
    if not isinstance(security, dict):
        return []
    rules = security.get("rules", ())
    return [rule for rule in rules if isinstance(rule, dict)]


def _recently_denied(store: object | None) -> list[str]:
    if store is None:
        return []
    load_records = getattr(store, "load_records", None)
    if not callable(load_records):
        return []
    try:
        records = load_records()
    except (OSError, ValueError, TypeError):
        return []
    if not isinstance(records, list | tuple):
        return []
    denied: list[str] = []
    for record in reversed(records):
        content = getattr(record, "content", None)
        if not isinstance(content, dict):
            continue
        data = content.get("data")
        if not isinstance(data, dict):
            continue
        status = data.get("status")
        if status not in {"denied", "approval_required"}:
            continue
        text = str(data.get("content") or data.get("tool_use_id") or status)
        denied.append(text)
        if len(denied) >= 10:
            break
    return list(reversed(denied))


def format_permission_rule(rule: dict[str, Any]) -> str:
    decision = str(rule.get("decision", "ask"))
    label = _tool_label(rule)
    parts = [f"{label}: {decision}"]
    target_type = rule.get("target_type")
    target = rule.get("target")
    if target_type:
        parts.append(f"target_type={target_type}")
    if target:
        parts.append(f"target={target}")
    return " ".join(parts)


def _tool_label(rule: dict[str, Any]) -> str:
    tool = str(rule.get("tool", "*"))
    prefix = rule.get("input_prefix")
    contains = rule.get("input_contains")
    regex = rule.get("input_regex")
    if tool == "bash" and isinstance(prefix, str) and prefix:
        return f"Bash({prefix}*)"
    if tool == "bash" and isinstance(contains, str) and contains:
        return f"Bash(*{contains}*)"
    if tool == "bash" and isinstance(regex, str) and regex:
        return f"Bash(/{regex}/)"
    return tool


def _format_grant(record: object) -> str:
    """格式化 GrantRecord 为可读行。"""
    cap = getattr(record, "capability", "?")
    op = getattr(record, "operation", "?")
    target = getattr(record, "target_pattern", "?")
    acc = getattr(record, "access", "?")
    dec = getattr(record, "decision", "?")
    target_kind = getattr(record, "target_kind", "?")
    scope = getattr(record, "scope", "?")
    grant_id = getattr(record, "grant_id", "?")
    return (
        f"  - {dec} {cap}/{op} {acc} on {target_kind} `{target}` "
        f"({scope}, id={grant_id})"
    )


def handle_model_command(command: str, app: object) -> None:
    parts = command.split(maxsplit=3)
    if len(parts) == 1:
        info = _model_info(app)
        if info:
            print(f"  Model    : {info.get('model', 'unknown')}")
            print(f"  Base URL : {info.get('base_url', '')}")
        else:
            print("Model info not available.")
        return

    try:
        parsed = parse_model_mode(parts[1])
    except ValueError as exc:
        print(str(exc))
        return
    model_name = parsed.model
    profile = parsed.provider or "main"
    thinking: bool | None = None
    reasoning_effort: str | None = None
    if parsed.thinking_level is not None:
        if parsed.thinking_level == "off":
            thinking = False
            reasoning_effort = None
        else:
            thinking = True
            reasoning_effort = parsed.thinking_level

    if len(parts) >= 4 and parts[2] != "--thinking":
        print(f"Warning: unrecognized option '{parts[2]}' ignored.")
    if len(parts) >= 4 and parts[2] == "--thinking":
        level = parts[3].lower()
        if level not in (
            "off",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            print(
                f"Invalid thinking level: {level}. Use off/none/minimal/low/medium/high/xhigh/max."
            )
            return
        if level == "off":
            thinking = False
            reasoning_effort = None
        else:
            thinking = True
            reasoning_effort = level

    if not _is_model_control_app(app):
        print("Model switching is not supported in this app.")
        return

    try:
        new_model = app.set_model(
            model=model_name,
            profile=profile,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        print(f"Switched to model: {new_model}")
    except Exception as exc:
        print(f"Failed to switch model: {exc}")


def handle_effort_command(command: str, app: object) -> None:
    parts = command.split(maxsplit=1)
    info = _model_info(app)
    transport = info.get("transport", "") if info else ""
    supported_levels = reasoning_effort_levels_for_transport(transport)
    if len(parts) == 1:
        current = info.get("reasoning_effort", "not set") if info else "unknown"
        print(f"  Reasoning effort: {current}")
        if supported_levels:
            print(f"  Supported: {'/'.join(supported_levels)}")
        else:
            print("  Supported: not available for current provider")
        return

    level = parts[1].lower()
    if level == "off":
        if not _is_model_control_app(app):
            print("Model switching is not supported in this app.")
            return
        info = app.get_model_info()
        current_model = info.get("model", "unknown") if info else "unknown"
        try:
            app.set_model(model=current_model, thinking=False, reasoning_effort=None)
            print("Reasoning effort disabled.")
        except Exception as exc:
            print(f"Failed to set reasoning effort: {exc}")
        return

    if not supported_levels:
        print("Current provider does not support reasoning effort.")
        return
    if level not in supported_levels:
        print(
            f"Invalid effort level for {transport or 'current provider'}. "
            f"Use: {'/'.join(supported_levels)}"
        )
        return

    if not _is_model_control_app(app):
        print("Model switching is not supported in this app.")
        return

    info = app.get_model_info()
    current_model = info.get("model", "unknown") if info else "unknown"

    try:
        app.set_model(model=current_model, thinking=True, reasoning_effort=level)
        print(f"Reasoning effort set to: {level}")
    except Exception as exc:
        print(f"Failed to set reasoning effort: {exc}")


def handle_thinking_command(command: str, app: object) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        info = _model_info(app)
        thinking = info.get("thinking", "unknown") if info else "unknown"
        print(f"  Thinking: {thinking}")
        return

    state = parts[1].lower()
    if state not in ("on", "off"):
        print("Usage: /thinking on|off")
        return

    if not _is_model_control_app(app):
        print("Model switching is not supported in this app.")
        return

    info = app.get_model_info()
    current_model = info.get("model", "unknown") if info else "unknown"
    current_effort = info.get("reasoning_effort", "high") if info else "high"

    try:
        if state == "off":
            app.set_model(model=current_model, thinking=False, reasoning_effort=None)
            print("Thinking disabled.")
        else:
            app.set_model(
                model=current_model,
                thinking=True,
                reasoning_effort=current_effort,
            )
            print(f"Thinking enabled (effort: {current_effort}).")
    except Exception as exc:
        print(f"Failed to set thinking: {exc}")
