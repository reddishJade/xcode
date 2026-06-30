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
        choice = questionary.select(
            "Permissions:",
            choices=[
                "Add a new rule...",
                "Allow rules",
                "Ask rules",
                "Deny rules",
                "Workspace",
                "Recently denied",
                "Done",
            ],
            default="Add a new rule...",
        ).ask()
        if choice in (None, "Done"):
            return
        if choice == "Add a new rule...":
            changed = add_permission_rule_interactive(config, config_path)
            if changed:
                _apply_runtime_permission_policy(app, config)
                print(
                    f"Saved to {config_path.name}. The current REPL policy was refreshed when possible."
                )
            continue
        if choice == "Workspace":
            _print_workspace_permissions(config, restricted_dirs)
            continue
        if choice == "Recently denied":
            _print_recently_denied(store)
            continue
        decision = choice.split()[0].lower()
        _print_rules_for_decision(_security_rules(config), decision)


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


def add_permission_rule_interactive(
    config: dict[str, Any],
    config_path: Path,
) -> bool:
    decision = questionary.select(
        "Rule decision:",
        choices=["allow", "ask", "deny"],
        default="ask",
    ).ask()
    if decision is None:
        return False

    template = questionary.select(
        "Rule template:",
        choices=[
            "Bash(git add *)",
            "Bash(python *)",
            "Bash(uv run *)",
            "PowerShell(uv run *)",
            "Custom...",
        ],
        default="Bash(uv run *)",
    ).ask()
    if template is None:
        return False

    rule = _rule_from_template(str(template), str(decision))
    if rule is None:
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


def _rule_from_template(template: str, decision: str) -> dict[str, Any] | None:
    templates = {
        "Bash(git add *)": {
            "tool": "bash",
            "input_prefix": "git add ",
            "target_type": "command",
        },
        "Bash(python *)": {
            "tool": "bash",
            "input_prefix": "python ",
            "target_type": "command",
        },
        "Bash(uv run *)": {
            "tool": "bash",
            "input_prefix": "uv run ",
            "target_type": "command",
        },
        "PowerShell(uv run *)": {
            "tool": "bash",
            "input_prefix": "uv run ",
            "target_type": "command",
        },
    }
    base = templates.get(template)
    if base is None:
        return None
    return {"decision": decision, **base}


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


def _print_rules_for_decision(rules: list[dict[str, Any]], decision: str) -> None:
    matched = [rule for rule in rules if rule.get("decision") == decision]
    title = decision.capitalize()
    if not matched:
        print(f"{title} rules: none")
        return
    print(f"{title} rules ({len(matched)})")
    for index, rule in enumerate(matched, start=1):
        print(f"  {index}. {format_permission_rule(rule)}")


def _print_workspace_permissions(
    config: dict[str, Any],
    restricted_dirs: tuple[str, ...],
) -> None:
    security = config.get("security", {})
    if not isinstance(security, dict):
        security = {}
    configured_restricted = security.get("restricted_dirs", ())
    if not isinstance(configured_restricted, list | tuple):
        configured_restricted = ()
    print("Workspace")
    print(f"  global_default: {security.get('global_default', 'not set')}")
    print(f"  approval_policy: {security.get('approval_policy', 'not set')}")
    print(f"  permission_mode: {security.get('permission_mode', 'not set')}")
    dirs = tuple(str(item) for item in configured_restricted) or restricted_dirs
    if not dirs:
        print("  restricted_dirs: none")
        return
    print("  restricted_dirs:")
    for directory in dirs:
        print(f"    - {directory}")


def _print_recently_denied(store: object | None) -> None:
    denied = _recently_denied(store)
    if not denied:
        print("Recently denied: none")
        return
    print(f"Recently denied ({len(denied)})")
    for index, item in enumerate(denied, start=1):
        print(f"  {index}. {item}")


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
