from __future__ import annotations

from typing import Protocol, TypeGuard

from xcode.ai.model_modes import parse_model_mode
from xcode.harness.observability import (
    FileGrantStore,
    InMemoryGrantStore,
    PermissionPolicy,
)
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
) -> None:
    parts = command.split(maxsplit=2)
    sub = parts[1] if len(parts) >= 2 else ""
    if sub == "clear" and session_grant_store is not None:
        session_grant_store.clear()
        print("Session permissions cleared.")
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
        lines.append(f"    {rule.tool} = {rule.decision}{input_part}{target_part}")
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
    lines = ["<permissions>"]
    has_any = False

    if static_policy is not None and static_policy.rules:
        has_any = True
        lines.append("  static:")
        for line in _format_rules(static_policy.rules):
            lines.append("    " + line)

        if static_policy.global_default is not None:
            has_any = True
            lines.append(f"  global_default: {static_policy.global_default}")

    if restricted_dirs:
        has_any = True
        lines.append("  restricted_dirs:")
        for d in restricted_dirs:
            lines.append(f"    {d}")

    if session_grant_store is not None:
        records = session_grant_store.records()
        if records:
            has_any = True
            lines.append("  session:")
            for rec in records:
                lines.append(_format_grant(rec))

    if permanent_grant_store is not None:
        records = permanent_grant_store.records()
        if records:
            has_any = True
            lines.append("  persistent:")
            for rec in records:
                lines.append(_format_grant(rec))

    if not has_any:
        lines.append("  (none)")
    lines.append("</permissions>")
    print("\n".join(lines))


def _format_grant(record: object) -> str:
    """格式化 GrantRecord 为可读行。"""
    cap = getattr(record, "capability", "?")
    op = getattr(record, "operation", "?")
    target = getattr(record, "target_pattern", "?")
    acc = getattr(record, "access", "?")
    dec = getattr(record, "decision", "?")
    return f"    {cap}/{op} on {target} ({acc}) = {dec}"


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
