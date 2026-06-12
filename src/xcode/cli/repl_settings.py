from __future__ import annotations

from typing import Protocol, TypeGuard

from xcode.ai.model_modes import parse_model_mode
from xcode.harness.observability import (
    PermissionPolicy,
    PersistentPermissionStore,
    SessionPermissionPolicy,
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
    session_policy: SessionPermissionPolicy | None,
    persistent_store: PersistentPermissionStore | None,
    static_policy: PermissionPolicy | None = None,
    restricted_dirs: tuple[str, ...] = (),
    allowlist_mode: bool = False,
) -> None:
    parts = command.split(maxsplit=2)
    sub = parts[1] if len(parts) >= 2 else ""
    if sub == "revoke" and len(parts) >= 3 and persistent_store is not None:
        tool_name = parts[2]
        persistent_store.revoke(tool_name)
        print(f"Revoked persistent permission for: {tool_name}")
        return
    if sub == "clear" and session_policy is not None:
        session_policy.clear()
        print("Session permissions cleared.")
        return
    list_permissions(
        session_policy, persistent_store, static_policy, restricted_dirs, allowlist_mode
    )


def _format_rules(rules: tuple) -> list[str]:
    lines = []
    for rule in rules:
        input_part = f" (input: {rule.input_contains})" if rule.input_contains else ""
        lines.append(f"    {rule.tool} = {rule.decision}{input_part}")
    return lines


def list_permissions(
    session_policy: SessionPermissionPolicy | None,
    persistent_store: PersistentPermissionStore | None,
    static_policy: PermissionPolicy | None = None,
    restricted_dirs: tuple[str, ...] = (),
    allowlist_mode: bool = False,
) -> None:
    lines = ["<permissions>"]
    has_any = False

    if static_policy is not None and static_policy.rules:
        has_any = True
        lines.append("  static:")
        for line in _format_rules(static_policy.rules):
            lines.append("    " + line)

    if restricted_dirs:
        has_any = True
        lines.append("  restricted_dirs:")
        for d in restricted_dirs:
            lines.append(f"    {d}")

    if allowlist_mode:
        has_any = True
        lines.append("  mode: allowlist (non-allowed tools ask)")

    if session_policy is not None:
        rules = session_policy.rules
        if rules:
            has_any = True
            lines.append("  session:")
            for rule in rules:
                input_part = (
                    f" (input: {rule.input_contains})" if rule.input_contains else ""
                )
                lines.append(f"    {rule.tool} = {rule.decision}{input_part}")

    if persistent_store is not None:
        policy = persistent_store.load()
        if policy.rules:
            has_any = True
            lines.append("  persistent:")
            for rule in policy.rules:
                input_part = (
                    f" (input: {rule.input_contains})" if rule.input_contains else ""
                )
                lines.append(f"    {rule.tool} = {rule.decision}{input_part}")

    if not has_any:
        lines.append("  (none)")
    lines.append("</permissions>")
    print("\n".join(lines))


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
