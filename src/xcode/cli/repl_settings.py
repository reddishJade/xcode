from __future__ import annotations

from typing import Protocol, TypeGuard

from xcode.harness.observability import (
    PersistentPermissionStore,
    SessionPermissionPolicy,
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
    list_permissions(session_policy, persistent_store)


def list_permissions(
    session_policy: SessionPermissionPolicy | None,
    persistent_store: PersistentPermissionStore | None,
) -> None:
    lines = ["<permissions>"]
    if session_policy is not None:
        rules = session_policy.rules
        if rules:
            lines.append("  session:")
            for rule in rules:
                input_contains = (
                    f" (input: {rule.input_contains})" if rule.input_contains else ""
                )
                lines.append(f"    {rule.tool} = {rule.decision}{input_contains}")
    if persistent_store is not None:
        policy = persistent_store.load()
        if policy.rules:
            lines.append("  persistent:")
            for rule in policy.rules:
                input_contains = (
                    f" (input: {rule.input_contains})" if rule.input_contains else ""
                )
                lines.append(f"    {rule.tool} = {rule.decision}{input_contains}")
    if len(lines) == 1:
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

    model_name = parts[1]
    thinking: bool | None = None
    reasoning_effort: str | None = None

    if len(parts) >= 4 and parts[2] != "--thinking":
        print(f"Warning: unrecognized option '{parts[2]}' ignored.")
    if len(parts) >= 4 and parts[2] == "--thinking":
        level = parts[3].lower()
        if level not in ("off", "minimal", "low", "medium", "high", "xhigh"):
            print(
                f"Invalid thinking level: {level}. Use off/minimal/low/medium/high/xhigh."
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
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        print(f"Switched to model: {new_model}")
    except Exception as exc:
        print(f"Failed to switch model: {exc}")


def handle_effort_command(command: str, app: object) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        info = _model_info(app)
        current = info.get("reasoning_effort", "not set") if info else "unknown"
        print(f"  Reasoning effort: {current}")
        return

    level = parts[1].lower()
    if level not in ("off", "minimal", "low", "medium", "high", "xhigh"):
        print("Invalid effort level. Use: off/minimal/low/medium/high/xhigh")
        return

    if not _is_model_control_app(app):
        print("Model switching is not supported in this app.")
        return

    info = app.get_model_info()
    current_model = info.get("model", "unknown") if info else "unknown"

    try:
        if level == "off":
            app.set_model(model=current_model, thinking=False, reasoning_effort=None)
            print("Reasoning effort disabled.")
        else:
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
