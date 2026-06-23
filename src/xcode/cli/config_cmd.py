"""`xcode config` 子命令：管理 provider 配置 profile。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .setup_wizard import (
    CONFIG_FILENAME,
    _load_existing_config,
    _save_config,
    _prompt_api_key,
    _prompt_base_url,
    _prompt_model,
    _prompt_thinking_config,
    _print_summary,
    _resolve_transport,
    _select_provider,
)


BOOL_FIELDS = frozenset({"thinking", "clear_thinking", "tool_stream"})

STRING_FIELDS = frozenset(
    {"transport", "chat_model", "base_url", "api_key", "reasoning_effort"}
)


def handle_config_command(args: Any, project_root: Path) -> None:
    action = args.config_action
    if action == "list":
        _cmd_list(project_root)
    elif action == "add":
        _cmd_add(project_root, args.name)
    elif action == "delete":
        _cmd_delete(project_root, args.name)
    elif action == "set":
        _cmd_set(project_root, args.name, args.field, args.value)


def _config_path(project_root: Path) -> Path:
    return project_root / CONFIG_FILENAME


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{'*' * max(0, len(key) - 4)}{key[-4:]}"


def _cmd_list(project_root: Path) -> None:
    config = _load_existing_config(_config_path(project_root))
    profiles = config.get("provider", {}).get("model_profiles", {})

    if not profiles:
        print(f"No profiles found in {CONFIG_FILENAME}.")
        return

    print(f"Profiles in {CONFIG_FILENAME}:\n")
    for name, profile in profiles.items():
        if isinstance(profile, str):
            print(f"  {name}: (inherits from main, model={profile})")
            continue
        print(f"  {name}:")
        for field in (
            "transport",
            "chat_model",
            "base_url",
            "api_key",
            "thinking",
            "reasoning_effort",
        ):
            val = profile.get(field)
            if val is None:
                continue
            if field == "api_key" and val:
                val = _mask_key(str(val))
            print(f"    {field:20s}: {val}")
        print()


def _cmd_add(project_root: Path, name: str) -> None:
    import questionary

    provider_result = _select_provider()
    if provider_result is None:
        return
    provider_key, preset = provider_result

    print(f"\n  Provider: {preset['label']}")

    api_key = _prompt_api_key(preset)
    if api_key is None:
        return

    base_url = _prompt_base_url(preset)
    if base_url is None:
        return

    model = _prompt_model(preset)
    if model is None:
        return

    transport = _resolve_transport(provider_key)

    thinking_result = _prompt_thinking_config(transport)
    if thinking_result is None:
        return
    thinking, reasoning_effort = thinking_result

    _print_summary(
        preset["label"], model, base_url, thinking, reasoning_effort, api_key
    )

    confirm = questionary.confirm("Save this configuration?", default=True).ask()
    if not confirm:
        return

    profile_data: dict[str, Any] = {
        "transport": transport,
        "chat_model": model,
        "base_url": base_url,
        "api_key": api_key,
        "thinking": thinking,
    }
    if reasoning_effort is not None:
        profile_data["reasoning_effort"] = reasoning_effort

    config_path = _config_path(project_root)
    existing = _load_existing_config(config_path)
    existing.setdefault("provider", {}).setdefault("model_profiles", {})[name] = (
        profile_data
    )
    _save_config(existing, config_path)
    print(f"  Profile '{name}' saved to {CONFIG_FILENAME}.")


def _cmd_delete(project_root: Path, name: str) -> None:
    import questionary

    config_path = _config_path(project_root)
    config = _load_existing_config(config_path)

    profiles = config.get("provider", {}).get("model_profiles", {})
    if name not in profiles:
        print(f"Profile '{name}' not found in {CONFIG_FILENAME}.")
        return

    if not questionary.confirm(f"Delete profile '{name}'?", default=False).ask():
        return

    del profiles[name]
    _save_config(config, config_path)
    print(f"  Profile '{name}' deleted from {CONFIG_FILENAME}.")


def _coerce_value(field: str, value: str) -> Any:
    if field in BOOL_FIELDS:
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Invalid bool value for '{field}': {value}")
    if value.lower() == "null":
        return None
    return value


def _cmd_set(project_root: Path, name: str, field: str, value: str) -> None:
    config_path = _config_path(project_root)
    config = _load_existing_config(config_path)

    profiles = config.setdefault("provider", {}).setdefault("model_profiles", {})
    if name not in profiles:
        print(f"Profile '{name}' not found in {CONFIG_FILENAME}.")
        return

    profile = profiles[name]
    if isinstance(profile, str):
        print(
            f"Profile '{name}' is a string alias (inherits from main). "
            f"Convert it to a full profile first with 'config set {name} chat_model={profile}'."
        )
        return

    try:
        coerced = _coerce_value(field, value)
    except ValueError as exc:
        print(str(exc))
        return

    if coerced is None:
        profile.pop(field, None)
    else:
        profile[field] = coerced

    _save_config(config, config_path)
    print(f"  {name}.{field} = {value}")
