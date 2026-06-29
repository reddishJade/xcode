"""`xcode config` 子命令：管理 provider 配置 profile。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import questionary

from .reasoning_effort import (
    reasoning_effort_levels_for_transport,
    supports_reasoning_effort,
)
from .setup_wizard import (
    CONFIG_FILENAME,
    PROVIDER_PRESETS,
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

TRANSPORT_TO_PROVIDER_KEY: dict[str, str] = {
    "openai_chat": "openai",
    "deepseek_chat": "deepseek",
    "mimo_chat": "mimo",
    "chatglm_chat": "chatglm",
}

BOOL_FIELDS = frozenset({"thinking", "clear_thinking", "tool_stream"})


def handle_config_command(args: Any, project_root: Path) -> None:
    config_path = args.config or project_root / CONFIG_FILENAME
    action = args.config_action
    if action == "list":
        _cmd_list(config_path)
    elif action == "add":
        _cmd_add(config_path, args.name)
    elif action == "edit":
        _cmd_edit(config_path, args.name)
    elif action == "delete":
        _cmd_delete(config_path, args.name)
    elif action == "set":
        _cmd_set(config_path, args.name, args.field, args.value)


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{'*' * max(0, len(key) - 4)}{key[-4:]}"


def _fmt(filename: str) -> str:
    return f"  (in {filename})"


def _cmd_list(config_path: Path) -> None:
    config = _load_existing_config(config_path)
    profiles = config.get("provider", {}).get("model_profiles", {})

    if not profiles:
        print(f"No profiles found in {config_path.name}.")
        return

    print(f"Profiles in {config_path.name}:\n")
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


def _build_profile_data(
    provider_key: str,
    transport: str,
    model: str,
    base_url: str,
    api_key: str,
    thinking: bool,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "transport": transport,
        "chat_model": model,
        "base_url": base_url,
        "api_key": api_key,
        "thinking": thinking,
    }
    if reasoning_effort is not None:
        data["reasoning_effort"] = reasoning_effort
    return data


def _prompt_interactive_profile(
    provider_key: str, preset: dict[str, Any]
) -> dict[str, Any] | None:
    """Run the interactive prompts and return profile data, or None if cancelled."""
    api_key = _prompt_api_key(preset)
    if api_key is None:
        return None

    base_url = _prompt_base_url(preset)
    if base_url is None:
        return None

    model = _prompt_model(preset)
    if model is None:
        return None

    transport = _resolve_transport(provider_key)

    thinking_result = _prompt_thinking_config(transport)
    if thinking_result is None:
        return None
    thinking, reasoning_effort = thinking_result

    _print_summary(
        preset["label"], model, base_url, thinking, reasoning_effort, api_key
    )

    confirm = questionary.confirm("Save this configuration?", default=True).ask()
    if not confirm:
        return None

    return _build_profile_data(
        provider_key, transport, model, base_url, api_key, thinking, reasoning_effort
    )


def _cmd_add(config_path: Path, name: str) -> None:
    provider_result = _select_provider()
    if provider_result is None:
        return
    provider_key, preset = provider_result

    print(f"\n  Provider: {preset['label']}")

    profile_data = _prompt_interactive_profile(provider_key, preset)
    if profile_data is None:
        return

    existing = _load_existing_config(config_path)
    existing.setdefault("provider", {}).setdefault("model_profiles", {})[name] = (
        profile_data
    )
    _save_config(existing, config_path)
    print(f"  Profile '{name}' saved.{_fmt(config_path.name)}")


def _find_preset_for_transport(transport: str) -> tuple[str, dict[str, Any]] | None:
    provider_key = TRANSPORT_TO_PROVIDER_KEY.get(transport)
    if provider_key is None:
        return None

    preset = PROVIDER_PRESETS.get(provider_key)
    if preset is None:
        return None
    return provider_key, preset


def _cmd_edit(config_path: Path, name: str) -> None:
    config = _load_existing_config(config_path)
    profiles = config.get("provider", {}).get("model_profiles", {})

    if name not in profiles:
        print(f"Profile '{name}' not found.{_fmt(config_path.name)}")
        return

    old = profiles[name]
    if isinstance(old, str):
        print(
            f"Profile '{name}' is a string alias (inherits from main). "
            f"Use 'config set {name} ...' or 'config add {name}' to convert it."
        )
        return

    transport: str = old.get("transport", "openai_chat")
    current_model: str = old.get("chat_model", "")
    current_base_url: str = old.get("base_url", "")
    current_api_key: str = old.get("api_key", "")
    current_thinking: bool = old.get("thinking", True)
    current_effort: str | None = old.get("reasoning_effort")

    provider_info = _find_preset_for_transport(transport)
    if provider_info is None:
        print(f"Unknown transport '{transport}'. Edit cancelled.")
        return
    provider_key, preset = provider_info

    print(f"\n  Editing profile '{name}' ({preset['label']}/{transport})")
    print(f"  Current model: {current_model}")
    print()

    new_api_key = questionary.text(
        "API Key (leave empty to keep current):",
        default="",
    ).ask()
    if new_api_key is None:
        return
    if not new_api_key:
        new_api_key = current_api_key
        print("  (keeping existing API key)")

    new_base_url = questionary.text(
        "Base URL:", default=current_base_url or preset["base_url"]
    ).ask()
    if new_base_url is None:
        return

    model_choices = [*(preset.get("models", [])), "Custom (enter name)"]
    default_model = current_model or preset["default_model"]
    new_model = questionary.select(
        "Model:", choices=model_choices, default=default_model
    ).ask()
    if new_model is None:
        return
    if new_model == "Custom (enter name)":
        new_model = questionary.text("Model name:", default=default_model).ask()
        if new_model is None:
            return
        if not new_model:
            new_model = default_model

    new_thinking = questionary.select(
        "Thinking:",
        choices=["enabled", "disabled"],
        default="enabled" if current_thinking else "disabled",
    ).ask()
    if new_thinking is None:
        return
    new_thinking_bool = new_thinking == "enabled"
    new_effort: str | None = current_effort
    if new_thinking_bool and supports_reasoning_effort(transport):
        levels = list(reasoning_effort_levels_for_transport(transport))
        effort_default = current_effort if current_effort in levels else "high"
        new_effort = questionary.select(
            "Reasoning effort:", choices=levels, default=effort_default
        ).ask()
        if new_effort is None:
            return
    elif not new_thinking_bool:
        new_effort = None

    _print_summary(
        preset["label"],
        new_model,
        new_base_url,
        new_thinking_bool,
        new_effort,
        new_api_key,
    )

    confirm = questionary.confirm("Save changes?", default=True).ask()
    if not confirm:
        return

    profile_data = _build_profile_data(
        provider_key,
        transport,
        new_model,
        new_base_url,
        new_api_key,
        new_thinking_bool,
        new_effort,
    )

    profiles[name] = profile_data
    _save_config(config, config_path)
    print(f"  Profile '{name}' updated.{_fmt(config_path.name)}")


def _cmd_delete(config_path: Path, name: str) -> None:
    config = _load_existing_config(config_path)

    profiles = config.get("provider", {}).get("model_profiles", {})
    if name not in profiles:
        print(f"Profile '{name}' not found.{_fmt(config_path.name)}")
        return

    if not questionary.confirm(f"Delete profile '{name}'?", default=False).ask():
        return

    del profiles[name]
    _save_config(config, config_path)
    print(f"  Profile '{name}' deleted.{_fmt(config_path.name)}")


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


def _cmd_set(config_path: Path, name: str, field: str, value: str) -> None:
    config = _load_existing_config(config_path)

    profiles = config.setdefault("provider", {}).setdefault("model_profiles", {})
    if name not in profiles:
        print(f"Profile '{name}' not found.{_fmt(config_path.name)}")
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
