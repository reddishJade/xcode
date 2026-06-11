from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

JsonObject = dict[str, object]

from .reasoning_effort import (
    reasoning_effort_levels_for_transport,
    supports_reasoning_effort,
)

CONFIG_FILENAME = "xcode.config.json"

PROVIDER_PRESETS: dict[str, Any] = {
    "openai": {
        "label": "OpenAI",
        "base_url": "",
        "models": ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"],
        "default_model": "gpt-5.5",
        "env_key": "OPENAI_API_KEY",
        "env_base_url": "OPENAI_BASE_URL",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "default_model": "deepseek-v4-flash",
        "env_key": "DEEPSEEK_API_KEY",
        "env_base_url": "DEEPSEEK_BASE_URL",
    },
    "mimo": {
        "label": "Xiaomi MiMo",
        "base_url": "https://api.xiaomimimo.com/v1",
        "models": ["mimo-v2.5-pro", "mimo-v2.5"],
        "default_model": "mimo-v2.5-pro",
        "env_key": "MIMO_API_KEY",
        "env_base_url": "MIMO_BASE_URL",
    },
    "chatglm": {
        "label": "ChatGLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "models": ["glm-4.7", "glm-4-flash", "glm-5", "glm-5.1"],
        "default_model": "glm-4.7",
        "env_key": "CHATGLM_API_KEY",
        "env_base_url": "CHATGLM_BASE_URL",
    },
}


def has_valid_config(project_root: Path) -> bool:
    """检查是否已有可用配置（config 文件或 .env）。"""
    config_path = project_root / CONFIG_FILENAME
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            profiles = data.get("provider", {}).get("model_profiles", {})
            for name in ("main",):
                profile = profiles.get(name, {})
                if profile.get("api_key"):
                    return True
        except (json.JSONDecodeError, OSError):
            pass

    env_paths = [
        project_root / ".env",
        project_root / "xcode" / ".env",
    ]
    for env_path in env_paths:
        env = dotenv_values(env_path)
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "MIMO_API_KEY",
            "CHATGLM_API_KEY",
            "ZHIPUAI_API_KEY",
            "BIGMODEL_API_KEY",
            "API_KEY",
        ):
            if env.get(key):
                return True

    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "MIMO_API_KEY",
        "CHATGLM_API_KEY",
        "ZHIPUAI_API_KEY",
        "BIGMODEL_API_KEY",
        "API_KEY",
    ):
        if os.environ.get(key):
            return True

    return False


def run_setup_wizard(project_root: Path) -> tuple[str, Path | None]:
    """首次启动配置向导。返回状态和临时配置路径。"""
    import questionary

    print()
    print("=" * 60)
    print("  Welcome to Xcode - AI Coding Agent")
    print("=" * 60)
    print()
    print("No API key configured. Let's set up your LLM provider.")
    print()

    choices = {preset["label"]: key for key, preset in PROVIDER_PRESETS.items()}
    provider_label = questionary.select("Select provider:", choices=list(choices)).ask()
    if provider_label is None:
        return ("cancelled", None)
    provider_key = choices[provider_label]
    preset = PROVIDER_PRESETS[provider_key]

    print()
    print(f"  Provider: {preset['label']}")

    env_key = preset["env_key"]
    env_val = os.environ.get(env_key) or ""

    api_key = questionary.text(
        "API Key:", default=env_val[:16] if env_val else "sk-"
    ).ask()
    if api_key is None:
        return ("cancelled", None)
    if not api_key:
        api_key = env_val

    default_base_url = preset["base_url"]
    env_base_url = os.environ.get(preset["env_base_url"], "")
    base_url = questionary.text(
        "Base URL:", default=env_base_url or default_base_url
    ).ask()
    if base_url is None:
        return ("cancelled", None)
    if not base_url:
        base_url = env_base_url or default_base_url

    model_default = preset["default_model"]
    model_choices = [*preset["models"], "Custom (enter name)"]
    model = questionary.select("Model:", choices=model_choices).ask()
    if model is None:
        return ("cancelled", None)
    if model == "Custom (enter name)":
        model = questionary.text("Model name:").ask()
        if model is None:
            return ("cancelled", None)
        if not model:
            model = model_default

    transport = "openai_chat"
    if provider_key == "deepseek":
        transport = "deepseek_chat"
    elif provider_key == "mimo":
        transport = "mimo_chat"
    elif provider_key == "chatglm":
        transport = "chatglm_chat"

    thinking_choice = questionary.select(
        "Thinking:", choices=["enabled", "disabled"], default="enabled"
    ).ask()
    if thinking_choice is None:
        return ("cancelled", None)
    thinking = thinking_choice == "enabled"
    reasoning_effort: str | None = None
    if thinking and supports_reasoning_effort(transport):
        effort_default = "high"
        effort = questionary.select(
            "Reasoning effort:",
            choices=list(reasoning_effort_levels_for_transport(transport)),
            default=effort_default,
        ).ask()
        if effort is None:
            return ("cancelled", None)
        reasoning_effort = effort

    # 确认
    print()
    print("  Summary:")
    print(f"    Provider  : {preset['label']}")
    print(f"    Model     : {model}")
    if base_url:
        print(f"    Base URL  : {base_url}")
    print(f"    Thinking  : {'enabled' if thinking else 'disabled'}")
    if reasoning_effort is not None:
        print(f"    Effort    : {reasoning_effort}")
    print(
        f"    API Key   : {'*' * max(0, len(api_key) - 4)}{api_key[-4:] if api_key else '(empty)'}"
    )
    print()
    confirm = questionary.confirm("Save this configuration?", default=True).ask()
    if confirm is None:
        return ("cancelled", None)

    config_data = {
        "provider": {
            "provider_type": provider_key,
            "model_profiles": {
                "main": {
                    "transport": transport,
                    "chat_model": model,
                    "base_url": base_url,
                    "api_key": api_key,
                    "thinking": thinking,
                    **(
                        {"reasoning_effort": reasoning_effort}
                        if reasoning_effort is not None
                        else {}
                    ),
                }
            },
        }
    }

    config_path = project_root / CONFIG_FILENAME
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    def deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    merged = deep_merge(existing, config_data)

    if confirm:
        config_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  Configuration saved to {CONFIG_FILENAME}")
        print()
        return ("saved", None)

    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="xcode_config_")
    os.close(fd)
    Path(tmp_path).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("  Running with temporary configuration (not saved).")
    print()
    return ("no_save", Path(tmp_path))
