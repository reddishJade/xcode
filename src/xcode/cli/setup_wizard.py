from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from xcode.ai.providers.factory import load_env_file

CONFIG_FILENAME = "xcode.config.json"

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "base_url": "",
        "models": ["gpt-4o", "gpt-4o-mini", "o3", "o4-mini"],
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "env_base_url": "OPENAI_BASE_URL",
    },
    "anthropic": {
        "label": "Anthropic",
        "base_url": "",
        "models": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
        "default_model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "env_base_url": "ANTHROPIC_BASE_URL",
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
        env = load_env_file(env_path)
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


def run_setup_wizard(project_root: Path) -> None:
    """首次启动配置向导。"""
    print()
    print("=" * 60)
    print("  Welcome to Xcode - AI Coding Agent")
    print("=" * 60)
    print()
    print("No API key configured. Let's set up your LLM provider.")
    print()

    providers_by_num: dict[int, str] = {}
    num = 1
    for key, preset in PROVIDER_PRESETS.items():
        print(f"  [{num}] {preset['label']}")
        providers_by_num[num] = key
        num += 1
    print()

    while True:
        raw = input(f"Select provider (1-{len(providers_by_num)}): ").strip()
        try:
            choice = int(raw)
            if choice in providers_by_num:
                break
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(providers_by_num)}.")

    provider_key = providers_by_num[choice]
    preset = PROVIDER_PRESETS[provider_key]

    print()
    print(f"  Provider: {preset['label']}")

    env_key = preset["env_key"]
    env_val = os.environ.get(env_key) or ""
    default_key = env_val or ""

    api_key = input(f"  API Key [{default_key[:4]}... if set in env]: ").strip()
    if not api_key:
        api_key = env_val

    default_base_url = preset["base_url"]
    env_base_url = os.environ.get(preset["env_base_url"], "")
    base_url_hint = env_base_url or default_base_url or "(default)"
    base_url = input(f"  Base URL [{base_url_hint}]: ").strip()
    if not base_url:
        base_url = env_base_url or default_base_url

    default_model = preset["default_model"]
    model_hint = ", ".join(preset["models"])
    model = input(f"  Model [{default_model}] ({model_hint}): ").strip()
    if not model:
        model = default_model

    # 确认
    print()
    print("  Summary:")
    print(f"    Provider  : {preset['label']}")
    print(f"    Model     : {model}")
    if base_url:
        print(f"    Base URL  : {base_url}")
    print(
        f"    API Key   : {'*' * max(0, len(api_key) - 4)}{api_key[-4:] if api_key else '(empty)'}"
    )
    print()
    confirm = input("  Save this configuration? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("  Setup skipped. You can configure later by editing xcode.config.json.")
        return

    transport = "openai_chat"
    if provider_key == "anthropic":
        transport = "anthropic_messages"
    elif provider_key == "deepseek":
        transport = "deepseek_chat"
    elif provider_key == "mimo":
        transport = "mimo_chat"
    elif provider_key == "chatglm":
        transport = "chatglm_chat"

    config_data = {
        "provider": {
            "provider_type": provider_key,
            "model_profiles": {
                "main": {
                    "transport": transport,
                    "chat_model": model,
                    "base_url": base_url,
                    "api_key": api_key,
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

    config_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Configuration saved to {CONFIG_FILENAME}")
    print()
