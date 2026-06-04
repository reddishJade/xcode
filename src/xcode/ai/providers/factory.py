from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values

from .protocol import ModelProvider
from .runtime import ProviderRuntime, RetryPolicy, RateLimitPolicy


def load_env_file(path: Path) -> dict[str, str | None]:
    return dotenv_values(path)


def get_config_value(name: str, env_files: tuple[Path, ...] = ()) -> str | None:
    if value := os.environ.get(name):
        return value
    for env_file in env_files:
        values = load_env_file(env_file)
        if value := values.get(name):
            return value
    return None


class ModelProfileProto(Protocol):
    @property
    def transport(self) -> str: ...
    @property
    def chat_model(self) -> str: ...
    @property
    def base_url(self) -> str: ...
    @property
    def api_key(self) -> str: ...
    @property
    def thinking(self) -> bool: ...
    @property
    def reasoning_effort(self) -> str | None: ...
    @property
    def clear_thinking(self) -> bool: ...
    @property
    def tool_stream(self) -> bool: ...
    @property
    def response_format(self) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class ModelProfileConfig:
    transport: str = "openai_chat"
    chat_model: str = ""
    base_url: str = ""
    api_key: str = ""
    thinking: bool = True
    reasoning_effort: str | None = None
    clear_thinking: bool = False
    tool_stream: bool = True
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderSettings:
    env_files: tuple[Path, ...]
    model_profiles: dict[str, ModelProfileProto]
    retry: RetryPolicy = RetryPolicy()
    rate_limit: RateLimitPolicy = RateLimitPolicy()


@dataclass(frozen=True)
class ProviderBundle:
    llm: ModelProvider
    llms: dict[str, ModelProvider]


def build_provider_bundle(settings: ProviderSettings) -> ProviderBundle:
    runtime = ProviderRuntime(
        retry=settings.retry,
        rate_limit=settings.rate_limit,
    )
    llms = _build_llm_profiles(settings, runtime)
    return ProviderBundle(
        llm=llms["main"],
        llms=llms,
    )


def _resolve_api_key(
    configured: str,
    profile_name: str,
    env_files: tuple[Path, ...],
    transport: str = "",
) -> str:
    if configured:
        return configured
    provider_candidates = {
        "anthropic_messages": ("ANTHROPIC_API_KEY",),
        "chatglm": ("CHATGLM_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
        "chatglm_chat": ("CHATGLM_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
        "deepseek_chat": ("DEEPSEEK_API_KEY",),
        "mimo_chat": ("MIMO_API_KEY",),
    }
    candidates = [
        f"{profile_name.upper()}_API_KEY",
        *provider_candidates.get(transport, ()),
        "OPENAI_API_KEY",
        "API_KEY",
    ]
    for name in candidates:
        value = get_config_value(name, env_files)
        if value:
            return value
    raise RuntimeError(
        f"Missing API key for '{profile_name}'. "
        f"Set via 'api_key' in profile config, or env var: "
        f"{' / '.join(candidates)}."
    )


def _build_llm_profiles(
    settings: ProviderSettings,
    runtime: ProviderRuntime,
) -> dict[str, ModelProvider]:
    profile_settings = dict(settings.model_profiles)
    profile_settings.setdefault("main", ModelProfileConfig())
    profile_settings.setdefault("subagent", profile_settings["main"])
    profile_settings.setdefault("judge", profile_settings["main"])
    profile_settings.setdefault("refiner", profile_settings["main"])
    return {
        name: _build_llm_profile(profile, name, settings.env_files, runtime)
        for name, profile in profile_settings.items()
    }


def _build_llm_profile(
    profile: ModelProfileProto,
    profile_name: str,
    env_files: tuple[Path, ...],
    runtime: ProviderRuntime,
) -> ModelProvider:
    transport = _canonical_transport(profile.transport)
    api_key = _resolve_api_key(profile.api_key, profile_name, env_files, transport)
    from . import PROVIDER_REGISTRY

    provider_cls = PROVIDER_REGISTRY.get(transport)
    if provider_cls is None:
        raise ValueError(
            f"Unknown transport '{profile.transport}'. Available: {', '.join(PROVIDER_REGISTRY)}"
        )
    kwargs: dict[str, object] = {
        "api_key": api_key,
        "model": profile.chat_model,
    }
    if transport == "anthropic_messages":
        return provider_cls(**kwargs)

    kwargs.update(
        {
            "base_url": profile.base_url,
            "thinking": profile.thinking,
            "runtime": runtime,
        }
    )
    if transport in {"openai_chat", "openai_responses", "deepseek_chat"}:
        kwargs["reasoning_effort"] = profile.reasoning_effort
    if transport == "chatglm_chat":
        kwargs["clear_thinking"] = profile.clear_thinking
        kwargs["tool_stream"] = profile.tool_stream
        kwargs["response_format"] = profile.response_format
    return provider_cls(**kwargs)


def _canonical_transport(transport: str) -> str:
    aliases = {
        "chat_completions": "openai_chat",
        "responses_stateful": "openai_responses",
        "chatglm": "chatglm_chat",
    }
    return aliases.get(transport, transport)
