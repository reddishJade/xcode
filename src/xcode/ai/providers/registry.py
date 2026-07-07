"""Provider 注册表与工厂。

职责：
1. 维护 transport → provider class 的映射
2. 从 settings 构建 provider 实例
3. 解析 API key（环境变量回退）
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values

from xcode.ai.providers.base import ModelProvider
from xcode.ai.types import ProviderConfig

from ._runtime import ProviderRuntime, RetryPolicy, RateLimitPolicy
from .chatglm import ChatGLMProvider
from .deepseek import DeepSeekProvider
from .mimo import MiMoProvider
from .openai import OpenAIChatProvider


# ── 注册表 ──

PROVIDER_REGISTRY: dict[str, type] = {
    "openai_chat": OpenAIChatProvider,
    "chatglm_chat": ChatGLMProvider,
    "deepseek_chat": DeepSeekProvider,
    "mimo_chat": MiMoProvider,
}


# ── 配置类型 ──


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
    model_profiles: Mapping[str, ModelProfileProto]
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limit: RateLimitPolicy = field(default_factory=RateLimitPolicy)


@dataclass(frozen=True)
class ProviderBundle:
    llm: ModelProvider
    llms: dict[str, ModelProvider]


# ── 环境变量工具 ──


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


# ── Factory ──


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


def _build_llm_profiles(
    settings: ProviderSettings,
    runtime: ProviderRuntime,
) -> dict[str, ModelProvider]:
    """构造所有 model profile 的 provider 实例。"""
    profile_settings = dict(settings.model_profiles)
    profile_settings.setdefault("main", ModelProfileConfig())
    profile_settings.setdefault("subagent", profile_settings["main"])
    profile_settings.setdefault("judge", profile_settings["main"])
    profile_settings.setdefault("refiner", profile_settings["main"])
    return {
        name: _build_llm_profile(profile, name, settings.env_files)
        for name, profile in profile_settings.items()
    }


_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "chatglm": ("CHATGLM_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
    "chatglm_chat": ("CHATGLM_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
    "deepseek_chat": ("DEEPSEEK_API_KEY",),
    "mimo_chat": ("MIMO_API_KEY",),
}


def _resolve_api_key(
    configured: str,
    profile_name: str,
    env_files: tuple[Path, ...],
    transport: str = "",
) -> str:
    """按回退优先级解析 API key。"""
    if configured:
        return configured
    candidates = [
        f"{profile_name.upper()}_API_KEY",
        *_PROVIDER_ENV_VARS.get(transport, ()),
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


def _build_llm_profile(
    profile: ModelProfileProto,
    profile_name: str,
    env_files: tuple[Path, ...],
) -> ModelProvider:
    """构造单个 provider 实例。"""
    transport = profile.transport
    api_key = _resolve_api_key(
        profile.api_key, profile_name, env_files, transport
    )

    provider_cls = PROVIDER_REGISTRY.get(transport)
    if provider_cls is None:
        raise ValueError(
            f"Unknown transport '{profile.transport}'. "
            f"Available: {', '.join(PROVIDER_REGISTRY)}"
        )

    config = ProviderConfig(
        api_key=api_key,
        model=profile.chat_model,
        base_url=profile.base_url,
        thinking=profile.thinking,
        reasoning_effort=profile.reasoning_effort,
        response_format=profile.response_format,
        extra={
            "clear_thinking": profile.clear_thinking,
            "tool_stream": profile.tool_stream,
        },
    )
    return provider_cls(config)
