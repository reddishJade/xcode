"""Provider 工厂：从配置构造 provider 实例。"""

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
    """按回退优先级解析 API key。

    回退顺序设计原因：
    1. profile 显式配置优先（显式覆盖）
    2. profile_name 专属环境变量（{PROFILE}_API_KEY）
    3. provider 官方环境变量（各家官方 SDK 约定）
    4. OPENAI_API_KEY（OpenAI-compatible 兼容层通用约定）
    5. API_KEY（通用回退）
    """
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
    """构造所有 model profile 的 provider 实例。

    确保四个标准 profile 存在的原因：
    - main: 主循环默认模型，所有场景都需要
    - subagent: 子任务代理模型，未配置时继承 main（成本控制）
    - judge: 评审/验证模型，未配置时继承 main（质量保证）
    - refiner: 精化/重写模型，未配置时继承 main（输出优化）

    用户只需配置 main，其余三个按需覆盖。
    """
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
    transport = profile.transport
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
    # Anthropic Messages API 特殊处理原因：
    # 1. 官方 SDK 无 base_url 参数（非 OpenAI-compatible）
    # 2. 无 thinking/reasoning_effort 等扩展参数
    if transport == "anthropic_messages":
        return provider_cls(**kwargs)

    kwargs.update(
        {
            "base_url": profile.base_url,
            "thinking": profile.thinking,
            "runtime": runtime,
        }
    )
    if transport in {"openai_chat", "deepseek_chat"}:
        kwargs["reasoning_effort"] = profile.reasoning_effort
    if transport in {"openai_chat"}:
        kwargs["response_format"] = profile.response_format
    if transport == "chatglm_chat":
        kwargs["clear_thinking"] = profile.clear_thinking
        kwargs["tool_stream"] = profile.tool_stream
        kwargs["response_format"] = profile.response_format
    if transport == "deepseek_chat":
        kwargs["response_format"] = profile.response_format
    return provider_cls(**kwargs)
