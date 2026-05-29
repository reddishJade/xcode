from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .runtime import ProviderRuntime, RetryPolicy, RateLimitPolicy


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


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


@dataclass(frozen=True)
class ModelProfileConfig:
    transport: str = "chat_completions"
    chat_model: str = ""
    base_url: str = ""
    api_key: str = ""
    thinking: bool = True
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ProviderSettings:
    env_files: tuple[Path, ...]
    model_profiles: dict[str, ModelProfileProto]
    retry: RetryPolicy = RetryPolicy()
    rate_limit: RateLimitPolicy = RateLimitPolicy()


@dataclass(frozen=True)
class ProviderBundle:
    llm: Any  # ModelProvider
    llms: dict[str, Any]


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
) -> str:
    if configured:
        return configured
    candidates = [
        f"{profile_name.upper()}_API_KEY",
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
        f"{' / '.join(candidates[:3])}."
    )


def _build_llm_profiles(
    settings: ProviderSettings,
    runtime: ProviderRuntime,
) -> dict[str, Any]:
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
) -> Any:
    api_key = _resolve_api_key(profile.api_key, profile_name, env_files)
    from . import PROVIDER_REGISTRY

    provider_cls = PROVIDER_REGISTRY.get(profile.transport)
    if provider_cls is None:
        raise ValueError(
            f"Unknown transport '{profile.transport}'. Available: {', '.join(PROVIDER_REGISTRY)}"
        )
    return provider_cls(
        api_key=api_key,
        base_url=profile.base_url,
        model=profile.chat_model,
        thinking=profile.thinking,
        reasoning_effort=profile.reasoning_effort,
        runtime=runtime,
    )
