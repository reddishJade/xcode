"""模型注册表与模型选择语法解析。

合并原 registry.py 与 model_modes.py 的职责。
"""

from __future__ import annotations

from dataclasses import dataclass

from xcode.ai.types import Cost, Model

THINKING_LEVELS = frozenset(
    ("off", "none", "minimal", "low", "medium", "high", "xhigh", "max")
)


@dataclass(frozen=True)
class ModelMode:
    """解析后的模型选择。"""

    model: str
    provider: str | None = None
    thinking_level: str | None = None


# ── 模型注册表 ──

_MODELS: dict[str, dict[str, Model]] = {
    "openai": {
        "gpt-5.5": Model(
            id="gpt-5.5",
            name="GPT-5.5",
            api="openai-completions",
            provider="openai",
            reasoning=True,
            context_window=1_050_000,
            max_tokens=128_000,
            cost=Cost(input=5, output=30, cache_read=0.50),
        ),
        "gpt-5.4": Model(
            id="gpt-5.4",
            name="GPT-5.4",
            api="openai-completions",
            provider="openai",
            reasoning=True,
            context_window=1_050_000,
            max_tokens=128_000,
            cost=Cost(input=2.50, output=15, cache_read=0.25),
        ),
        "gpt-5.4-mini": Model(
            id="gpt-5.4-mini",
            name="GPT-5.4 Mini",
            api="openai-completions",
            provider="openai",
            reasoning=True,
            context_window=400_000,
            max_tokens=128_000,
            cost=Cost(input=0.75, output=4.50, cache_read=0.075),
        ),
    },
    "deepseek": {
        "deepseek-v4-pro": Model(
            id="deepseek-v4-pro",
            name="DeepSeek V4 Pro",
            api="deepseek-chat",
            provider="deepseek",
            reasoning=True,
            context_window=1_000_000,
            max_tokens=384_000,
            cost=Cost(input=0.435, output=0.87, cache_read=0.003625),
        ),
        "deepseek-v4-flash": Model(
            id="deepseek-v4-flash",
            name="DeepSeek V4 Flash",
            api="deepseek-chat",
            provider="deepseek",
            reasoning=True,
            context_window=1_000_000,
            max_tokens=384_000,
            cost=Cost(input=0.14, output=0.28, cache_read=0.0028),
        ),
    },
    "chatglm": {
        "glm-5.1": Model(
            id="glm-5.1",
            name="GLM-5.1",
            api="openai-completions",
            provider="chatglm",
            reasoning=True,
            context_window=200000,
            max_tokens=131072,
            cost=Cost(input=5, output=20),
        ),
        "glm-5": Model(
            id="glm-5",
            name="GLM-5",
            api="openai-completions",
            provider="chatglm",
            reasoning=True,
            context_window=200000,
            max_tokens=131072,
            cost=Cost(input=2, output=8),
        ),
        "glm-5-turbo": Model(
            id="glm-5-turbo",
            name="GLM-5 Turbo",
            api="openai-completions",
            provider="chatglm",
            reasoning=True,
            context_window=200000,
            max_tokens=131072,
            cost=Cost(input=0.5, output=2),
        ),
        "glm-4.7": Model(
            id="glm-4.7",
            name="GLM-4.7",
            api="openai-completions",
            provider="chatglm",
            reasoning=True,
            context_window=200000,
            max_tokens=131072,
            cost=Cost(input=1, output=4),
        ),
        "glm-4.7-flash": Model(
            id="glm-4.7-flash",
            name="GLM-4.7 Flash",
            api="openai-completions",
            provider="chatglm",
            reasoning=True,
            context_window=200000,
            max_tokens=131072,
            cost=Cost(input=0, output=0),
        ),
    },
    "mimo": {
        "mimo-v2.5-pro": Model(
            id="mimo-v2.5-pro",
            name="MiMo V2.5 Pro",
            api="mimo-chat",
            provider="mimo",
            reasoning=True,
            context_window=1_048_576,
            max_tokens=32_000,
            cost=Cost(input=0.435, output=0.87, cache_read=0.0036),
        ),
        "mimo-v2.5": Model(
            id="mimo-v2.5",
            name="MiMo V2.5",
            api="mimo-chat",
            provider="mimo",
            reasoning=True,
            context_window=262_144,
            max_tokens=32_000,
            cost=Cost(input=0.14, output=0.28, cache_read=0.0028),
        ),
    },
}


def get_providers() -> list[str]:
    return list(_MODELS)


def get_models(provider_name: str) -> list[Model]:
    return list(_MODELS.get(provider_name, {}).values())


def get_model(provider_name: str, model_id: str) -> Model | None:
    return _MODELS.get(provider_name, {}).get(model_id)


def resolve_model(provider_name: str, model_id: str) -> Model:
    """解析模型定义，支持三层回退策略。"""
    model = get_model(provider_name, model_id)
    if model is not None:
        return model
    provider = _MODELS.get(provider_name)
    if provider:
        fallback = next(iter(provider.values()))
        return fallback
    return Model(
        id=model_id, name=model_id, api="openai-completions", provider=provider_name
    )


# ── 模型选择语法解析 ──


def parse_model_mode(value: str) -> ModelMode:
    """解析 `provider/model:thinking_level` 模型选择语法。"""
    text = value.strip()
    if not text:
        raise ValueError("model must not be empty")

    provider: str | None = None
    model_part = text
    if "/" in text:
        provider_text, model_part = text.split("/", 1)
        provider = provider_text.strip() or None

    model = model_part
    thinking_level: str | None = None
    if ":" in model_part:
        model, level = model_part.rsplit(":", 1)
        thinking_level = level.strip().lower()
        if thinking_level not in THINKING_LEVELS:
            allowed = "/".join(sorted(THINKING_LEVELS))
            raise ValueError(
                f"invalid thinking level: {thinking_level}. Use {allowed}."
            )

    model = model.strip()
    if not model:
        raise ValueError("model must not be empty")
    return ModelMode(model=model, provider=provider, thinking_level=thinking_level)
