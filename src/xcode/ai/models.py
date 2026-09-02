"""模型注册表与模型选择语法解析。

合并原 registry.py 与 model_modes.py 的职责。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
            cost=Cost(
                input=1.32,
                output=3.96,
                cache_read=0.044,
                peak_hours=((1, 4), (6, 10)),
                off_peak_factor=0.5,
            ),
        ),
        "deepseek-v4-flash": Model(
            id="deepseek-v4-flash",
            name="DeepSeek V4 Flash",
            api="deepseek-chat",
            provider="deepseek",
            reasoning=True,
            context_window=1_000_000,
            max_tokens=384_000,
            cost=Cost(
                input=0.44,
                output=1.32,
                cache_read=0.014,
                peak_hours=((1, 4), (6, 10)),
                off_peak_factor=0.5,
            ),
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


def get_model_cost(model_name: str, now: datetime | None = None) -> Cost | None:
    """解析模型单价（美元/百万 token），声明了分时计价的模型按时刻折算费率。"""
    for provider_models in _MODELS.values():
        model = provider_models.get(model_name)
        if model is not None:
            return model.cost.effective(now)
    return None


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


def get_model_context_window(model: str | None) -> int | None:
    if not model:
        return None
    model_lower = model.lower()
    candidates = (
        (model_id, profile)
        for provider_models in _MODELS.values()
        for model_id, profile in provider_models.items()
    )
    for model_id, profile in sorted(
        candidates,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if model_id in model_lower:
            return profile.context_window
    return None


def effective_rollover_threshold(
    model: str | None,
    reserve_tokens: int = 0,
    fallback_threshold: int = 32000,
    trigger_ratio: float = 0.95,
    context_window_override: int | None = None,
) -> int:
    """计算自动换窗触发线。

    context_window_override 优先于模型注册表默认窗口，允许在配置中
    限制实际使用的上下文窗口（如 1M 窗口的模型只用 256K）。
    """
    context_window = (
        context_window_override
        if context_window_override is not None and context_window_override > 0
        else get_model_context_window(model)
    )
    if context_window is not None:
        ratio_threshold = int(context_window * trigger_ratio)
        reserved_threshold = context_window - max(reserve_tokens, 0)
        return max(1, min(ratio_threshold, reserved_threshold))
    return fallback_threshold


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
