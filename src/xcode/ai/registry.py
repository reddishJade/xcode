from __future__ import annotations

from .types import Cost, Model

"""内置模型注册表：预定义常见模型的元数据。"""

_MODELS: dict[str, dict[str, Model]] = {
    "openai": {
        "gpt-5.5": Model(
            id="gpt-5.5",
            name="GPT-5.5",
            api="openai-completions",
            provider="openai",
            reasoning=True,
            context_window=1_000_000,
            max_tokens=131072,
            cost=Cost(input=5, output=30),
        ),
        "gpt-5.4": Model(
            id="gpt-5.4",
            name="GPT-5.4",
            api="openai-completions",
            provider="openai",
            reasoning=True,
            context_window=1_000_000,
            max_tokens=131072,
            cost=Cost(input=2.50, output=15),
        ),
        "gpt-5.4-mini": Model(
            id="gpt-5.4-mini",
            name="GPT-5.4 Mini",
            api="openai-completions",
            provider="openai",
            reasoning=True,
            context_window=400_000,
            max_tokens=131072,
            cost=Cost(input=0.75, output=4.50),
        ),
    },
    "deepseek": {
        "deepseek-v4-pro": Model(
            id="deepseek-v4-pro",
            name="DeepSeek V4 Pro",
            api="deepseek-chat",
            provider="deepseek",
            reasoning=True,
            context_window=131072,
            max_tokens=16384,
            cost=Cost(input=2, output=8),
        ),
        "deepseek-v4-flash": Model(
            id="deepseek-v4-flash",
            name="DeepSeek V4 Flash",
            api="deepseek-chat",
            provider="deepseek",
            reasoning=True,
            context_window=131072,
            max_tokens=16384,
            cost=Cost(input=0.3, output=1.2),
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
            context_window=131072,
            max_tokens=16384,
            cost=Cost(input=2, output=8),
        ),
        "mimo-v2.5": Model(
            id="mimo-v2.5",
            name="MiMo V2.5",
            api="mimo-chat",
            provider="mimo",
            reasoning=True,
            context_window=131072,
            max_tokens=16384,
            cost=Cost(input=1, output=4),
        ),
        "mimo-v2-flash": Model(
            id="mimo-v2-flash",
            name="MiMo V2 Flash",
            api="mimo-chat",
            provider="mimo",
            reasoning=False,
            context_window=131072,
            max_tokens=16384,
            cost=Cost(input=0.15, output=0.6),
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
