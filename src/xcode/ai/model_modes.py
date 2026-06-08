from __future__ import annotations

from dataclasses import dataclass

THINKING_LEVELS = frozenset(("off", "minimal", "low", "medium", "high", "xhigh"))


@dataclass(frozen=True)
class ModelMode:
    """解析后的模型选择。"""

    model: str
    provider: str | None = None
    thinking_level: str | None = None


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
