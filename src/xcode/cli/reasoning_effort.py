"""推理 effort 相关的命令和配置辅助函数。"""

from __future__ import annotations

from collections.abc import Iterable

EFFORT_COMMAND_LEVELS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

OPENAI_CHAT_EFFORT_LEVELS: tuple[str, ...] = EFFORT_COMMAND_LEVELS
DEEPSEEK_CHAT_EFFORT_LEVELS: tuple[str, ...] = ("off", "high", "max")

SUPPORTED_EFFORT_TRANSPORTS: frozenset[str] = frozenset(
    {"openai_chat", "deepseek_chat"}
)


def supports_reasoning_effort(transport: str) -> bool:
    """判断指定 transport 是否支持 reasoning_effort。"""
    return transport in SUPPORTED_EFFORT_TRANSPORTS


def reasoning_effort_levels_for_transport(transport: str) -> tuple[str, ...]:
    """返回指定 transport 可用的 effort 选项。"""
    if transport == "deepseek_chat":
        return DEEPSEEK_CHAT_EFFORT_LEVELS
    if transport == "openai_chat":
        return OPENAI_CHAT_EFFORT_LEVELS
    return ()


def normalize_reasoning_effort_options(
    options: Iterable[str] | None,
) -> tuple[str, ...]:
    """将 effort 选项标准化为去重后的元组。"""
    if options is None:
        return ()
    normalized: list[str] = []
    for option in options:
        text = option.strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)
