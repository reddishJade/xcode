"""缓存统计与工具稳定化工具。

统一的缓存统计口径和工具 schema 规范化，确保跨 provider 一致性。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from xcode.ai.types import ToolDefinition


@dataclass(frozen=True)
class CacheUsage:
    """缓存使用统计。

    优先级规则：
    1. 原生字段优先（DeepSeek: prompt_cache_hit_tokens / prompt_cache_miss_tokens）
    2. 兼容字段回退（OpenAI/ChatGLM: prompt_tokens_details.cached_tokens）
    3. 命中率公式：hit / (hit + miss)，而非 hit / prompt_tokens
    """

    hit_tokens: int = 0
    miss_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """缓存总 token 数（命中 + 未命中）。"""
        return self.hit_tokens + self.miss_tokens

    @property
    def hit_rate(self) -> float:
        """缓存命中率。

        公式：hit / (hit + miss)
        原因：DeepSeek 原生 miss 口径不保证等于 prompt_tokens - hit。
        """
        total = self.total_tokens
        return round(self.hit_tokens / total, 4) if total > 0 else 0.0


def extract_cache_usage(response: Any) -> CacheUsage:
    """从 provider 响应中提取缓存统计。

    优先级：
    1. DeepSeek 原生：prompt_cache_hit_tokens / prompt_cache_miss_tokens
    2. 兼容字段：prompt_tokens_details.cached_tokens
    3. 推算 miss：prompt_tokens - cached_tokens（仅当原生 miss 不存在时）
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return CacheUsage()

    # 优先：DeepSeek 原生字段
    hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0

    if hit > 0:
        # 有原生 hit，但可能缺少原生 miss，从 prompt_tokens 推算
        if miss == 0:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            if prompt_tokens > hit:
                miss = prompt_tokens - hit
        return CacheUsage(hit_tokens=hit, miss_tokens=miss)

    # 回退：兼容字段
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cached = getattr(details, "cached_tokens", 0) or 0
        if cached > 0:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            return CacheUsage(
                hit_tokens=cached,
                miss_tokens=max(0, prompt_tokens - cached),
            )

    return CacheUsage()


def canonical_tool_schema(tool: ToolDefinition) -> dict[str, Any]:
    """规范化工具 schema（字典键排序）。

    确保同一工具的 schema 在不同调用间字节稳定。
    """
    result: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "schema": tool.parameters,
    }
    if tool.builtin is not None:
        result["builtin"] = tool.builtin
    return _sort_dict_recursive(result)


def _sort_dict_recursive(obj: Any) -> Any:
    """递归排序字典键。"""
    if isinstance(obj, dict):
        return {k: _sort_dict_recursive(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_sort_dict_recursive(item) for item in obj]
    return obj


def canonical_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """按 name 排序并规范化工具列表。

    确保工具列表在不同调用间顺序稳定。
    """
    sorted_tools = sorted(tools, key=lambda t: t.name)
    return [canonical_tool_schema(t) for t in sorted_tools]


def tool_catalog_fingerprint(tools: list[ToolDefinition]) -> str:
    """计算工具集合指纹（SHA256）。

    用于检测工具 catalog 漂移。
    """
    canonical = canonical_tools(tools)
    serialized = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
