"""会话级 token 用量与成本累计。

统一口径：
- ↑ 非缓存输入 token（按输入价计费）
- ↓ 输出 token
- R 缓存读取 token（按 cache_read 价计费）
- W 缓存写入 token
- CH 最近一次请求的缓存命中率
- $ 累计估算成本（按模型注册表费率，含分时计价）
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from xcode.ai.cache import CacheUsage
from xcode.ai.models import get_model_cost


@dataclass
class UsageTotals:
    """累计 token 用量与估算成本。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    requests: int = 0

    def add(self, other: UsageTotals) -> UsageTotals:
        """合并两组累计值（如主 provider + 回退 provider）。"""
        return replace(
            self,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            requests=self.requests + other.requests,
        )


class UsageAccumulator:
    """按请求累计用量，按模型费率实时折算成本。"""

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._totals = UsageTotals()
        self._latest_hit_rate: float | None = None

    @property
    def totals(self) -> UsageTotals:
        return self._totals

    @property
    def cache_hit_rate(self) -> float | None:
        """最近一次请求的缓存命中率；尚无 usage 记录时为 None。"""
        return self._latest_hit_rate

    def record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cache_usage: CacheUsage | None = None,
        cache_write_tokens: int = 0,
        now: datetime | None = None,
    ) -> None:
        """记录一次请求的 usage。prompt_tokens 为含缓存命中的总输入。"""
        cache_usage = cache_usage or CacheUsage()
        billed_input = (
            cache_usage.miss_tokens
            if cache_usage.miss_tokens > 0
            else max(0, prompt_tokens - cache_usage.hit_tokens)
        )
        cost = _estimate_cost_usd(
            self._model,
            input_tokens=billed_input,
            output_tokens=completion_tokens,
            cache_read_tokens=cache_usage.hit_tokens,
            cache_write_tokens=cache_write_tokens,
            now=now,
        )
        self._totals.input_tokens += billed_input
        self._totals.output_tokens += completion_tokens
        self._totals.cache_read_tokens += cache_usage.hit_tokens
        self._totals.cache_write_tokens += cache_write_tokens
        self._totals.cost_usd += cost
        self._totals.requests += 1
        if prompt_tokens > 0:
            self._latest_hit_rate = cache_usage.hit_rate


def _estimate_cost_usd(
    model: str | None,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    now: datetime | None,
) -> float:
    """按模型注册表费率估算单次请求成本；未知模型返回 0。"""
    if not model:
        return 0.0
    rate = get_model_cost(model, now)
    if rate is None:
        return 0.0
    return (
        input_tokens * rate.input
        + output_tokens * rate.output
        + cache_read_tokens * rate.cache_read
        + cache_write_tokens * rate.cache_write
    ) / 1_000_000


def format_tokens(value: int) -> str:
    """紧凑 token 格式：47k、8.4M。"""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def format_usage_stats(
    totals: UsageTotals,
    cache_hit_rate: float | None = None,
) -> str:
    """生成底栏用量摘要，如 ↑80k ↓47k R8.4M CH99.8% $0.333。"""
    parts: list[str] = []
    if totals.input_tokens:
        parts.append(f"↑{format_tokens(totals.input_tokens)}")
    if totals.output_tokens:
        parts.append(f"↓{format_tokens(totals.output_tokens)}")
    if totals.cache_read_tokens:
        parts.append(f"R{format_tokens(totals.cache_read_tokens)}")
    if totals.cache_write_tokens:
        parts.append(f"W{format_tokens(totals.cache_write_tokens)}")
    if cache_hit_rate is not None and (
        totals.cache_read_tokens or totals.cache_write_tokens
    ):
        parts.append(f"CH{cache_hit_rate * 100:.1f}%")
    if totals.cost_usd > 0:
        parts.append(f"${totals.cost_usd:.3f}")
    return " ".join(parts)
