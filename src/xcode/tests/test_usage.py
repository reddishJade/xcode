"""用量累计、成本估算与摘要格式的单元测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from xcode.ai.cache import CacheUsage
from xcode.ai.usage import (
    UsageAccumulator,
    UsageTotals,
    format_tokens,
    format_usage_stats,
)


def _off_peak_now() -> datetime:
    """deepseek 高峰时段为 UTC 1-4、6-10，选 12 点为非高峰。"""
    return datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _peak_now() -> datetime:
    return datetime(2025, 1, 1, 3, 0, tzinfo=UTC)


class TestUsageAccumulator:
    def test_deepseek_style_billing_with_miss(self) -> None:
        acc = UsageAccumulator("deepseek-v4-flash")
        acc.record(
            prompt_tokens=1_000_000,
            completion_tokens=100_000,
            cache_usage=CacheUsage(hit_tokens=900_000, miss_tokens=100_000),
            now=_peak_now(),
        )
        totals = acc.totals
        assert totals.input_tokens == 100_000
        assert totals.output_tokens == 100_000
        assert totals.cache_read_tokens == 900_000
        assert totals.requests == 1
        # 峰值费率：0.44 / 1.32 / 0.014 每百万
        expected = (100_000 * 0.44 + 100_000 * 1.32 + 900_000 * 0.014) / 1e6
        assert totals.cost_usd == round(expected, 6)
        assert acc.cache_hit_rate == 0.9

    def test_off_peak_half_price(self) -> None:
        acc = UsageAccumulator("deepseek-v4-flash")
        acc.record(
            prompt_tokens=1_000_000,
            completion_tokens=100_000,
            cache_usage=CacheUsage(hit_tokens=900_000, miss_tokens=100_000),
            now=_off_peak_now(),
        )
        expected = (100_000 * 0.22 + 100_000 * 0.66 + 900_000 * 0.007) / 1e6
        assert acc.totals.cost_usd == round(expected, 6)

    def test_missing_miss_falls_back_to_prompt_minus_hit(self) -> None:
        acc = UsageAccumulator("deepseek-v4-flash")
        acc.record(
            prompt_tokens=10_000,
            completion_tokens=1_000,
            cache_usage=CacheUsage(hit_tokens=8_000, miss_tokens=0),
            now=_peak_now(),
        )
        assert acc.totals.input_tokens == 2_000
        assert acc.totals.cache_read_tokens == 8_000

    def test_unknown_model_has_zero_cost(self) -> None:
        acc = UsageAccumulator("some-unknown-model")
        acc.record(prompt_tokens=1_000, completion_tokens=100)
        assert acc.totals.cost_usd == 0.0
        assert acc.totals.input_tokens == 1_000

    def test_no_usage_yields_none_hit_rate(self) -> None:
        acc = UsageAccumulator("deepseek-v4-flash")
        assert acc.cache_hit_rate is None
        assert acc.totals.requests == 0

    def test_accumulates_across_requests(self) -> None:
        acc = UsageAccumulator("deepseek-v4-flash")
        for _ in range(2):
            acc.record(
                prompt_tokens=100,
                completion_tokens=50,
                now=_peak_now(),
            )
        totals = acc.totals
        assert totals.input_tokens == 200
        assert totals.output_tokens == 100
        assert totals.requests == 2


class TestUsageTotals:
    def test_add_merges(self) -> None:
        a = UsageTotals(
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=30,
            cost_usd=0.5,
            requests=1,
        )
        b = UsageTotals(
            input_tokens=1,
            output_tokens=2,
            cache_read_tokens=3,
            cost_usd=0.25,
            requests=2,
        )
        merged = a.add(b)
        assert merged.input_tokens == 11
        assert merged.output_tokens == 22
        assert merged.cache_read_tokens == 33
        assert merged.cost_usd == 0.75
        assert merged.requests == 3


class TestFormatTokens:
    def test_small(self) -> None:
        assert format_tokens(47) == "47"

    def test_k(self) -> None:
        assert format_tokens(47_000) == "47k"
        assert format_tokens(80_000) == "80k"

    def test_m(self) -> None:
        assert format_tokens(8_400_000) == "8.4M"
        assert format_tokens(1_050_000) == "1.1M"


class TestFormatUsageStats:
    def test_summary_line(self) -> None:
        totals = UsageTotals(
            input_tokens=80_000,
            output_tokens=47_000,
            cache_read_tokens=8_400_000,
            cost_usd=0.333,
            requests=3,
        )
        text = format_usage_stats(totals, cache_hit_rate=0.998)
        assert text == "↑80k ↓47k R8.4M CH99.8% $0.333"

    def test_empty_totals(self) -> None:
        assert format_usage_stats(UsageTotals()) == ""

    def test_no_hit_rate_omits_ch(self) -> None:
        totals = UsageTotals(input_tokens=1_000, output_tokens=500)
        assert format_usage_stats(totals) == "↑1k ↓500"


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0


@dataclass
class _Chunk:
    usage: _Usage | None = None
    choices: list[object] = field(default_factory=list)


class _Completions:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks

    def create(self, **kwargs: object) -> list[object]:
        del kwargs
        return self.chunks


class _Client:
    def __init__(self, chunks: list[object]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(chunks)


async def test_provider_accumulates_usage_from_stream() -> None:
    from xcode.ai.providers.openai import OpenAIChatProvider
    from xcode.ai.types import ProviderConfig

    chunk = _Chunk(
        usage=_Usage(
            prompt_tokens=1_000,
            completion_tokens=100,
            prompt_cache_hit_tokens=900,
            prompt_cache_miss_tokens=100,
        )
    )
    provider = OpenAIChatProvider(
        ProviderConfig(api_key="test", model="deepseek-v4-flash"),
        client=_Client([chunk]),
    )

    events = [
        event
        async for event in provider.stream([{"role": "user", "content": "hi"}], [])
    ]

    assert len(events) == 1
    assert events[0].input_tokens == 1_000  # type: ignore[attr-defined]
    totals = provider.usage_totals
    assert totals.requests == 1
    assert totals.input_tokens == 100
    assert totals.cache_read_tokens == 900
    assert provider.cache_hit_rate == 0.9
