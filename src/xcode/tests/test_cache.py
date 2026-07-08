"""缓存统计单元测试。"""

from __future__ import annotations

from xcode.ai.cache import CacheUsage, extract_cache_usage


class TestCacheUsage:
    def test_hit_rate_zero_total(self) -> None:
        assert CacheUsage().hit_rate == 0.0

    def test_hit_rate_full_hit(self) -> None:
        assert CacheUsage(hit_tokens=100, miss_tokens=0).hit_rate == 1.0

    def test_hit_rate_mixed(self) -> None:
        assert CacheUsage(hit_tokens=75, miss_tokens=25).hit_rate == 0.75

    def test_hit_rate_partial(self) -> None:
        assert CacheUsage(hit_tokens=1, miss_tokens=9).hit_rate == 0.1


def _fake_response(usage: object = None) -> object:
    class FakeChunk:
        pass

    chunk = FakeChunk()
    chunk.usage = usage
    return chunk


def test_extract_cache_usage_no_usage() -> None:
    result = extract_cache_usage(_fake_response(None))
    assert result == CacheUsage()


def test_extract_cache_usage_deepseek_native() -> None:
    class Usage:
        prompt_cache_hit_tokens = 50
        prompt_cache_miss_tokens = 10
        prompt_tokens = 60

    result = extract_cache_usage(_fake_response(Usage()))
    assert result == CacheUsage(hit_tokens=50, miss_tokens=10)


def test_extract_cache_usage_deepseek_miss_from_prompt() -> None:
    class Usage:
        prompt_cache_hit_tokens = 50
        prompt_cache_miss_tokens = 0
        prompt_tokens = 100

    result = extract_cache_usage(_fake_response(Usage()))
    # miss should be derived: 100 - 50 = 50
    assert result == CacheUsage(hit_tokens=50, miss_tokens=50)


def test_extract_cache_usage_compat_cached_tokens() -> None:
    class Details:
        cached_tokens = 30

    class Usage:
        prompt_tokens = 100
        prompt_cache_hit_tokens = 0
        prompt_cache_miss_tokens = 0
        prompt_tokens_details = Details()

    result = extract_cache_usage(_fake_response(Usage()))
    assert result == CacheUsage(hit_tokens=30, miss_tokens=70)
