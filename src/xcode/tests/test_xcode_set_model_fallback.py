"""回归测试：set_model 在 profile=="main" 时保留 _FallbackWithRetryPrimary 容灾包装层。

/model、/thinking、/effort 命令均走 App.set_model，热替换主 provider 时
须原地 replace_primary 而非裸替换，保留 fallback 容灾。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from xcode.ai.providers.factory import ModelProfileConfig
from xcode.harness.agent_runtime.fallback import _FallbackWithRetryPrimary
from xcode.harness.app import XcodeApp


class _FakeProvider:
    """最小 provider 桩，满足 _FallbackWithRetryPrimary 读取的属性。"""

    def __init__(self, model: str) -> None:
        self.model = model
        self.base_url = "http://example.test"
        self.transport = "fake"
        self.thinking = True
        self.reasoning_effort: str | None = None


class _FakeAgent:
    def __init__(self, provider: object) -> None:
        self.provider = provider


def _build_app_with_fallback() -> XcodeApp:
    """构造一个已激活 fallback 容灾包装的 XcodeApp。"""
    primary = _FakeProvider("main-initial")
    fallback = _FakeProvider("fallback-initial")
    wrapped = _FallbackWithRetryPrimary(primary, fallback)
    agent = _FakeAgent(wrapped)
    return XcodeApp(
        agent=agent,  # type: ignore[arg-type]
        _model_profiles={"main": ModelProfileConfig()},
        _env_files=(),
    )


def test_set_model_preserves_fallback_wrapper() -> None:
    """profile=="main" 时 set_model 必须保留 _FallbackWithRetryPrimary 包装。"""
    app = _build_app_with_fallback()
    before = app.agent.provider
    assert isinstance(before, _FallbackWithRetryPrimary)

    rebuilt_primary = _FakeProvider("main-rebuilt")
    new_bundle = SimpleNamespace(
        llm=rebuilt_primary,
        llms={"main": rebuilt_primary},
    )
    with patch("xcode.ai.providers.build_provider_bundle", return_value=new_bundle):
        app.set_model(model="main-rebuilt", profile="main")

    after = app.agent.provider
    assert isinstance(after, _FallbackWithRetryPrimary), (
        "set_model 裸替换丢失了 fallback 容灾包装层"
    )
    assert after is before, "应原地 replace_primary 而非新建包装实例"
    assert after.active_provider.model == "main-rebuilt"
    assert after._fallback.model == "fallback-initial", "fallback provider 不应被替换"


def test_set_model_resets_fallback_counters() -> None:
    """replace_primary 应清空旧主错误历史，下一轮优先尝试新主。"""
    app = _build_app_with_fallback()
    wrapper: _FallbackWithRetryPrimary = app.agent.provider  # type: ignore[assignment]
    wrapper._consecutive_errors = 2
    wrapper._using_fallback = True
    wrapper._fallback_successes = 1

    rebuilt_primary = _FakeProvider("main-rebuilt")
    new_bundle = SimpleNamespace(
        llm=rebuilt_primary,
        llms={"main": rebuilt_primary},
    )
    with patch("xcode.ai.providers.build_provider_bundle", return_value=new_bundle):
        app.set_model(model="main-rebuilt", profile="main")

    assert wrapper._consecutive_errors == 0
    assert wrapper._using_fallback is False
    assert wrapper._fallback_successes == 0


def test_set_model_non_main_profile_keeps_wrapper_untouched() -> None:
    """非 main profile（如 subagent）不应触碰已包装的主 provider。"""
    app = _build_app_with_fallback()
    wrapper_before = app.agent.provider

    subagent_provider = _FakeProvider("subagent-new")
    new_bundle = SimpleNamespace(
        llm=subagent_provider,
        llms={"subagent": subagent_provider},
    )
    with patch("xcode.ai.providers.build_provider_bundle", return_value=new_bundle):
        app.set_model(model="subagent-new", profile="subagent")

    assert app.agent.provider is wrapper_before, (
        "非 main profile 的 set_model 不应替换主 provider"
    )


def test_set_model_without_fallback_wrapper_assigns_directly() -> None:
    """未激活 fallback 包装时，set_model 仍走直接赋值路径。"""
    primary = _FakeProvider("main-initial")
    agent = _FakeAgent(primary)
    app = XcodeApp(
        agent=agent,  # type: ignore[arg-type]
        _model_profiles={"main": ModelProfileConfig()},
        _env_files=(),
    )

    rebuilt = _FakeProvider("main-rebuilt")
    new_bundle = SimpleNamespace(llm=rebuilt, llms={"main": rebuilt})
    with patch("xcode.ai.providers.build_provider_bundle", return_value=new_bundle):
        app.set_model(model="main-rebuilt", profile="main")

    assert app.agent.provider is rebuilt


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
