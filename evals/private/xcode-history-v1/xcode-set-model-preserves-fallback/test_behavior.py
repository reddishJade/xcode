"""Agent 不可见的 fallback 热切换行为 oracle。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from xcode.ai.providers.factory import ModelProfileConfig
from xcode.harness.agent_runtime.fallback import _FallbackWithRetryPrimary
from xcode.harness.app import XcodeApp


class _Provider:
    def __init__(self, model: str) -> None:
        self.model = model
        self.base_url = "http://provider.invalid"
        self.transport = "test"
        self.thinking = True
        self.reasoning_effort: str | None = None


def _app() -> XcodeApp:
    wrapper = _FallbackWithRetryPrimary(_Provider("old-main"), _Provider("fallback"))
    agent = SimpleNamespace(provider=wrapper)
    return XcodeApp(
        agent=agent,
        _model_profiles={"main": ModelProfileConfig()},
        _env_files=(),
    )


def test_main_replacement_preserves_fallback_and_resets_state() -> None:
    app = _app()
    wrapper = app.agent.provider
    wrapper._consecutive_errors = 2
    wrapper._fallback_successes = 1
    wrapper._using_fallback = True
    replacement = _Provider("new-main")
    bundle = SimpleNamespace(llm=replacement, llms={"main": replacement})

    with patch("xcode.ai.providers.build_provider_bundle", return_value=bundle):
        app.set_model(model="new-main", profile="main")

    assert app.agent.provider is wrapper
    assert wrapper.active_provider is replacement
    assert wrapper._fallback.model == "fallback"
    assert wrapper._consecutive_errors == 0
    assert wrapper._fallback_successes == 0
    assert wrapper._using_fallback is False
