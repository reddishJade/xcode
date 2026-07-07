"""回归测试：/thinking 只翻转 thinking，不改动 reasoning_effort。

/thinking off→on 须保持 reasoning_effort 沿用 profile 现值，
不得在 effort 为默认 None 时凭空注入非默认值。
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest

from xcode.ai.providers import ProviderSettings
from xcode.ai.providers.factory import ModelProfileConfig
from xcode.cli.repl_settings import handle_thinking_command
from xcode.harness.app import XcodeApp


class _FakeProvider:
    def __init__(
        self, model: str, thinking: bool, reasoning_effort: str | None
    ) -> None:
        self.model = model
        self.base_url = "http://example.test"
        self.transport = "fake"
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort


class _FakeAgent:
    def __init__(self, provider: object) -> None:
        self.provider = provider


def _build_app(*, thinking: bool, reasoning_effort: str | None) -> XcodeApp:
    provider = _FakeProvider("m", thinking, reasoning_effort)
    return XcodeApp(
        agent=_FakeAgent(provider),  # type: ignore[arg-type]
        _model_profiles={
            "main": ModelProfileConfig(
                thinking=thinking, reasoning_effort=reasoning_effort
            )
        },
        _env_files=(),
    )


def _patch_bundle() -> object:
    """按 set_model 实际传入的 ProviderSettings 构建 provider。

    忠实反映 new_cfg.reasoning_effort 的流转：handle_thinking_command 若
    显式传 "high" 到 set_model，build_provider_bundle 收到的 profile config
    就带 reasoning_effort="high"，产出的 provider 也就带 "high"。这样才能
    捕捉旧实现把默认 None 注入为 "high" 的回归。
    """

    def _fake(settings: ProviderSettings) -> object:
        from types import SimpleNamespace

        cfg = settings.model_profiles["main"]
        provider = _FakeProvider(
            cfg.chat_model or "m",
            cfg.thinking,
            cfg.reasoning_effort,
        )
        return SimpleNamespace(llm=provider, llms={"main": provider})

    return patch("xcode.ai.providers.build_provider_bundle", side_effect=_fake)


def _run_off_on(app: XcodeApp) -> None:
    with redirect_stdout(StringIO()):
        handle_thinking_command("/thinking off", app)
        handle_thinking_command("/thinking on", app)


def test_thinking_off_on_preserves_none_effort() -> None:
    """默认 effort=None 时 off→on 不得注入 "high"。"""
    app = _build_app(thinking=True, reasoning_effort=None)

    with _patch_bundle():
        _run_off_on(app)

    final = app.agent.provider
    assert isinstance(final, _FakeProvider)
    assert final.thinking is True
    assert final.reasoning_effort is None, "off→on 把默认 None effort 注入为非默认值"


def test_thinking_off_on_preserves_custom_effort() -> None:
    """自定义 effort（如 medium）在 off→on 后保持不变。"""
    app = _build_app(thinking=True, reasoning_effort="medium")

    with _patch_bundle():
        _run_off_on(app)

    final = app.agent.provider
    assert isinstance(final, _FakeProvider)
    assert final.thinking is True
    assert final.reasoning_effort == "medium"


def test_thinking_off_only_flips_thinking() -> None:
    """off 只翻转 thinking，effort 不被清空。"""
    app = _build_app(thinking=True, reasoning_effort="medium")

    with _patch_bundle():
        with redirect_stdout(StringIO()):
            handle_thinking_command("/thinking off", app)

    final = app.agent.provider
    assert isinstance(final, _FakeProvider)
    assert final.thinking is False
    assert final.reasoning_effort == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
