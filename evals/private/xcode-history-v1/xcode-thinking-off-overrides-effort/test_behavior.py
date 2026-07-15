"""Agent 不可见的 thinking request 参数行为 oracle。"""

from types import SimpleNamespace

from xcode.ai.providers.openai import OpenAIChatProvider
from xcode.harness.app import XcodeApp


def _provider(*, thinking: bool, effort: str | None) -> OpenAIChatProvider:
    return OpenAIChatProvider(
        api_key="hidden-test-key",
        base_url="https://provider.invalid",
        model="model",
        thinking=thinking,
        reasoning_effort=effort,
        client=SimpleNamespace(),
    )


def test_explicit_off_wins_over_configured_effort() -> None:
    provider = _provider(thinking=False, effort="high")
    params: dict[str, object] = {}

    provider._build_thinking_params(params)

    assert params["reasoning_effort"] == "none"


def test_enabled_thinking_keeps_configured_effort() -> None:
    provider = _provider(thinking=True, effort="high")
    params: dict[str, object] = {}

    provider._build_thinking_params(params)

    assert params["reasoning_effort"] == "high"


def test_app_status_reports_disabled_thinking() -> None:
    provider = _provider(thinking=False, effort="high")
    agent = SimpleNamespace(provider=provider)
    app = XcodeApp(agent=agent)

    assert app.get_model_info()["thinking"] == "off"
