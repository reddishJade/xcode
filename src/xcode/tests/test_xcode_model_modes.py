from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import cast
import unittest

from xcode.ai.model_modes import parse_model_mode
from xcode.cli.repl_settings import handle_effort_command, handle_model_command
from xcode.harness.app import XcodeApp
from xcode.harness.agent_runtime import StructuredAgent


class XcodeModelModeTests(unittest.TestCase):
    def test_parse_model_mode_plain_model(self) -> None:
        parsed = parse_model_mode("deepseek-v4")

        self.assertEqual(parsed.model, "deepseek-v4")
        self.assertIsNone(parsed.provider)
        self.assertIsNone(parsed.thinking_level)

    def test_parse_model_mode_provider_and_thinking(self) -> None:
        parsed = parse_model_mode("judge/gpt-5:xhigh")

        self.assertEqual(parsed.provider, "judge")
        self.assertEqual(parsed.model, "gpt-5")
        self.assertEqual(parsed.thinking_level, "xhigh")

    def test_parse_model_mode_rejects_unknown_thinking_level(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_mode("gpt-5:turbo")

    def test_model_command_applies_profile_and_thinking_level(self) -> None:
        app = _ModelApp()

        with redirect_stdout(StringIO()):
            handle_model_command("/model judge/gpt-5:off", app)

        self.assertEqual(app.calls[0]["profile"], "judge")
        self.assertEqual(app.calls[0]["model"], "gpt-5")
        self.assertFalse(app.calls[0]["thinking"])
        self.assertIsNone(app.calls[0]["reasoning_effort"])

    def test_effort_command_applies_supported_levels(self) -> None:
        app = _EffortApp(transport="openai_chat")

        with redirect_stdout(StringIO()):
            handle_effort_command("/effort xhigh", app)

        self.assertEqual(app.calls[0]["reasoning_effort"], "xhigh")

    def test_effort_command_rejects_unsupported_transports(self) -> None:
        app = _EffortApp(transport="chatglm_chat")
        output = StringIO()

        with redirect_stdout(output):
            handle_effort_command("/effort high", app)

        self.assertEqual(app.calls, [])
        self.assertIn("does not support reasoning effort", output.getvalue())

    def test_get_model_info_uses_active_provider_transport(self) -> None:
        agent = cast(
            StructuredAgent,
            _Agent(_ProviderWrapper(_Provider("deepseek_chat"))),
        )
        app = XcodeApp(agent=agent)

        info = app.get_model_info()

        self.assertEqual(info["transport"], "deepseek_chat")
        self.assertEqual(info["reasoning_effort"], "high")


class _ModelApp:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_model_info(self) -> dict[str, str]:
        return {"model": "old"}

    def set_model(
        self,
        *,
        model: str,
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "profile": profile,
                "base_url": base_url,
                "api_key": api_key,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
            }
        )
        return model


class _EffortApp:
    def __init__(self, transport: str) -> None:
        self.transport = transport
        self.calls: list[dict[str, object]] = []

    def get_model_info(self) -> dict[str, str]:
        return {
            "model": "current",
            "transport": self.transport,
            "reasoning_effort": "high",
        }

    def set_model(
        self,
        *,
        model: str,
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "profile": profile,
                "base_url": base_url,
                "api_key": api_key,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
            }
        )
        return model


class _Provider:
    def __init__(self, transport: str) -> None:
        self.transport = transport
        self.model = "current"
        self.base_url = "https://example.test"
        self.thinking = True
        self.reasoning_effort = "high"


class _ProviderWrapper:
    def __init__(self, active_provider: _Provider) -> None:
        self.active_provider = active_provider


class _Agent:
    def __init__(self, provider: _ProviderWrapper) -> None:
        self.provider = provider


if __name__ == "__main__":
    unittest.main()
