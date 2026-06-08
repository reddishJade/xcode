from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

from xcode.ai.model_modes import parse_model_mode
from xcode.cli.repl_settings import handle_model_command


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


if __name__ == "__main__":
    unittest.main()
