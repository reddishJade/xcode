from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xcode.cli.setup_wizard import run_setup_wizard


class XcodeSetupWizardTests(unittest.TestCase):
    def test_provider_choices_write_canonical_transport_names(self) -> None:
        transport_map = {
            "OpenAI": "openai_chat",
            "Anthropic": "anthropic_messages",
            "DeepSeek": "deepseek_chat",
            "Xiaomi MiMo": "mimo_chat",
            "ChatGLM": "chatglm_chat",
        }
        for provider_label, expected_transport in transport_map.items():
            with self.subTest(provider_label=provider_label):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "xcode.config.json"

                    with (
                        patch("questionary.select") as mock_select,
                        patch("questionary.text") as mock_text,
                        patch("questionary.confirm") as mock_confirm,
                        patch("builtins.print"),
                    ):
                        select_responses = iter([provider_label, "gpt-4o"])
                        mock_select.side_effect = lambda *a, **kw: type(
                            "Q", (), {"ask": lambda _self=None: next(select_responses)}
                        )()
                        text_responses = iter(["test-key", ""])
                        mock_text.side_effect = lambda *a, **kw: type(
                            "Q", (), {"ask": lambda _self=None: next(text_responses)}
                        )()
                        mock_confirm.return_value.ask.return_value = True

                        run_setup_wizard(Path(temp_dir))

                    data = json.loads(path.read_text(encoding="utf-8"))
                    transport = data["provider"]["model_profiles"]["main"]["transport"]
                    self.assertEqual(transport, expected_transport)


if __name__ == "__main__":
    unittest.main()
