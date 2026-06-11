from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xcode.cli.setup_wizard import PROVIDER_PRESETS, run_setup_wizard


class XcodeSetupWizardTests(unittest.TestCase):
    def test_provider_choices_write_canonical_transport_names(self) -> None:
        provider_cases = {
            "openai": ("openai_chat", "high"),
            "deepseek": ("deepseek_chat", "high"),
            "mimo": ("mimo_chat", None),
            "chatglm": ("chatglm_chat", None),
        }
        for provider_key, (
            expected_transport,
            expected_effort,
        ) in provider_cases.items():
            with self.subTest(provider_label=provider_key):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "xcode.config.json"
                    preset = PROVIDER_PRESETS[provider_key]
                    select_responses = [preset["label"], preset["default_model"]]
                    if expected_effort is not None:
                        select_responses.append(expected_effort)

                    with (
                        patch("questionary.select") as mock_select,
                        patch("questionary.text") as mock_text,
                        patch("questionary.confirm") as mock_confirm,
                        patch("builtins.print"),
                    ):
                        responses = iter(select_responses)
                        mock_select.side_effect = lambda *a, **kw: type(
                            "Q", (), {"ask": lambda _self=None: next(responses)}
                        )()
                        text_responses = iter(["test-key", ""])
                        mock_text.side_effect = lambda *a, **kw: type(
                            "Q", (), {"ask": lambda _self=None: next(text_responses)}
                        )()
                        mock_confirm.return_value.ask.return_value = True

                        run_setup_wizard(Path(temp_dir))

                    data = json.loads(path.read_text(encoding="utf-8"))
                    profile = data["provider"]["model_profiles"]["main"]
                    transport = profile["transport"]
                    self.assertEqual(transport, expected_transport)
                    if expected_effort is None:
                        self.assertNotIn("reasoning_effort", profile)
                    else:
                        self.assertEqual(profile["reasoning_effort"], expected_effort)


if __name__ == "__main__":
    unittest.main()
