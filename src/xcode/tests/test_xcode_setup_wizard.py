from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xcode.cli.setup_wizard import run_setup_wizard


class XcodeSetupWizardTests(unittest.TestCase):
    def test_provider_choices_write_canonical_transport_names(self) -> None:
        cases = [
            ("1", "openai_chat"),
            ("2", "anthropic_messages"),
            ("3", "deepseek_chat"),
            ("4", "mimo_chat"),
            ("5", "chatglm_chat"),
        ]
        for provider_choice, expected_transport in cases:
            with self.subTest(provider_choice=provider_choice):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "xcode.config.json"
                    answers = iter([provider_choice, "test-key", "", "", "y"])

                    with patch("builtins.input", lambda _prompt: next(answers)):
                        with patch("builtins.print"):
                            run_setup_wizard(Path(temp_dir))

                    data = json.loads(path.read_text(encoding="utf-8"))
                    transport = data["provider"]["model_profiles"]["main"]["transport"]
                    self.assertEqual(transport, expected_transport)


if __name__ == "__main__":
    unittest.main()
