from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from xcode.cli.setup_wizard import PROVIDER_PRESETS, run_setup_wizard
import pytest

PROVIDER_CASES: list[tuple[str, str, str | None]] = [
    ("openai", "openai_chat", "high"),
    ("deepseek", "deepseek_chat", "high"),
    ("mimo", "mimo_chat", None),
    ("chatglm", "chatglm_chat", None),
]


class XcodeSetupWizardTests:
    @pytest.mark.parametrize(
        "provider_key,expected_transport,expected_effort", PROVIDER_CASES
    )
    def test_provider_choices_write_canonical_transport_names(
        self,
        provider_key: str,
        expected_transport: str,
        expected_effort: str | None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "xcode.config.json"
            preset = PROVIDER_PRESETS[provider_key]
            select_responses = [
                preset["label"],
                preset["default_model"],
                "enabled",
            ]
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

            model_select_kwargs = mock_select.call_args_list[1].kwargs
            assert "default" not in model_select_kwargs
            assert model_select_kwargs["choices"][0] == preset["default_model"]

            data = json.loads(path.read_text(encoding="utf-8"))
            profile = data["provider"]["model_profiles"]["main"]
            transport = profile["transport"]
            assert transport == expected_transport
            if expected_effort is None:
                assert "reasoning_effort" not in profile
            else:
                assert profile["reasoning_effort"] == expected_effort


if __name__ == "__main__":
    pytest.main()
