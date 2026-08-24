"""交互式配置注册表单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xcode.cli.config_registry import (
    SETTING_SPECS,
    SettingKind,
    apply_setting,
    commit_setting_value,
    find_setting,
    format_setting,
    load_effective_config,
    matching_settings,
    parse_setting,
    save_setting_text,
    setting_detail,
)
from xcode.harness.config import XcodeRuntimeConfig


def _spec(key: str):
    matches = [item for item in SETTING_SPECS if item.key == key]
    assert len(matches) == 1, f"missing spec: {key}"
    return matches[0]


class TestRegistryIntegrity:
    def test_keys_unique(self) -> None:
        keys = [spec.key for spec in SETTING_SPECS]
        assert len(keys) == len(set(keys))

    def test_enum_specs_declare_choices(self) -> None:
        for spec in SETTING_SPECS:
            if spec.kind is SettingKind.ENUM:
                assert spec.choices, spec.key
            else:
                assert not spec.choices, spec.key

    def test_labels_alignable(self) -> None:
        longest = max(len(spec.label) for spec in SETTING_SPECS)
        assert longest <= 28

    def test_curated_rows_exact(self) -> None:
        keys = {spec.key for spec in SETTING_SPECS}
        assert keys == {
            "execution_modes.default_mode",
            "security.approval_policy",
            "tools.shell",
        }


class TestFormatSetting:
    def test_defaults_on_empty_config(self) -> None:
        config = XcodeRuntimeConfig()
        assert format_setting(_spec("execution_modes.default_mode"), config) == "act"
        assert format_setting(_spec("security.approval_policy"), config) == "on-request"
        assert format_setting(_spec("tools.shell"), config) == "auto"


class TestDescribeChoice:
    def test_known_token_uses_declared_description(self) -> None:
        spec = _spec("execution_modes.default_mode")
        text = spec.describe_choice("plan")
        assert "Read-only" in text or "read-only" in text

    def test_unknown_token_falls_back_to_description(self) -> None:
        spec = _spec("tools.shell")
        assert spec.describe_choice("bash") == spec.description


class TestParseSetting:
    def test_enum_rejects_unknown_choice(self) -> None:
        spec = _spec("execution_modes.default_mode")
        assert parse_setting(spec, "PLAN") == "plan"
        with pytest.raises(ValueError):
            parse_setting(spec, "yolo")

    def test_approval_policy_enum(self) -> None:
        spec = _spec("security.approval_policy")
        assert parse_setting(spec, "never") == "never"
        with pytest.raises(ValueError):
            parse_setting(spec, "sometimes")

    def test_shell_enum_accepts_all_choices(self) -> None:
        spec = _spec("tools.shell")
        assert parse_setting(spec, "ZSH") == "zsh"


class TestApplySetting:
    def test_creates_nested_dicts(self) -> None:
        raw: dict = {}
        apply_setting(raw, _spec("tools.shell"), "zsh")
        assert raw == {"tools": {"shell": "zsh"}}

    def test_none_pops_leaf_key(self) -> None:
        raw: dict = {
            "execution_modes": {
                "default_mode": "build",
                "other_key": 1,
            }
        }
        apply_setting(raw, _spec("execution_modes.default_mode"), None)
        assert raw["execution_modes"] == {"other_key": 1}


class TestCommitSettingValue:
    def _write(self, path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_valid_value_saved(self, tmp_path: Path) -> None:
        config_path = self._write(tmp_path / "xcode.config.json", {})
        ok, message = commit_setting_value(
            config_path, _spec("execution_modes.default_mode"), "build"
        )
        assert ok
        assert "build" in message
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["execution_modes"]["default_mode"] == "build"

    def test_type_mismatch_not_saved(self, tmp_path: Path) -> None:
        config_path = self._write(
            tmp_path / "xcode.config.json", {"tools": {"shell": "bash"}}
        )
        ok, _ = commit_setting_value(
            config_path, _spec("execution_modes.default_mode"), 123
        )
        assert not ok
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved == {"tools": {"shell": "bash"}}

    def test_existing_fields_preserved(self, tmp_path: Path) -> None:
        config_path = self._write(
            tmp_path / "xcode.config.json",
            {
                "provider": {
                    "model_profiles": {
                        "main": {"transport": "deepseek_chat", "api_key": "sk-x"}
                    }
                },
                "agent": {"max_steps": 10},
            },
        )
        ok, _ = save_setting_text(config_path, _spec("tools.shell"), "bash")
        assert ok
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["provider"]["model_profiles"]["main"]["transport"] == (
            "deepseek_chat"
        )
        assert saved["agent"] == {"max_steps": 10}
        assert saved["tools"]["shell"] == "bash"

    def test_save_text_reports_parse_errors(self, tmp_path: Path) -> None:
        config_path = tmp_path / "xcode.config.json"
        ok, message = save_setting_text(
            config_path, _spec("security.approval_policy"), "sometimes"
        )
        assert not ok
        assert "Expected" in message
        assert not config_path.exists()


class TestLookupAndDetails:
    def test_find_by_label_and_key(self) -> None:
        assert find_setting("default mode") is not None
        assert find_setting("tools.shell") is not None
        assert find_setting("nonexistent-keyword-xyz") is None

    def test_matching_returns_all_hits(self) -> None:
        hits = matching_settings("shell")
        assert [spec.label for spec in hits] == ["Shell"]

    def test_detail_fallback_renders_value(self) -> None:
        config = load_effective_config(Path("/nonexistent/xcode.config.json"))
        lines = setting_detail(_spec("tools.shell"), config)
        assert lines == ["  auto"]
