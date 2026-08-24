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

    def test_curated_core_rows(self) -> None:
        keys = {spec.key for spec in SETTING_SPECS}
        assert "execution_modes.default_mode" in keys
        assert "security.approval_policy" in keys
        assert "tools.shell" in keys
        # 调参类字段不进入运行时浏览器，只保留在 JSON 配置中。
        assert not any(key.startswith("agent.") for key in keys)
        assert not any(key.startswith("request_hygiene.") for key in keys)


class TestFormatSetting:
    def test_defaults_on_empty_config(self) -> None:
        config = XcodeRuntimeConfig()
        assert format_setting(_spec("execution_modes.default_mode"), config) == "act"
        assert format_setting(_spec("skills.trust_project_skills"), config) == "off"
        assert (
            format_setting(_spec("security.approval_policy"), config)
            == "on-request"
        )
        assert (
            format_setting(_spec("security.restricted_dirs"), config) == "(none)"
        )
        assert (
            format_setting(_spec("security.external_directories"), config)
            == "0 allowed"
        )

    def test_seconds_and_bool(self) -> None:
        config = XcodeRuntimeConfig()
        assert (
            format_setting(_spec("security.auto_review_timeout_seconds"), config)
            == "90s"
        )
        assert format_setting(_spec("skills.trust_project_skills"), config) == "off"

    def test_global_default_unset_renders_default(self) -> None:
        config = XcodeRuntimeConfig()
        assert (
            format_setting(_spec("security.global_default"), config) == "default"
        )


class TestDescribeChoice:
    def test_known_token_uses_declared_description(self) -> None:
        spec = _spec("execution_modes.default_mode")
        text = spec.describe_choice("plan")
        assert "Read-only" in text or "read-only" in text

    def test_unknown_token_falls_back_to_description(self) -> None:
        spec = _spec("tools.shell")
        assert spec.describe_choice("bash") == spec.description


class TestParseSetting:
    def test_bool_tokens(self) -> None:
        spec = _spec("skills.trust_project_skills")
        assert parse_setting(spec, "on") is True
        assert parse_setting(spec, "FALSE") is False
        with pytest.raises(ValueError):
            parse_setting(spec, "maybe")

    def test_float(self) -> None:
        spec = _spec("security.auto_review_timeout_seconds")
        assert parse_setting(spec, "45.5") == 45.5
        with pytest.raises(ValueError):
            parse_setting(spec, "abc")

    def test_enum_rejects_unknown_choice(self) -> None:
        spec = _spec("execution_modes.default_mode")
        assert parse_setting(spec, "PLAN") == "plan"
        with pytest.raises(ValueError):
            parse_setting(spec, "yolo")

    def test_global_default_clears_via_none_choice(self) -> None:
        spec = _spec("security.global_default")
        assert parse_setting(spec, "deny") == "deny"
        assert parse_setting(spec, "default") is None

    def test_str_list_comma_separated(self) -> None:
        spec = _spec("tools.subagent_extra_tools")
        assert parse_setting(spec, "todowrite, webfetch") == ("todowrite", "webfetch")
        assert parse_setting(spec, "none") is None


class TestApplySetting:
    def test_creates_nested_dicts(self) -> None:
        raw: dict = {}
        apply_setting(raw, _spec("tools.shell"), "zsh")
        assert raw == {"tools": {"shell": "zsh"}}

    def test_none_pops_leaf_key(self) -> None:
        raw: dict = {
            "security": {"global_default": "ask", "approval_policy": "never"}
        }
        apply_setting(raw, _spec("security.global_default"), None)
        assert raw["security"] == {"approval_policy": "never"}

    def test_empty_str_list_restores_default(self) -> None:
        raw: dict = {"tools": {"subagent_extra_tools": ["todowrite"]}}
        apply_setting(raw, _spec("tools.subagent_extra_tools"), ())
        assert raw["tools"] == {}


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

    def test_out_of_range_value_not_saved(self, tmp_path: Path) -> None:
        config_path = self._write(
            tmp_path / "xcode.config.json", {"tools": {"shell": "bash"}}
        )
        ok, message = save_setting_text(
            config_path, _spec("security.auto_review_timeout_seconds"), "301"
        )
        assert not ok
        assert "less than or equal to 300" in message
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
                }
            },
        )
        ok, _ = save_setting_text(config_path, _spec("tools.shell"), "bash")
        assert ok
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["provider"]["model_profiles"]["main"]["transport"] == (
            "deepseek_chat"
        )
        assert saved["tools"]["shell"] == "bash"

    def test_save_text_reports_parse_errors(self, tmp_path: Path) -> None:
        config_path = tmp_path / "xcode.config.json"
        ok, message = save_setting_text(
            config_path, _spec("security.approval_policy"), "sometimes"
        )
        assert not ok
        assert "Expected" in message
        assert not config_path.exists()

    def test_info_rows_reject_saving(self, tmp_path: Path) -> None:
        config_path = tmp_path / "xcode.config.json"
        ok, message = save_setting_text(
            config_path, _spec("security.external_directories"), "1"
        )
        assert not ok
        assert "read-only" in message


class TestLookupAndDetails:
    def test_find_by_label_and_key(self) -> None:
        assert find_setting("default mode") is not None
        assert find_setting("tools.shell") is not None
        assert find_setting("nonexistent-keyword-xyz") is None

    def test_matching_returns_all_hits(self) -> None:
        hits = matching_settings("dirs")
        labels = {spec.label for spec in hits}
        assert labels == {"Restricted Dirs", "External Dirs"}

    def test_external_dirs_detail_lists_entries(self) -> None:
        config = load_effective_config(Path("/nonexistent/xcode.config.json"))
        lines = setting_detail(_spec("security.external_directories"), config)
        assert lines == ["  (none)"]
