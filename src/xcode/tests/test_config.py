"""运行时配置 schema 与合并逻辑的单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xcode.harness.config import XcodeRuntimeConfig, _config_from_dict


class TestExecutionModesDefaultMode:
    def test_defaults_to_act(self) -> None:
        cfg = XcodeRuntimeConfig()
        assert cfg.execution_modes.default_mode == "act"

    def test_parse_build(self) -> None:
        cfg = XcodeRuntimeConfig.model_validate(
            {"execution_modes": {"default_mode": "build"}}
        )
        assert cfg.execution_modes.default_mode == "build"

    def test_parse_plan(self) -> None:
        cfg = XcodeRuntimeConfig.model_validate(
            {"execution_modes": {"default_mode": "plan"}}
        )
        assert cfg.execution_modes.default_mode == "plan"

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            XcodeRuntimeConfig.model_validate(
                {"execution_modes": {"default_mode": "auto"}}
            )

    def test_invalid_value_error_names_field(self) -> None:
        with pytest.raises(ValueError, match="execution_modes.default_mode"):
            _config_from_dict({"execution_modes": {"default_mode": "auto"}})

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            XcodeRuntimeConfig.model_validate(
                {"execution_modes": {"initial_mode": "build"}}
            )
