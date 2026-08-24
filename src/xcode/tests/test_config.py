"""运行时配置 schema 与合并逻辑的单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xcode.harness.config import XcodeRuntimeConfig, _config_from_dict
from xcode.harness.execution_env import NetworkAccess, SandboxMode


class TestProviderContextWindow:
    def test_defaults_to_none(self) -> None:
        cfg = XcodeRuntimeConfig()
        assert cfg.provider.model_profiles["main"].context_window is None

    def test_parse_override(self) -> None:
        cfg = XcodeRuntimeConfig.model_validate(
            {"provider": {"model_profiles": {"main": {"context_window": 262_144}}}}
        )
        assert cfg.provider.model_profiles["main"].context_window == 262_144

    def test_null_means_registry_default(self) -> None:
        cfg = XcodeRuntimeConfig.model_validate(
            {"provider": {"model_profiles": {"main": {"context_window": None}}}}
        )
        assert cfg.provider.model_profiles["main"].context_window is None

    def test_non_positive_rejected(self) -> None:
        with pytest.raises(ValidationError):
            XcodeRuntimeConfig.model_validate(
                {"provider": {"model_profiles": {"main": {"context_window": 0}}}}
            )

    def test_non_int_rejected(self) -> None:
        with pytest.raises(ValidationError):
            XcodeRuntimeConfig.model_validate(
                {"provider": {"model_profiles": {"main": {"context_window": "262144"}}}}
            )


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


class TestApprovalConfiguration:
    def test_defaults_to_on_request(self) -> None:
        cfg = XcodeRuntimeConfig()

        assert cfg.security.approval_policy == "on-request"

    def test_removed_always_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            XcodeRuntimeConfig.model_validate(
                {"security": {"approval_policy": "always"}}
            )

    def test_global_reviewer_switch_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            XcodeRuntimeConfig.model_validate(
                {"security": {"approvals_reviewer": "user"}}
            )


class TestSandboxConfiguration:
    def test_defaults_to_workspace_write_without_network(self) -> None:
        cfg = XcodeRuntimeConfig()

        assert cfg.security.sandbox.mode is SandboxMode.WORKSPACE_WRITE
        assert cfg.security.sandbox.network_access is NetworkAccess.DENY

    def test_parses_explicit_full_access(self) -> None:
        cfg = XcodeRuntimeConfig.model_validate(
            {
                "security": {
                    "sandbox": {
                        "mode": "danger-full-access",
                        "network_access": "allow",
                    }
                }
            }
        )

        assert cfg.security.sandbox.mode is SandboxMode.DANGER_FULL_ACCESS
        assert cfg.security.sandbox.network_access is NetworkAccess.ALLOW

    def test_rejects_unknown_sandbox_mode(self) -> None:
        with pytest.raises(ValidationError):
            XcodeRuntimeConfig.model_validate(
                {"security": {"sandbox": {"mode": "directory-check"}}}
            )
