from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xcode.harness.observability import (
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    StaticPermission,
)
import pytest

type ToolInput = dict[str, object]
type ConstraintSummary = tuple[str, str]
type VerdictSummary = tuple[str, str]


@dataclass(frozen=True)
class StructuredPermissionParityCase:
    """结构化文件工具的 legacy 与 shadow parity 案例。"""

    name: str
    tool_name: str
    tool_input: ToolInput
    expected_legacy_decision: str
    expected_constraints: tuple[ConstraintSummary, ...]
    expected_verdict: VerdictSummary
    expected_shadow_diff: str | None = None


CASES: tuple[StructuredPermissionParityCase, ...] = (
    StructuredPermissionParityCase(
        name="read_file static allow",
        tool_name="read_file",
        tool_input={"path": "docs/permission-model.md"},
        expected_legacy_decision="allow",
        expected_constraints=(("rule", "allow"), ("boundary", "allow")),
        expected_verdict=("allow", "rule"),
    ),
    StructuredPermissionParityCase(
        name="write_file static ask",
        tool_name="write_file",
        tool_input={
            "path": "docs/static-policy-probe.md",
            "content": "probe",
        },
        expected_legacy_decision="ask",
        expected_constraints=(("rule", "ask"), ("boundary", "allow")),
        expected_verdict=("ask", "rule"),
    ),
    StructuredPermissionParityCase(
        name="edit_file static deny",
        tool_name="edit_file",
        tool_input={
            "path": "src/xcode/static-policy-blocked.py",
            "old": "A",
            "new": "B",
        },
        expected_legacy_decision="deny",
        expected_constraints=(("rule", "deny"), ("boundary", "allow")),
        expected_verdict=("deny", "rule"),
    ),
    StructuredPermissionParityCase(
        name="edit_file boundary deny on .env",
        tool_name="edit_file",
        tool_input={"path": ".env", "old": "A", "new": "B"},
        expected_legacy_decision="deny",
        expected_constraints=(("rule", "ask"), ("boundary", "deny")),
        expected_verdict=("deny", "boundary"),
    ),
    StructuredPermissionParityCase(
        name="apply_patch allowlist fallback ask",
        tool_name="apply_patch",
        tool_input={"paths": ["src/xcode/static_policy_probe.py"]},
        expected_legacy_decision="ask",
        expected_constraints=(("rule", "ask"), ("boundary", "allow")),
        expected_verdict=("ask", "rule"),
    ),
)


class StructuredPermissionParityTests:
    """固定 STEP A shadow parity 结果，避免后续 cutover 回归。"""

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
    def test_structured_file_tools_static_policy_shadow_parity(
        self, case: StructuredPermissionParityCase
    ) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=self._static_policy(),
                shadow_model_enabled=True,
                project_root=Path.cwd(),
            )
        )

        result = engine.decide(
            case.tool_name,
            case.tool_input,
        )

        assert result.decision == case.expected_legacy_decision
        assert result.shadow_verdict is not None
        assert (
            result.shadow_verdict.decision,
            result.shadow_verdict.source,
        ) == case.expected_verdict
        assert (
            tuple(
                (constraint.source, constraint.decision)
                for constraint in result.shadow_verdict.constraints
            )
            == case.expected_constraints
        )
        assert result.shadow_diff == case.expected_shadow_diff

    def _static_policy(self) -> PermissionPolicy:
        """构造同时覆盖 allow、ask、deny 与 allowlist fallback 的规则。"""
        return PermissionPolicy(
            rules=(
                StaticPermission(tool="read_file", decision="allow"),
                StaticPermission(tool="write_file", decision="ask"),
                StaticPermission(
                    tool="edit_file", decision="deny", input_contains="static-policy"
                ),
                StaticPermission(
                    tool="edit_file", decision="ask", input_contains=".env"
                ),
            ),
            global_default="ask",
        )


if __name__ == "__main__":
    pytest.main()
