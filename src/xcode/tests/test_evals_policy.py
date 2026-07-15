"""Eval 非交互权限策略测试。"""

from xcode.harness.assembly.security import permission_policy_from_security
from xcode.harness.config import ModeRuleRuntimeConfig
from xcode.evals.policy import (
    EVAL_EXECUTION_MODE,
    approve_eval_action,
    build_eval_runtime,
)
from xcode.harness.config import XcodeRuntimeConfig


def test_eval_uses_build_mode_without_mutating_source_config() -> None:
    original = XcodeRuntimeConfig.model_validate(
        {
            "security": {
                "approval_policy": "always",
                "rules": [{"tool": "bash", "decision": "ask"}],
                "global_default": "ask",
            },
            "execution_modes": {
                "build": {
                    "rules": [
                        {"action": "bash", "effect": "ask"},
                    ]
                }
            },
        }
    )
    configured = build_eval_runtime(original)

    assert EVAL_EXECUTION_MODE == "build"
    assert original.execution_modes.build.rules == (
        ModeRuleRuntimeConfig(action="bash", effect="ask"),
    )
    assert configured.security.approval_policy == "never"
    assert configured.security.rules == ()
    assert configured.security.global_default is None
    assert configured.security.restricted_dirs == ()
    assert permission_policy_from_security(configured.security) is None
    assert not any(
        rule.effect == "ask" for rule in configured.execution_modes.build.rules
    )
    assert approve_eval_action(object(), {}).decision == "allow"


def test_eval_denies_external_lookup_paths() -> None:
    configured = build_eval_runtime(XcodeRuntimeConfig())
    rules = configured.execution_modes.build.rules

    assert any(rule.action == "websearch" and rule.effect == "deny" for rule in rules)
    assert any(rule.action == "webfetch" and rule.effect == "deny" for rule in rules)
    assert any(
        rule.action == "bash"
        and rule.command == "git"
        and rule.subcommand_in == ("clone", "fetch", "pull")
        for rule in rules
    )
