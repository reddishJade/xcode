"""Eval 专用的非交互权限策略。

宿主安全由 bubblewrap 文件系统边界承担；本策略让 workspace 内工程操作不被 HITL
阻塞，同时减少公开历史任务通过外部检索发现参考修复的污染风险。
"""

from __future__ import annotations

from xcode.harness.config import ModeRuleRuntimeConfig, XcodeRuntimeConfig
from xcode.harness.observability import HITLResult


EVAL_EXECUTION_MODE = "build"


def approve_eval_action(_tool: object, _action_input: dict[str, object]) -> HITLResult:
    """在 OS 沙盒内非交互放行残余 ask，显式 deny 不会进入此回调。"""
    return HITLResult("allow", "once")


def build_eval_runtime(runtime: XcodeRuntimeConfig) -> XcodeRuntimeConfig:
    """返回带独立 Eval build ruleset 的不可变配置副本。"""
    eval_rules = (
        ModeRuleRuntimeConfig(action="websearch", effect="deny"),
        ModeRuleRuntimeConfig(action="webfetch", effect="deny"),
        ModeRuleRuntimeConfig(action="bash", effect="deny", command="curl"),
        ModeRuleRuntimeConfig(action="bash", effect="deny", command="wget"),
        ModeRuleRuntimeConfig(
            action="bash",
            effect="deny",
            command="git",
            subcommand_in=("clone", "fetch", "pull"),
        ),
    )
    return runtime.model_copy(
        update={
            "security": runtime.security.model_copy(
                update={
                    "approval_policy": "never",
                    "rules": (),
                    "global_default": None,
                    "restricted_dirs": (),
                    "external_directories": (),
                }
            ),
            "execution_modes": runtime.execution_modes.model_copy(
                update={
                    "build": runtime.execution_modes.build.model_copy(
                        update={"rules": eval_rules}
                    )
                }
            ),
        }
    )
