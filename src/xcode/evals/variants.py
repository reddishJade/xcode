"""Eval Variant 的可执行配置，而非仅用于展示的标签。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xcode.harness.config import XcodeRuntimeConfig

if TYPE_CHECKING:
    from xcode.harness.app import XcodeApp

FULL_VARIANT_ID = "full"
MINIMAL_VARIANT_ID = "minimal"
EVAL_VARIANT_PROFILE_VERSION = "v1"

_MINIMAL_PROMPT_EXCLUSIONS = frozenset({"git_preflight", "contextual_retrieval"})


class EvalVariantError(ValueError):
    """请求了未定义或无法执行的 Eval Variant。"""


def build_eval_variant_runtime(
    runtime: XcodeRuntimeConfig,
    variant_id: str,
) -> XcodeRuntimeConfig:
    """生成 Variant 的正式 runtime 配置快照。"""
    if variant_id == FULL_VARIANT_ID:
        return runtime
    if variant_id != MINIMAL_VARIANT_ID:
        raise EvalVariantError(f"unknown Eval variant: {variant_id}")

    profiles = {
        name: profile
        for name, profile in runtime.provider.model_profiles.items()
        if name != "fallback"
    }
    modules = tuple(
        module
        for module in runtime.prompt.modules
        if module not in _MINIMAL_PROMPT_EXCLUSIONS
    )
    return runtime.model_copy(
        update={
            "provider": runtime.provider.model_copy(
                update={"model_profiles": profiles}
            ),
            "agent": runtime.agent.model_copy(
                update={
                    "tool_workers": 1,
                    "watchdog_repeated_tool_limit": 0,
                }
            ),
            "prompt": runtime.prompt.model_copy(update={"modules": modules}),
            "request_hygiene": runtime.request_hygiene.model_copy(
                update={"enabled": False}
            ),
        }
    )


def configure_eval_variant_app(app: XcodeApp, variant_id: str) -> None:
    """应用无法仅靠序列化配置表达的运行时差异。"""
    if variant_id == FULL_VARIANT_ID:
        return
    if variant_id != MINIMAL_VARIANT_ID:
        raise EvalVariantError(f"unknown Eval variant: {variant_id}")
    app.agent.compactor = None
    app.agent._runtime.compactor = None
    app.agent._compact_controller = None


def variant_capabilities(variant_id: str) -> dict[str, bool]:
    """返回与实际执行路径一一对应的能力声明。"""
    shared = {
        "context_assembly": True,
        "tools": True,
        "permission_feedback": True,
        "session": True,
        "mcp": True,
        "memory": True,
    }
    if variant_id == FULL_VARIANT_ID:
        return {
            **shared,
            "compaction": True,
            "provider_fallback": True,
            "parallel_tools": True,
            "request_hygiene": True,
            "repeated_tool_watchdog": True,
            "contextual_retrieval_prompt": True,
        }
    if variant_id == MINIMAL_VARIANT_ID:
        return {
            **shared,
            "compaction": False,
            "provider_fallback": False,
            "parallel_tools": False,
            "request_hygiene": False,
            "repeated_tool_watchdog": False,
            "contextual_retrieval_prompt": False,
        }
    raise EvalVariantError(f"unknown Eval variant: {variant_id}")
