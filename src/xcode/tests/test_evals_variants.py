"""Eval Variant 必须对应真实 runtime 差异。"""

from types import SimpleNamespace
from typing import cast

from xcode.evals.variants import (
    build_eval_variant_runtime,
    configure_eval_variant_app,
    EvalVariantError,
    variant_capabilities,
)
from xcode.harness.app import XcodeApp
from xcode.harness.config import XcodeRuntimeConfig


def test_full_variant_preserves_runtime_without_copy() -> None:
    runtime = XcodeRuntimeConfig()

    assert build_eval_variant_runtime(runtime, "full") is runtime


def test_minimal_variant_disables_only_declared_enhancements() -> None:
    runtime = XcodeRuntimeConfig()

    minimal = build_eval_variant_runtime(runtime, "minimal")
    capabilities = variant_capabilities("minimal")

    assert (
        minimal.provider.model_profiles["main"]
        == runtime.provider.model_profiles["main"]
    )
    assert "fallback" not in minimal.provider.model_profiles
    assert minimal.agent.tool_workers == 1
    assert minimal.agent.watchdog_repeated_tool_limit == 0
    assert minimal.request_hygiene.enabled is False
    assert "git_preflight" not in minimal.prompt.modules
    assert "contextual_retrieval" not in minimal.prompt.modules
    assert capabilities["compaction"] is False
    assert capabilities["provider_fallback"] is False
    assert capabilities["tools"] is True
    assert capabilities["permission_feedback"] is True


def test_minimal_variant_removes_runtime_compactor() -> None:
    app = SimpleNamespace(
        agent=SimpleNamespace(
            compactor=object(),
            _runtime=SimpleNamespace(compactor=object()),
            _compact_controller=object(),
        )
    )

    configure_eval_variant_app(cast(XcodeApp, app), "minimal")

    assert app.agent.compactor is None
    assert app.agent._runtime.compactor is None
    assert app.agent._compact_controller is None


def test_no_compaction_variant_preserves_other_capabilities() -> None:
    runtime = XcodeRuntimeConfig()
    variant = build_eval_variant_runtime(runtime, "no-compaction")
    capabilities = variant_capabilities("no-compaction")

    assert variant is runtime
    assert capabilities["compaction"] is False
    assert capabilities["provider_fallback"] is True
    assert capabilities["parallel_tools"] is True
    assert capabilities["request_hygiene"] is True
    assert capabilities["repeated_tool_watchdog"] is True
    assert capabilities["contextual_retrieval_prompt"] is True


def test_unknown_variant_is_rejected() -> None:
    try:
        build_eval_variant_runtime(XcodeRuntimeConfig(), "unknown")
    except EvalVariantError as error:
        assert "unknown Eval variant" in str(error)
    else:
        raise AssertionError("unknown variant was accepted")
