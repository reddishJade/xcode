"""模型注册表与选择语法单元测试。"""

from __future__ import annotations

import pytest

from xcode.ai.models import (
    ModelMode,
    effective_rollover_threshold,
    get_model,
    get_models,
    get_providers,
    parse_model_mode,
    resolve_model,
)


class TestParseModelMode:
    def test_basic_model(self) -> None:
        assert parse_model_mode("gpt-4") == ModelMode(model="gpt-4")

    def test_provider_model(self) -> None:
        assert parse_model_mode("openai/gpt-4") == ModelMode(
            model="gpt-4", provider="openai"
        )

    def test_provider_model_thinking(self) -> None:
        assert parse_model_mode("openai/gpt-4:low") == ModelMode(
            model="gpt-4", provider="openai", thinking_level="low"
        )

    def test_model_thinking_no_provider(self) -> None:
        assert parse_model_mode("gpt-4:high") == ModelMode(
            model="gpt-4", thinking_level="high"
        )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_model_mode("")

    def test_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_model_mode("   ")

    def test_invalid_thinking_level_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid thinking level"):
            parse_model_mode("gpt-4:ultra")

    def test_all_valid_thinking_levels(self) -> None:
        for level in (
            "off",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            result = parse_model_mode(f"gpt-4:{level}")
            assert result.thinking_level == level

    def test_provider_only_no_model_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_model_mode("openai/")

    def test_thinking_level_case_normalized(self) -> None:
        assert parse_model_mode("gpt-4:HIGH").thinking_level == "high"


class TestRolloverThreshold:
    @pytest.mark.parametrize(
        ("model", "expected"),
        (
            ("gpt-5.5", 997_500),
            ("chatglm/glm-5.1", 183_616),
        ),
    )
    def test_known_model_uses_its_registered_context_window(
        self, model: str, expected: int
    ) -> None:
        threshold = effective_rollover_threshold(
            model,
            reserve_tokens=16_384,
            trigger_ratio=0.95,
        )

        assert threshold == expected

    def test_reserve_remains_hard_upper_bound(self) -> None:
        threshold = effective_rollover_threshold(
            "chatglm/glm-5.1",
            reserve_tokens=80_000,
            trigger_ratio=0.95,
        )

        assert threshold == 120_000

    def test_unknown_model_keeps_fallback_threshold(self) -> None:
        threshold = effective_rollover_threshold(
            "unknown-model",
            fallback_threshold=32_000,
            trigger_ratio=0.95,
        )

        assert threshold == 32_000

    def test_context_window_override_wins(self) -> None:
        threshold = effective_rollover_threshold(
            "gpt-5.5",
            reserve_tokens=16_384,
            trigger_ratio=0.95,
            context_window_override=262_144,
        )

        assert threshold == 245_760

    def test_context_window_override_respects_reserve(self) -> None:
        threshold = effective_rollover_threshold(
            "gpt-5.5",
            reserve_tokens=200_000,
            context_window_override=262_144,
        )

        assert threshold == 62_144

    def test_non_positive_override_falls_back_to_registry(self) -> None:
        threshold = effective_rollover_threshold(
            "gpt-5.5",
            reserve_tokens=16_384,
            trigger_ratio=0.95,
            context_window_override=0,
        )

        assert threshold == 997_500

    def test_specific_model_id_wins_over_prefix_model(self) -> None:
        threshold = effective_rollover_threshold(
            "openai/gpt-5.4-mini",
            reserve_tokens=0,
            trigger_ratio=1,
        )

        assert threshold == 400_000


class TestResolveModel:
    def test_exact_match(self) -> None:
        model = resolve_model("openai", "gpt-5.5")
        assert model is not None
        assert model.id == "gpt-5.5"

    def test_fallback_to_first(self) -> None:
        model = resolve_model("openai", "nonexistent-model")
        assert model is not None
        assert model.id == "gpt-5.5"  # first in openai dict

    def test_unknown_provider_returns_generic(self) -> None:
        model = resolve_model("unknown_provider", "some-model")
        assert model.id == "some-model"
        assert model.provider == "unknown_provider"

    def test_empty_model_id(self) -> None:
        model = resolve_model("unknown_provider", "")
        assert model.id == ""


class TestRegistryAccess:
    def test_get_providers(self) -> None:
        providers = get_providers()
        assert "openai" in providers
        assert "deepseek" in providers
        assert "chatglm" in providers
        assert "mimo" in providers

    def test_get_models_openai(self) -> None:
        models = get_models("openai")
        ids = [m.id for m in models]
        assert "gpt-5.5" in ids
        assert "gpt-5.4" in ids
        assert "gpt-5.4-mini" in ids

    def test_get_models_unknown_provider(self) -> None:
        assert get_models("nonexistent") == []

    def test_get_model_existing(self) -> None:
        model = get_model("deepseek", "deepseek-v4-pro")
        assert model is not None
        assert model.name == "DeepSeek V4 Pro"

    def test_get_model_nonexistent(self) -> None:
        assert get_model("openai", "does-not-exist") is None
