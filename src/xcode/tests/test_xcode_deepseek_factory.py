from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xcode.ai.providers.factory import _build_llm_profile
from xcode.ai.providers.deepseek import DeepSeekProvider
from xcode.ai.providers.runtime import ProviderRuntime
import pytest


@dataclass(frozen=True)
class MockProfile:
    """模拟 ModelProfileProto 用于测试。"""

    transport: str = "deepseek_chat"
    chat_model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = "test-key"
    thinking: bool = True
    reasoning_effort: str | None = "high"
    clear_thinking: bool = False
    tool_stream: bool = True
    response_format: dict[str, Any] | None = None


class XcodeDeepSeekFactoryTests:
    def test_factory_preserves_response_format_for_deepseek(self) -> None:
        """验证 factory 构建 DeepSeek profile 时保留 response_format。"""
        profile = MockProfile(
            response_format={"type": "json_object"},
        )
        runtime = ProviderRuntime()

        provider = _build_llm_profile(
            profile,
            profile_name="test",
            env_files=(),
            runtime=runtime,
        )

        assert isinstance(provider, DeepSeekProvider)
        assert provider.response_format == {"type": "json_object"}

    def test_factory_passes_reasoning_effort_to_deepseek(self) -> None:
        """验证 factory 传递 reasoning_effort 到 DeepSeek。"""
        profile = MockProfile(reasoning_effort="medium")
        runtime = ProviderRuntime()

        provider = _build_llm_profile(
            profile,
            profile_name="test",
            env_files=(),
            runtime=runtime,
        )

        assert isinstance(provider, DeepSeekProvider)
        assert provider.reasoning_effort == "medium"


if __name__ == "__main__":
    pytest.main()
