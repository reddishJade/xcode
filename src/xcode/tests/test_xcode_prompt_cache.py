from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from xcode.harness.agent_runtime.prompting import (
    SystemPromptBuilder,
    PromptContext,
)
from xcode.harness.skills import ToolSpec
import pytest


class TestXcodePromptCacheMemoization:
    """系统提示词三级缓存 (Memoization Cache) 单元测试。"""

    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.builder = SystemPromptBuilder()

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_stable_and_dynamic_regions_are_cached(self) -> None:
        """测试静态稳定区与动态环境区在连续构建中正确命中缓存，无多余耗时计算。"""
        # 准备两个空工具的 mock 注册表
        registry = (
            ToolSpec("tool_a", "desc a", "hint a", lambda x: "a"),
            ToolSpec("tool_b", "desc b", "hint b", lambda x: "b"),
        )

        context = PromptContext(
            project_root=self.root,
            registry=registry,
            question="What is 1+1?",
            modules=("identity", "tools", "environment", "cwd"),
        )

        # 首次构建
        first_prompt = self.builder.build(context)

        # 再次构建（完全相同的 context，应命中静态和动态缓存）
        # 我们用 patch build_tool_prompt 来验证是否走缓存逻辑
        with patch(
            "xcode.harness.agent_runtime.prompting.builder.build_tool_prompt"
        ) as mock_build_tool:
            second_prompt = self.builder.build(context)
            mock_build_tool.assert_not_called()  # 没有重新生成工具提示词，说明命中了静态缓存！

        assert first_prompt == second_prompt

    def test_stable_cache_invalidates_when_registry_changes(self) -> None:
        """测试当注册的工具变更时，静态缓存能正确失效并重建。"""
        registry_1 = (ToolSpec("tool_a", "desc a", "hint a", lambda x: "a"),)
        registry_2 = (
            ToolSpec("tool_a", "desc a", "hint a", lambda x: "a"),
            ToolSpec("tool_new", "desc new", "hint new", lambda x: "new"),
        )

        context_1 = PromptContext(self.root, registry_1, "test")
        context_2 = PromptContext(self.root, registry_2, "test")

        first_prompt = self.builder.build(context_1)
        second_prompt = self.builder.build(context_2)

        # 注册表增加了新工具，缓存应当更新重建，产生不同的系统提示词
        assert first_prompt != second_prompt
        assert "tool_new" in second_prompt

    def test_stable_cache_invalidates_when_tool_prompt_surface_changes(self) -> None:
        """测试同名工具的 prompt 可见内容变化时，静态缓存正确失效。"""
        registry_1 = (ToolSpec("tool_a", "desc a", "hint a", lambda x: "a"),)
        registry_2 = (
            ToolSpec(
                "tool_a",
                "desc changed",
                "hint a",
                lambda x: "a",
                prompt_snippet="snippet changed",
                prompt_guidelines=("Use tool_a for cache tests.",),
            ),
        )

        context_1 = PromptContext(self.root, registry_1, "test")
        context_2 = PromptContext(self.root, registry_2, "test")

        first_prompt = self.builder.build(context_1)
        second_prompt = self.builder.build(context_2)

        assert first_prompt != second_prompt
        assert "snippet changed" in second_prompt
        assert "Use tool_a for cache tests." in second_prompt

    def test_dynamic_cache_invalidates_when_project_root_changes(self) -> None:
        """测试当工作目录或项目路径变更时，动态环境缓存失效。"""
        context_1 = PromptContext(self.root, (), "test")

        with tempfile.TemporaryDirectory() as another_tmp:
            another_root = Path(another_tmp)
            context_2 = PromptContext(another_root, (), "test")

            first_prompt = self.builder.build(context_1)
            second_prompt = self.builder.build(context_2)

            assert first_prompt != second_prompt

    def test_volatile_region_always_recalculates(self) -> None:
        """测试易失区（如 notice 和 routed skills）即使在其他区域命中缓存时，依然每轮独立生成。"""
        # 利用 mock notice
        context_1 = PromptContext(self.root, (), "test", resumed_notice="Notice 1")
        context_2 = PromptContext(self.root, (), "test", resumed_notice="Notice 2")

        first_prompt = self.builder.build(context_1)
        second_prompt = self.builder.build(context_2)

        # 静态和动态区虽然被缓存，但 notices 作为易失区在两轮中分别独立生成
        assert "Notice 1" in first_prompt
        assert "Notice 2" not in first_prompt

        assert "Notice 2" in second_prompt
        assert "Notice 1" not in second_prompt


if __name__ == "__main__":
    pytest.main()
