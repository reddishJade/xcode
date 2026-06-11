from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xcode.harness.agent_runtime.prompting import (
    SystemPromptBuilder,
    PromptContext,
)
from xcode.harness.skills import ToolSpec


class TestXcodePromptCacheMemoization(unittest.TestCase):
    """系统提示词三级缓存 (Memoization Cache) 单元测试。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.builder = SystemPromptBuilder()

    def tearDown(self) -> None:
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

        self.assertEqual(first_prompt, second_prompt)

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
        self.assertNotEqual(first_prompt, second_prompt)
        self.assertIn("tool_new", second_prompt)

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

        self.assertNotEqual(first_prompt, second_prompt)
        self.assertIn("snippet changed", second_prompt)
        self.assertIn("Use tool_a for cache tests.", second_prompt)

    def test_stable_cache_invalidates_when_instructions_change(self) -> None:
        """测试当 AGENTS.md 或 CLAUDE.md 等配置文件内容变化时，静态缓存正确失效。"""
        agents_file = self.root / "AGENTS.md"
        agents_file.write_text("Instruction version 1", encoding="utf-8")

        context = PromptContext(self.root, (), "test")

        first_prompt = self.builder.build(context)
        self.assertIn("Instruction version 1", first_prompt)

        agents_file.write_text("Instruction version 2", encoding="utf-8")

        second_prompt = self.builder.build(context)
        self.assertNotEqual(first_prompt, second_prompt)
        self.assertIn("Instruction version 2", second_prompt)

    def test_stable_cache_ignores_instruction_mtime_when_content_matches(self) -> None:
        """测试项目指令内容不变时，单独修改 mtime 不会重建静态缓存。"""
        agents_file = self.root / "AGENTS.md"
        agents_file.write_text("Instruction version", encoding="utf-8")
        registry = (ToolSpec("tool_a", "desc a", "hint a", lambda x: "a"),)
        context = PromptContext(self.root, registry, "test")

        first_prompt = self.builder.build(context)
        stat = agents_file.stat()
        os.utime(agents_file, (stat.st_atime + 10, stat.st_mtime + 10))

        with patch(
            "xcode.harness.agent_runtime.prompting.builder.build_tool_prompt"
        ) as mock_build_tool:
            second_prompt = self.builder.build(context)
            mock_build_tool.assert_not_called()

        self.assertEqual(first_prompt, second_prompt)

    def test_dynamic_cache_invalidates_when_project_root_changes(self) -> None:
        """测试当工作目录或项目路径变更时，动态环境缓存失效。"""
        context_1 = PromptContext(self.root, (), "test")

        with tempfile.TemporaryDirectory() as another_tmp:
            another_root = Path(another_tmp)
            context_2 = PromptContext(another_root, (), "test")

            first_prompt = self.builder.build(context_1)
            second_prompt = self.builder.build(context_2)

            self.assertNotEqual(first_prompt, second_prompt)

    def test_volatile_region_always_recalculates(self) -> None:
        """测试易失区（如 notice 和 routed skills）即使在其他区域命中缓存时，依然每轮独立生成。"""
        # 利用 mock notice
        context_1 = PromptContext(self.root, (), "test", resumed_notice="Notice 1")
        context_2 = PromptContext(self.root, (), "test", resumed_notice="Notice 2")

        first_prompt = self.builder.build(context_1)
        second_prompt = self.builder.build(context_2)

        # 静态和动态区虽然被缓存，但 notices 作为易失区在两轮中分别独立生成
        self.assertIn("Notice 1", first_prompt)
        self.assertNotIn("Notice 2", first_prompt)

        self.assertIn("Notice 2", second_prompt)
        self.assertNotIn("Notice 1", second_prompt)


if __name__ == "__main__":
    unittest.main()
