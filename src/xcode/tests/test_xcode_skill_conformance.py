"""Skill 生态系统兼容性冒烟测试。

使用 on-disk fixture skill 目录模拟真实的克隆技能仓库，
验证 Xcode 可以发现/列出/加载真实生态形状的 Skill 包。

测试覆盖：
- 有效技能（SKILL.md + references/）可被发现和加载
- 缺少 SKILL.md 的目录不被发现
- 大文件 references/ 不影响发现和加载
- scripts/ 目录在发现期间不被执行（安全约束）
- load_skill 工具只加载技能正文和引用材料
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
import unittest

from xcode.harness.skills_registry import (
    SkillIndexCollector,
    SkillRegistry,
    build_load_skill_tool,
    build_skill_search_dirs,
)


SKILL_WITH_REFS_BODY = """---
name: code-review
description: Review code changes for bugs and risk.
---

# Code Review Skill

Always check:
1. Are there security concerns?
2. Are there edge cases?
3. Are tests sufficient?
"""

REFERENCE_GUIDE = """# Code Review Checklist

## Security
- Input validation
- Authentication checks
- Data leakage

## Performance
- N+1 queries
- Memory leaks
"""

DANGEROUS_SCRIPT = "#!/bin/sh\nrm -rf /tmp/danger\n"


def _make_skill_tree(
    base: Path,
    *parts: str,
    skil_md_content: str = "",
    references: dict[str, str] | None = None,
    scripts: dict[str, str] | None = None,
) -> Path:
    """在 base/parts.../ 下创建技能目录树并返回路径。

    Args:
        base: 根目录
        parts: 子目录路径片段
        skil_md_content: SKILL.md 文件内容（空字符串表示不创建）
        references: references/ 下的文件名 -> 内容映射
        scripts: scripts/ 下的文件名 -> 内容映射
    """
    skill_dir = base.joinpath(*parts)
    skill_dir.mkdir(parents=True, exist_ok=True)

    if skil_md_content:
        (skill_dir / "SKILL.md").write_text(skil_md_content, encoding="utf-8")

    if references:
        ref_dir = skill_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for name, content in references.items():
            (ref_dir / name).write_text(content, encoding="utf-8")

    if scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in scripts.items():
            path = scripts_dir / name
            path.write_text(content, encoding="utf-8")
            os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

    return skill_dir


class TestSkillConformance(unittest.TestCase):
    """Skill 生态系统兼容性冒烟测试。"""

    # ── 有效技能：SKILL.md + references/ ──

    def test_valid_skill_with_references_is_discovered(self) -> None:
        """带有 SKILL.md 和 references/ 目录的有效技能可被发现。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "code-review",
                skil_md_content=SKILL_WITH_REFS_BODY,
                references={"checklist.md": REFERENCE_GUIDE},
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            summaries = registry.list_summaries()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].name, "code-review")

    def test_valid_skill_with_references_loads_body(self) -> None:
        """带 references/ 的技能，load_skill 只返回正文，不含引用材料。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "code-review",
                skil_md_content=SKILL_WITH_REFS_BODY,
                references={"checklist.md": REFERENCE_GUIDE},
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            tool = build_load_skill_tool(registry)
            output = tool.handler({"name": "code-review"})

            self.assertIn("Code Review Skill", output)
            self.assertIn("security concerns", output)
            self.assertIn("edge cases", output)
            # references/ 材料不应出现在 load_skill 输出中
            self.assertNotIn("Input validation", output)
            self.assertNotIn("N+1 queries", output)
            self.assertNotIn("Authentication checks", output)

    def test_references_directory_does_not_block_discovery(self) -> None:
        """references/ 目录自身不会阻止 SKILL.md 发现。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "code-review",
                skil_md_content=SKILL_WITH_REFS_BODY,
                references={
                    "guide.md": REFERENCE_GUIDE,
                    "examples.md": "# Examples\n\nExample 1\n",
                },
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            skill = registry.load("code-review")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertIn("Code Review Skill", skill.content or "")

    # ── 缺失 SKILL.md ──

    def test_directory_without_skill_md_not_discovered(self) -> None:
        """仅有 references/ 但无 SKILL.md 的目录不被发现为技能。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "refs-only",
                references={"guide.md": REFERENCE_GUIDE},
            )
            # 显式确保无 SKILL.md
            self.assertFalse((skill_dir / "SKILL.md").exists())

            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            summaries = registry.list_summaries()
            self.assertEqual(len(summaries), 0)

    def test_skill_search_ignores_non_skill_directories(self) -> None:
        """无 SKILL.md 的其他目录不被注册为技能。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 纯目录但无 SKILL.md
            (root / ".xcode" / "skills" / "misc").mkdir(parents=True, exist_ok=True)
            # 真实技能放在另一边
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "real",
                skil_md_content="---\nname: real\ndescription: Real skill.\n---\n\nBody.",
            )

            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            summaries = registry.list_summaries()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].name, "real")

    # ── 大文件 references/ ──

    def test_large_reference_file_does_not_block_discovery(self) -> None:
        """references/ 中的大文件不影响 SKILL.md 发现和加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "big-ref",
                skil_md_content=SKILL_WITH_REFS_BODY,
                references={"large.bin": "x" * 1024 * 512},  # 512KB
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            summaries = registry.list_summaries()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].name, "code-review")

            skill = registry.load("code-review")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertIn("Code Review Skill", skill.content or "")

    # ── scripts/ 安全约束 ──

    def test_scripts_directory_not_executed_during_discovery(self) -> None:
        """scripts/ 目录中的脚本在发现期间不被执行。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "with-scripts",
                skil_md_content=SKILL_WITH_REFS_BODY,
                scripts={"run.sh": DANGEROUS_SCRIPT},
            )
            # 验证脚本文件存在且可执行
            script_path = (
                root / ".xcode" / "skills" / "with-scripts" / "scripts" / "run.sh"
            )
            self.assertTrue(script_path.exists())
            self.assertTrue(os.access(script_path, os.X_OK))

            # 发现不应执行任何脚本
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))

            # 技能正常发现
            summaries = registry.list_summaries()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].name, "code-review")

            # 验证无害——临时目录仍存在
            self.assertTrue(tmp.startswith(tempfile.gettempdir()))
            self.assertTrue(Path(tmp).is_dir())

    def test_script_permissions_not_inherited_by_skill_data(self) -> None:
        """脚本文件的可执行权限不影响技能元数据。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "mixed",
                skil_md_content=SKILL_WITH_REFS_BODY,
                scripts={"setup.sh": "#!/bin/sh\necho setup\n"},
                references={"readme.md": "# Readme\n"},
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))

            skill = registry.load("code-review")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(
                skill.description, "Review code changes for bugs and risk."
            )

    # ── 多种子目录共存 ──

    def test_multiple_skills_with_ancillary_directories(self) -> None:
        """多个技能各自带有 references/ 和 scripts/ 目录可正常发现。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "review",
                skil_md_content="---\nname: code-review\ndescription: Review code.\n---\n\nReview body.",
                references={"checklist.md": REFERENCE_GUIDE},
            )
            _make_skill_tree(
                root,
                ".xcode",
                "skills",
                "deploy",
                skil_md_content="---\nname: deploy\ndescription: Deploy to production.\n---\n\nDeploy body.",
                scripts={"deploy.sh": "#!/bin/sh\necho deploying\n"},
                references={"runbook.md": "# Runbook\n"},
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            summaries = registry.list_summaries()
            names = {s.name for s in summaries}
            self.assertEqual(names, {"code-review", "deploy"})

    # ── 生态系统形状兼容 ──

    def test_ecosystem_shape_roundtrip(self) -> None:
        """完整生态系统形状：克隆 -> 发现 -> 摘要 -> 加载 -> 使用。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 模拟从仓库克隆的技能目录
            skill_root = root / ".xcode" / "skills" / "automation"
            skill_root.mkdir(parents=True, exist_ok=True)
            (skill_root / "SKILL.md").write_text(
                "---\nname: automation\ndescription: Automate workflows.\n---\n\n"
                "# Automation Skill\n\nUse this skill to automate tasks.\n\n"
                "## Steps\n1. Plan\n2. Execute\n3. Verify\n",
                encoding="utf-8",
            )
            ref_dir = skill_root / "references"
            ref_dir.mkdir()
            (ref_dir / "guide.md").write_text("# Reference Guide\n\nDetails here.\n")
            scripts_dir = skill_root / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "setup.sh").write_text(
                "#!/bin/sh\necho setup\n", encoding="utf-8"
            )
            os.chmod(scripts_dir / "setup.sh", stat.S_IRWXU)

            # 步骤 1: 发现
            registry = SkillRegistry()
            registry.discover([(root / ".xcode" / "skills", 0)])

            # 步骤 2: 摘要（列出可用技能，不含正文）
            summaries = registry.list_summaries()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].name, "automation")
            self.assertIn("Automate", summaries[0].description)

            # 步骤 3: IndexCollector 生成摘要块
            collector = SkillIndexCollector(registry)
            blocks = collector.collect(object())
            self.assertIn("automation", blocks[0].content)
            self.assertNotIn("Automation Skill", blocks[0].content)

            # 步骤 4: 加载技能正文
            skill = registry.load("automation")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertIn("Automation Skill", skill.content or "")
            self.assertIn("Plan", skill.content or "")

            # 步骤 5: 通过 load_skill 工具使用
            tool = build_load_skill_tool(registry)
            output = tool.handler({"name": "automation"})
            self.assertIn("Automation Skill", output)
            # 引用材料不在 load_skill 返回中
            self.assertNotIn("Reference Guide", output)
            self.assertNotIn("Details here", output)


if __name__ == "__main__":
    unittest.main()
