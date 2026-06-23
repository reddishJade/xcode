"""SkillRegistry——技能发现、索引、懒加载的唯一后端。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from xcode.harness.agent_skills.models import SkillDef, SkillDiagnostic, SkillSummary
from xcode.harness.agent_skills.parsing import (
    find_body_start,
    frontmatter_metadata,
    frontmatter_optional_string,
    parse_frontmatter,
)
from xcode.harness.agent_skills.discovery import (
    load_reference_text,
    scan_skill_references,
    scan_skill_resources,
    source_for_priority,
)
from xcode.harness.skill_activation import activated_skill_names

logger = logging.getLogger(__name__)


class SkillRegistry:
    """技能发现、索引、懒加载的唯一后端。

    使用方式：
        registry = SkillRegistry()
        registry.discover(search_dirs)
        summaries = registry.list_summaries()
        skill = registry.load("code-review")
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDef] = {}
        self._ref_paths: dict[str, dict[str, Path]] = {}
        self._activated: set[str] = set()

    def discover(
        self,
        search_dirs: list[tuple[Path, int]],
    ) -> list[SkillDiagnostic]:
        """扫描目录查找 SKILL.md 文件，缓存元数据。

        相同 name 按搜索路径优先级 first-wins。重复时记录警告。
        跳过 malformed frontmatter 或缺少必填字段的技能。
        """
        diagnostics: list[SkillDiagnostic] = []
        all_dirs = sorted(search_dirs, key=lambda x: x[1])
        for search_dir, priority in all_dirs:
            if not search_dir.is_dir():
                continue
            source = source_for_priority(priority)
            try:
                for file_path in sorted(search_dir.rglob("SKILL.md")):
                    try:
                        text = file_path.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        msg = f"Failed to read {file_path}; skipping"
                        logger.warning(msg)
                        diagnostics.append(
                            SkillDiagnostic(
                                type="warning",
                                code="read_failed",
                                message=msg,
                                path=str(file_path),
                            )
                        )
                        continue
                    frontmatter = parse_frontmatter(
                        text,
                        skill_directory_name=file_path.parent.name,
                        source=file_path,
                    )
                    if frontmatter is None:
                        diagnostics.append(
                            SkillDiagnostic(
                                type="warning",
                                code="parse_failed",
                                message=f"Failed to parse frontmatter in {file_path}",
                                path=str(file_path),
                            )
                        )
                        continue
                    name = str(frontmatter["name"])
                    if name in self._skills:
                        logger.warning(
                            "Duplicate skill %r from %s; keeping %s",
                            name,
                            file_path,
                            self._skills[name].file_path,
                        )
                        continue
                    body_start = find_body_start(text)

                    refs, ref_map = scan_skill_references(file_path.parent)
                    scripts = scan_skill_resources(file_path.parent, "scripts")
                    assets = scan_skill_resources(file_path.parent, "assets")
                    raw_disable = frontmatter.get("disable_model_invocation", False)

                    self._skills[name] = SkillDef(
                        name=name,
                        description=str(frontmatter["description"]),
                        disable_model_invocation=bool(raw_disable),
                        source=source,
                        license=frontmatter_optional_string(frontmatter, "license"),
                        compatibility=frontmatter_optional_string(
                            frontmatter, "compatibility"
                        ),
                        metadata=frontmatter_metadata(frontmatter),
                        allowed_tools=frontmatter_optional_string(
                            frontmatter, "allowed-tools"
                        ),
                        file_path=file_path,
                        frontmatter=frontmatter,
                        content=body_start,
                        references=tuple(refs),
                        scripts=scripts,
                        assets=assets,
                    )
                    if ref_map:
                        self._ref_paths[name] = ref_map
            except Exception:
                logger.exception("SkillRegistry: failed to scan %s", search_dir)
                continue
        return diagnostics

    def list_summaries(self) -> list[SkillSummary]:
        """返回所有未被 disable_model_invocation 禁止的可见技能摘要。"""
        return [
            skill.to_summary()
            for skill in self._skills.values()
            if not skill.disable_model_invocation
        ]

    def available_names(self) -> tuple[str, ...]:
        """返回可向模型披露和激活的技能名称。"""
        return tuple(sorted(summary.name for summary in self.list_summaries()))

    def contains(self, skill_name: str) -> bool:
        """判断技能名称是否已被发现。"""
        return skill_name in self._skills

    def is_available(self, skill_name: str) -> bool:
        """判断技能是否允许用户或模型显式激活。"""
        skill = self._skills.get(skill_name)
        return skill is not None and not skill.disable_model_invocation

    def activate(self, skill_name: str) -> tuple[SkillDef | None, bool]:
        """激活技能并返回 (技能定义, 是否已激活)。"""
        skill = self.load(skill_name)
        if skill is None:
            return None, False
        already_activated = skill_name in self._activated
        self._activated.add(skill_name)
        return skill, already_activated

    def clear_activations(self) -> None:
        """清空当前会话的技能激活状态。"""
        self._activated.clear()

    def restore_activations(self, messages: Sequence[object]) -> None:
        """从历史或压缩上下文中的激活标记恢复状态。"""
        self.clear_activations()
        for message in messages:
            for name in activated_skill_names(message):
                if name in self._skills:
                    self._activated.add(name)

    def activated_names(self) -> tuple[str, ...]:
        """返回当前会话已激活技能名称。"""
        return tuple(sorted(self._activated))

    def load(self, skill_name: str) -> SkillDef | None:
        """懒加载技能正文。

        首次调用时读取文件内容并缓存。
        返回 None 表示技能不存在。
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            return None
        if skill.content is None:
            try:
                text = skill.file_path.read_text(encoding="utf-8", errors="replace")
                body_start = find_body_start(text)
                object.__setattr__(skill, "content", body_start)
            except Exception:
                logger.exception(
                    "Failed to load skill %r from %s",
                    skill_name,
                    skill.file_path,
                )
                return None
        return skill

    def load_reference(self, skill_name: str, ref_name: str) -> str | None:
        """加载指定技能中的引用文件内容。

        仅允许加载在 discover() 阶段发现且未跳过的引用。
        返回内容受 _REFERENCE_MAX_BYTES 截断保护。
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            return None

        ref_map = self._ref_paths.get(skill_name, {})
        ref_path = ref_map.get(ref_name)
        if ref_path is None:
            return None

        content, _ = load_reference_text(ref_path)
        return content
