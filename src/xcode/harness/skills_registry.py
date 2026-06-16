"""技能注册表——技能发现、索引、懒加载的唯一后端。

提供 SkillRegistry 作为技能管理单一后端，SkillIndexCollector 通过它
注入摘要信息，load_skill 工具通过它执行懒加载。
"""

from __future__ import annotations

import logging
import os as _os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import xml.sax.saxutils as _xml

import yaml

from xcode.agent.context_assembly import (
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextPriority,
)
from xcode.harness.skills import ToolInput, ToolSpec

logger = logging.getLogger(__name__)

# ── 搜索路径优先级 ──
# 0: <project>/.xcode/skills/      项目用户技能（最高）
# 1: <project>/.agents/skills/     旧版项目技能
# 2: <project>/skills/             内置技能
# 3: ~/.xcode/skills/              全局用户技能
# 4: ~/.agents/skills/             旧版全局技能


def build_skill_search_dirs(project_root: Path | None) -> list[tuple[Path, int]]:
    """构建优先级排序的技能搜索目录列表。

    返回 [(Path, priority), ...]，priority 越小优先级越高。
    """
    dirs: list[tuple[Path, int]] = []
    if project_root is not None:
        dirs.append((project_root / ".xcode" / "skills", 0))
        dirs.append((project_root / ".agents" / "skills", 1))
        dirs.append((project_root / "skills", 2))
    home = Path.home()
    dirs.append((home / ".xcode" / "skills", 3))
    dirs.append((home / ".agents" / "skills", 4))
    return dirs


# ── 数据模型 ──


@dataclass(frozen=True)
class SkillSummary:
    """技能摘要——不含正文的轻量元数据。"""

    name: str
    description: str
    hidden: bool = False


@dataclass(frozen=True)
class SkillReference:
    """技能引用文件元数据。

    name 是 references/ 内部的规范化 POSIX 相对路径，
    如 "checklist.md" 或 "subdir/guide.md"。
    跳过（hidden/symlink/binary）的文件以 skipped=True 标记。
    """

    name: str
    size: int = 0
    truncated: bool = False
    skipped: bool = False
    skipped_reason: str | None = None


@dataclass(frozen=True)
class SkillDef:
    """技能的完整定义，包含懒加载的正文内容和引用元数据。"""

    name: str
    description: str
    file_path: Path
    hidden: bool = False
    frontmatter: dict[str, object] = field(default_factory=dict)
    content: str | None = None
    references: tuple[SkillReference, ...] = ()

    def to_summary(self) -> SkillSummary:
        return SkillSummary(
            name=self.name,
            description=self.description,
            hidden=self.hidden,
        )


# ── Frontmatter 解析（YAML） ──
# 使用 yaml.safe_load 解析 SKILL.md 的 YAML frontmatter。
# 仅使用 name（必填字符串）、description（必填字符串）、
# hidden（可选布尔值，默认 false）。其余字段静默忽略。


def _parse_frontmatter(text: str) -> dict[str, Any] | None:
    """解析 SKILL.md 的 YAML frontmatter。

    从 --- 分隔符之间提取 YAML，返回 name/description/hidden 的 dict。
    无效 YAML、非 dict frontmatter、或缺少必填字段时返回 None。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return None

    yaml_text = "\n".join(lines[1:end_idx])
    try:
        parsed = yaml.safe_load(yaml_text)
    except Exception:
        logger.warning("Invalid YAML frontmatter; skipping")
        return None

    if not isinstance(parsed, dict):
        logger.warning("Frontmatter is not a dict; skipping")
        return None

    name = parsed.get("name")
    description = parsed.get("description")
    hidden = parsed.get("hidden", False)

    if not isinstance(name, str) or not name.strip():
        logger.warning("Skill frontmatter 'name' must be a non-empty string")
        return None
    if not isinstance(description, str) or not description.strip():
        logger.warning("Skill frontmatter 'description' must be a non-empty string")
        return None
    if not isinstance(hidden, bool):
        hidden = False

    return {"name": name, "description": description, "hidden": hidden}


# ── 引用文件扫描与加载 ──

_REFERENCE_MAX_BYTES = 50 * 1024  # 单个引用文件大小上限


def _collect_reference_files(ref_dir: Path) -> list[Path]:
    """收集 references/ 下的文件，不跟踪符号链接，跳过隐藏目录。

    返回按遍历顺序的文件列表。隐藏文件、符号链接文件仍包含在内，
    由上层调用 _scan_skill_references 分类处理。
    """
    files: list[Path] = []
    try:
        for dirpath, dirnames, filenames in _os.walk(ref_dir, followlinks=False):
            dir_path = Path(dirpath)
            # 不进入隐藏目录或符号链接目录
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not d.startswith(".") and not (dir_path / d).is_symlink()
            )
            for f in sorted(filenames):
                files.append(dir_path / f)
    except OSError:
        pass
    return files


def _is_binary_file(path: Path) -> bool:
    """检测文件是否为二进制（含空字节或非 UTF-8 可解码）。"""
    try:
        data = path.read_bytes()[:1024]
        if b"\0" in data:
            return True
        data.decode("utf-8")
        return False
    except (OSError, UnicodeDecodeError):
        return True


def _load_reference_text(path: Path) -> tuple[str | None, bool]:
    """读取引用文件内容，应用大小预算（截断）。

    返回 (content, truncated)，读取失败时 content 为 None。
    内容以 UTF-8 解码，含替换字符。
    """
    try:
        with open(path, "rb") as f:
            data = f.read(_REFERENCE_MAX_BYTES + 1)
        truncated = len(data) > _REFERENCE_MAX_BYTES
        if truncated:
            data = data[:_REFERENCE_MAX_BYTES]
        text = data.decode("utf-8", errors="replace")
        return text, truncated
    except (OSError, MemoryError):
        return None, False


def _scan_skill_references(
    skill_root: Path,
) -> tuple[list[SkillReference], dict[str, Path]]:
    """扫描技能根目录下的 references/ 并返回元数据。

    第二个返回值是引用名到解析路径的内部映射，用于懒加载。
    """
    ref_dir = skill_root / "references"
    if not ref_dir.is_dir():
        return [], {}

    references: list[SkillReference] = []
    ref_map: dict[str, Path] = {}
    seen_names: set[str] = set()

    for file_path in _collect_reference_files(ref_dir):
        rel_path = str(file_path.relative_to(ref_dir).as_posix())

        if rel_path in seen_names:
            logger.warning("Duplicate reference name %r; skipping both", rel_path)
            continue
        seen_names.add(rel_path)

        if any(part.startswith(".") for part in rel_path.split("/")):
            references.append(
                SkillReference(name=rel_path, skipped=True, skipped_reason="hidden")
            )
            continue

        if file_path.is_symlink():
            references.append(
                SkillReference(name=rel_path, skipped=True, skipped_reason="symlink")
            )
            continue

        try:
            st = file_path.stat()
        except OSError:
            references.append(
                SkillReference(name=rel_path, skipped=True, skipped_reason="unreadable")
            )
            continue

        if _is_binary_file(file_path):
            references.append(
                SkillReference(
                    name=rel_path,
                    size=st.st_size,
                    skipped=True,
                    skipped_reason="binary",
                )
            )
            continue

        truncated = st.st_size > _REFERENCE_MAX_BYTES
        references.append(
            SkillReference(
                name=rel_path,
                size=st.st_size,
                truncated=truncated,
            )
        )
        ref_map[rel_path] = file_path

    references.sort(key=lambda r: r.name)
    return references, ref_map


# ── SkillRegistry ──


class SkillRegistry:
    """技能发现、索引、懒加载的唯一后端。

    使用方式：
        registry = SkillRegistry()
        registry.discover(search_dirs)
        summaries = registry.list_summaries()
        skill = registry.load("code-review")  # permissioned
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDef] = {}
        self._ref_paths: dict[str, dict[str, Path]] = {}

    def discover(self, search_dirs: list[tuple[Path, int]]) -> None:
        """扫描目录查找 SKILL.md 文件，缓存元数据。

        相同 name 按搜索路径优先级 first-wins。重复时记录警告。
        跳过 malformed frontmatter 或缺少必填字段的技能。
        """
        all_dirs = sorted(search_dirs, key=lambda x: x[1])
        for search_dir, _priority in all_dirs:
            if not search_dir.is_dir():
                continue
            try:
                for file_path in sorted(search_dir.rglob("SKILL.md")):
                    try:
                        text = file_path.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        logger.warning("Failed to read %s; skipping", file_path)
                        continue
                    frontmatter = _parse_frontmatter(text)
                    if frontmatter is None:
                        logger.warning(
                            "Malformed frontmatter in %s; skipping", file_path
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
                    body_start = _find_body_start(text)

                    refs, ref_map = _scan_skill_references(file_path.parent)

                    self._skills[name] = SkillDef(
                        name=name,
                        description=str(frontmatter["description"]),
                        hidden=bool(frontmatter.get("hidden", False)),
                        file_path=file_path,
                        frontmatter=frontmatter,
                        content=body_start,
                        references=tuple(refs),
                    )
                    if ref_map:
                        self._ref_paths[name] = ref_map
            except Exception:
                logger.exception("SkillRegistry: failed to scan %s", search_dir)
                continue

    def list_summaries(self) -> list[SkillSummary]:
        """返回所有非隐藏技能的摘要列表。"""
        return [
            skill.to_summary() for skill in self._skills.values() if not skill.hidden
        ]

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
                body_start = _find_body_start(text)
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

        content, _ = _load_reference_text(ref_path)
        return content


def _find_body_start(text: str) -> str | None:
    """找到 frontmatter 结束后的正文内容。

    返回第二个 --- 分隔符之后的内容，去除前导空行。
    无正文时返回 None。
    """
    lines = text.splitlines()
    delim_count = 0
    body_lines: list[str] = []
    for line in lines:
        if line.strip() == "---":
            delim_count += 1
            continue
        if delim_count >= 2:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return body if body else None


# ── XML 转义辅助 ──


def _xml_escape_content(text: str) -> str:
    """对 XML 元素内容进行转义。"""
    return _xml.escape(text)


def _xml_escape_attr(text: str) -> str:
    """对 XML 属性值进行转义（含引号）。返回带引号的字符串。"""
    return _xml.quoteattr(text)


# ── load_skill 工具 ──


def build_load_skill_tool(
    registry: SkillRegistry,
) -> ToolSpec:
    """构建 load_skill 工具。

    支持 optional reference 参数加载指定引用文件。
    默认返回技能正文 + references 摘要元数据。
    """

    def handler(input: ToolInput) -> str:
        name = input.get("name", "")
        if not isinstance(name, str) or not name.strip():
            return "Error: 'name' is required and must be a non-empty string."

        skill = registry.load(name)
        if skill is None:
            return f"Unknown skill: {name}"

        ref_name = input.get("reference")
        if ref_name is not None:
            if not isinstance(ref_name, str) or not ref_name.strip():
                return "Error: 'reference' must be a non-empty string."
            ref_content = registry.load_reference(name, ref_name)
            if ref_content is None:
                return f"Unknown reference: {ref_name} in skill {name}"
            safe_name = _xml_escape_attr(name)
            safe_ref = _xml_escape_attr(ref_name)
            safe_content = _xml_escape_content(ref_content)
            return (
                f"<skill name={safe_name} reference={safe_ref}>\n"
                f"{safe_content}\n"
                f"</skill>"
            )

        ref_xml_parts: list[str] = []
        if skill.references:
            ref_xml_parts.append("")
            ref_xml_parts.append("<references>")
            for ref in skill.references:
                attrs = f'name={_xml_escape_attr(ref.name)} size="{ref.size}"'
                if ref.truncated:
                    attrs += ' truncated="true"'
                if ref.skipped:
                    safe_reason = _xml_escape_attr(ref.skipped_reason or "unknown")
                    attrs += f' skipped="true" reason={safe_reason}'
                ref_xml_parts.append(f"  <reference {attrs}/>")
            ref_xml_parts.append("</references>")
        ref_suffix = "\n".join(ref_xml_parts)

        safe_name = _xml_escape_attr(name)
        if skill.content:
            return f"<skill name={safe_name}>\n{skill.content}{ref_suffix}\n</skill>"
        return f"<skill name={safe_name}>\n{skill.description}{ref_suffix}\n</skill>"

    return ToolSpec(
        name="load_skill",
        description=(
            "Load a skill by name. Returns the full skill content "
            "with a references summary. Use the optional 'reference' "
            "parameter to load a specific reference file."
        ),
        input_hint='JSON: {"name": "code-review"}',
        handler=handler,
        group="skills",
        read_only=True,
        schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to load",
                },
                "reference": {
                    "type": "string",
                    "description": "Optional reference file to load from this skill",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    )


# ── SkillIndexCollector ──


class SkillIndexCollector:
    """技能摘要收集器。

    调用 SkillRegistry.list_summaries() 获取摘要信息，
    以 <available-skills> 块的形式注入 USER_CONTEXT。
    不加载技能正文。
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def collect(self, input: object) -> list[ContextBlock]:
        summaries = self._registry.list_summaries()
        if not summaries:
            return []
        lines: list[str] = ["<available-skills>"]
        for s in summaries:
            lines.append(f'  <skill name="{s.name}">{s.description}</skill>')
        lines.append("</available-skills>")
        body = "\n".join(lines)
        return [
            ContextBlock(
                source=ContextBlockSource.SKILL,
                target=ContextBlockTarget.USER_CONTEXT,
                priority=ContextPriority.MEDIUM,
                content=body,
            )
        ]
