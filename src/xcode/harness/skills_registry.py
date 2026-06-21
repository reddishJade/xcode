"""技能注册表——技能发现、索引、懒加载的唯一后端。

提供 SkillRegistry 作为技能管理单一后端，SkillIndexCollector 通过它
注入摘要信息，load_skill 工具通过它执行懒加载。
"""

from __future__ import annotations

import json
import logging
import os as _os
import re
import xml.sax.saxutils as _xml
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xcode.agent.context_assembly import (
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextPriority,
)
from xcode.harness.skills import ToolInput, ToolSpec
from xcode.harness.skill_activation import (
    activated_skill_names,
    SKILL_ACTIVATION_STATE_TAG,
)

logger = logging.getLogger(__name__)

# ── 搜索路径优先级 ──
# 0: 显式 skills_dir                 调用方配置目录（最高）
# 1: <project>/.xcode/skills/      项目级技能
# 2: <project>/.agents/skills/     项目级技能
# 3: ~/.xcode/skills/              用户级技能
# 4: ~/.agents/skills/             用户级技能


def build_skill_search_dirs(
    project_root: Path | None,
    *,
    trust_project_skills: bool = True,
    skills_dir: Path | None = None,
) -> list[tuple[Path, int]]:
    """构建优先级排序的技能搜索目录列表。

    返回 [(Path, priority), ...]，priority 越小优先级越高。
    项目级技能仅在调用方明确确认工作区可信时加入。
    显式 skills_dir 表示调用方已信任该目录，始终具有最高优先级。
    """
    dirs: list[tuple[Path, int]] = []
    if skills_dir is not None:
        explicit_dir = skills_dir.resolve()
        if not explicit_dir.is_dir():
            logger.warning(
                "Configured skill directory does not exist: %s", explicit_dir
            )
        dirs.append((explicit_dir, 0))
    if project_root is not None and trust_project_skills:
        dirs.append(((project_root / ".xcode" / "skills").resolve(), 1))
        dirs.append(((project_root / ".agents" / "skills").resolve(), 2))
    home = Path.home()
    dirs.append(((home / ".xcode" / "skills").resolve(), 3))
    dirs.append(((home / ".agents" / "skills").resolve(), 4))

    unique_dirs: list[tuple[Path, int]] = []
    seen_paths: set[Path] = set()
    for path, priority in dirs:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_dirs.append((path, priority))
    return unique_dirs


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
class SkillResource:
    """技能脚本或资产的相对路径元数据。"""

    path: str
    size: int = 0
    skipped: bool = False
    skipped_reason: str | None = None


@dataclass(frozen=True)
class SkillDef:
    """技能的完整定义，包含懒加载的正文内容和引用元数据。"""

    name: str
    description: str
    file_path: Path
    hidden: bool = False
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None
    frontmatter: dict[str, object] = field(default_factory=dict)
    content: str | None = None
    references: tuple[SkillReference, ...] = ()
    scripts: tuple[SkillResource, ...] = ()
    assets: tuple[SkillResource, ...] = ()

    def to_summary(self) -> SkillSummary:
        return SkillSummary(
            name=self.name,
            description=self.description,
            hidden=self.hidden,
        )


# ── Frontmatter 解析（YAML） ──
# 使用 yaml.safe_load 解析 SKILL.md 的 YAML frontmatter。
# name 格式问题按兼容性策略记录警告但继续加载；description 缺失或
# frontmatter 无法解析时跳过。可选规范字段会保留到 SkillDef。

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SKILL_NAME_MAX_CHARS = 64
_SKILL_DESCRIPTION_MAX_CHARS = 1024
_SKILL_COMPATIBILITY_MAX_CHARS = 500
_COLON_REPAIR_FIELDS = frozenset(
    {"description", "license", "compatibility", "allowed-tools"}
)


def _parse_frontmatter(
    text: str,
    *,
    skill_directory_name: str | None = None,
    source: Path | None = None,
) -> dict[str, Any] | None:
    """解析 SKILL.md 的 YAML frontmatter。

    从 --- 分隔符之间提取 YAML，并按客户端兼容策略规范化字段。
    description 缺失或 YAML 无法解析时返回 None。
    """
    yaml_text = _frontmatter_yaml(text)
    if yaml_text is None:
        logger.warning(
            "Missing or unclosed YAML frontmatter in %s; skipping",
            source or "skill",
        )
        return None
    parsed = _load_frontmatter_yaml(yaml_text, source)
    if parsed is None:
        return None
    return _normalize_frontmatter(
        parsed,
        skill_directory_name=skill_directory_name,
        source=source,
    )


def _frontmatter_yaml(text: str) -> str | None:
    """提取 frontmatter YAML 文本。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "\n".join(lines[1:index])
    return None


def _load_frontmatter_yaml(
    yaml_text: str,
    source: Path | None,
) -> dict[str, object] | None:
    """解析 YAML，并对常见未引用冒号值执行一次窄范围修复。"""
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        repaired = _repair_unquoted_colon_values(yaml_text)
        if repaired == yaml_text:
            logger.warning(
                "Invalid YAML frontmatter in %s; skipping", source or "skill"
            )
            return None
        try:
            parsed = yaml.safe_load(repaired)
        except yaml.YAMLError:
            logger.warning(
                "Invalid YAML frontmatter in %s; skipping", source or "skill"
            )
            return None
        logger.warning(
            "Recovered invalid YAML frontmatter with quoted scalar values in %s",
            source or "skill",
        )

    if not isinstance(parsed, dict):
        logger.warning("Frontmatter is not a dict in %s; skipping", source or "skill")
        return None
    return {str(key): value for key, value in parsed.items()}


def _repair_unquoted_colon_values(yaml_text: str) -> str:
    """修复规范字符串字段中未引用的 `: `，其余 YAML 保持不变。"""
    repaired_lines: list[str] = []
    for line in yaml_text.splitlines():
        prefix, separator, value = line.partition(":")
        field_name = prefix.strip()
        stripped_value = value.strip()
        if (
            separator
            and field_name in _COLON_REPAIR_FIELDS
            and ": " in stripped_value
            and not stripped_value.startswith(("'", '"', "|", ">"))
        ):
            indentation = prefix[: len(prefix) - len(prefix.lstrip())]
            repaired_lines.append(
                f"{indentation}{field_name}: "
                f"{json.dumps(stripped_value, ensure_ascii=False)}"
            )
            continue
        repaired_lines.append(line)
    return "\n".join(repaired_lines)


def _normalize_frontmatter(
    parsed: Mapping[str, object],
    *,
    skill_directory_name: str | None,
    source: Path | None,
) -> dict[str, Any] | None:
    """规范化规范字段，并对 cosmetic 问题记录警告。"""
    source_label = str(source or "skill")
    name = parsed.get("name")
    description = parsed.get("description")
    hidden = parsed.get("hidden", False)

    if not isinstance(name, str) or not name.strip():
        if not skill_directory_name:
            logger.warning(
                "Skill frontmatter 'name' is missing in %s and no directory fallback "
                "is available; skipping",
                source_label,
            )
            return None
        name = skill_directory_name
        logger.warning(
            "Skill frontmatter 'name' is missing in %s; using directory name %r",
            source_label,
            name,
        )
    else:
        name = name.strip()
    if not isinstance(description, str) or not description.strip():
        logger.warning(
            "Skill frontmatter 'description' must be a non-empty string in %s; "
            "skipping",
            source_label,
        )
        return None
    description = description.strip()
    _warn_skill_name_issues(name, skill_directory_name, source_label)
    if len(description) > _SKILL_DESCRIPTION_MAX_CHARS:
        logger.warning(
            "Skill description exceeds %d characters in %s; loading anyway",
            _SKILL_DESCRIPTION_MAX_CHARS,
            source_label,
        )
    if not isinstance(hidden, bool):
        logger.warning(
            "Skill frontmatter 'hidden' must be a boolean in %s; using false",
            source_label,
        )
        hidden = False

    license_value = _optional_string(parsed, "license", source_label)
    compatibility = _optional_string(parsed, "compatibility", source_label)
    if (
        compatibility is not None
        and len(compatibility) > _SKILL_COMPATIBILITY_MAX_CHARS
    ):
        logger.warning(
            "Skill compatibility exceeds %d characters in %s; loading anyway",
            _SKILL_COMPATIBILITY_MAX_CHARS,
            source_label,
        )
    allowed_tools = _optional_string(parsed, "allowed-tools", source_label)
    metadata = _metadata_strings(parsed.get("metadata"), source_label)

    return {
        "name": name,
        "description": description,
        "hidden": hidden,
        "license": license_value,
        "compatibility": compatibility,
        "metadata": metadata,
        "allowed-tools": allowed_tools,
    }


def _warn_skill_name_issues(
    name: str,
    skill_directory_name: str | None,
    source_label: str,
) -> None:
    """记录名称规范问题，但不阻止加载。"""
    if len(name) > _SKILL_NAME_MAX_CHARS:
        logger.warning(
            "Skill name %r exceeds %d characters in %s; loading anyway",
            name,
            _SKILL_NAME_MAX_CHARS,
            source_label,
        )
    if not _SKILL_NAME_PATTERN.fullmatch(name):
        logger.warning(
            "Skill name %r violates lowercase alphanumeric-hyphen rules in %s; "
            "loading anyway",
            name,
            source_label,
        )
    if skill_directory_name and name != skill_directory_name:
        logger.warning(
            "Skill name %r does not match directory %r in %s; loading anyway",
            name,
            skill_directory_name,
            source_label,
        )


def _optional_string(
    parsed: Mapping[str, object],
    field_name: str,
    source_label: str,
) -> str | None:
    """保留可选字符串字段；无效值记录警告后忽略。"""
    value = parsed.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        logger.warning(
            "Skill frontmatter %r must be a non-empty string in %s; ignoring",
            field_name,
            source_label,
        )
        return None
    return value.strip()


def _metadata_strings(value: object, source_label: str) -> dict[str, str]:
    """将 metadata 规范为字符串键值映射。"""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        logger.warning(
            "Skill frontmatter 'metadata' must be a mapping in %s; ignoring",
            source_label,
        )
        return {}
    metadata: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            logger.warning(
                "Skill metadata keys and values must be strings in %s; coercing entry",
                source_label,
            )
        metadata[str(key)] = str(item)
    return metadata


def _frontmatter_optional_string(
    frontmatter: Mapping[str, object],
    field_name: str,
) -> str | None:
    """读取已规范化 frontmatter 中的可选字符串。"""
    value = frontmatter.get(field_name)
    return value if isinstance(value, str) else None


def _frontmatter_metadata(
    frontmatter: Mapping[str, object],
) -> dict[str, str]:
    """读取已规范化 frontmatter metadata 的独立副本。"""
    value = frontmatter.get("metadata")
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


# ── 引用文件扫描与加载 ──

_REFERENCE_MAX_BYTES = 50 * 1024  # 单个引用文件大小上限
_CATALOG_DESCRIPTION_MAX_CHARS = 768


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


def _scan_skill_resources(
    skill_root: Path,
    directory_name: str,
) -> tuple[SkillResource, ...]:
    """扫描 scripts/ 或 assets/，仅返回相对路径元数据。"""
    resource_dir = skill_root / directory_name
    if not resource_dir.is_dir():
        return ()

    resources: list[SkillResource] = []
    for file_path in _collect_reference_files(resource_dir):
        relative_path = file_path.relative_to(skill_root).as_posix()
        if any(part.startswith(".") for part in relative_path.split("/")):
            resources.append(
                SkillResource(
                    path=relative_path,
                    skipped=True,
                    skipped_reason="hidden",
                )
            )
            continue
        if file_path.is_symlink():
            resources.append(
                SkillResource(
                    path=relative_path,
                    skipped=True,
                    skipped_reason="symlink",
                )
            )
            continue
        try:
            size = file_path.stat().st_size
        except OSError:
            resources.append(
                SkillResource(
                    path=relative_path,
                    skipped=True,
                    skipped_reason="unreadable",
                )
            )
            continue
        resources.append(SkillResource(path=relative_path, size=size))
    return tuple(sorted(resources, key=lambda resource: resource.path))


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
        self._activated: set[str] = set()

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
                    frontmatter = _parse_frontmatter(
                        text,
                        skill_directory_name=file_path.parent.name,
                        source=file_path,
                    )
                    if frontmatter is None:
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
                    scripts = _scan_skill_resources(file_path.parent, "scripts")
                    assets = _scan_skill_resources(file_path.parent, "assets")

                    self._skills[name] = SkillDef(
                        name=name,
                        description=str(frontmatter["description"]),
                        hidden=bool(frontmatter.get("hidden", False)),
                        license=_frontmatter_optional_string(
                            frontmatter,
                            "license",
                        ),
                        compatibility=_frontmatter_optional_string(
                            frontmatter,
                            "compatibility",
                        ),
                        metadata=_frontmatter_metadata(frontmatter),
                        allowed_tools=_frontmatter_optional_string(
                            frontmatter,
                            "allowed-tools",
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

    def list_summaries(self) -> list[SkillSummary]:
        """返回所有非隐藏技能的摘要列表。"""
        return [
            skill.to_summary() for skill in self._skills.values() if not skill.hidden
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
        return skill is not None and not skill.hidden

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
    return _xml.escape(_xml_safe_text(text))


def _xml_escape_attr(text: str) -> str:
    """对 XML 属性值进行转义（含引号）。返回带引号的字符串。"""
    return _xml.quoteattr(_xml_safe_text(text))


def _xml_safe_text(text: str) -> str:
    """替换 XML 1.0 不允许的控制字符。"""
    return "".join(
        character
        if character in "\t\n\r"
        or "\u0020" <= character <= "\ud7ff"
        or "\ue000" <= character <= "\ufffd"
        else "\ufffd"
        for character in text
    )


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

        activated_skill, already_activated = registry.activate(name)
        assert activated_skill is not None
        skill = activated_skill
        safe_name = _xml_escape_attr(name)
        safe_root = _xml_escape_attr(str(skill.file_path.parent.resolve()))
        if already_activated:
            return (
                f"<skill-activation name={safe_name} root={safe_root} "
                'status="already-active"/>'
            )

        ref_xml_parts: list[str] = []
        if skill.references:
            ref_xml_parts.append("")
            ref_xml_parts.append("<references>")
            for ref in skill.references:
                relative_path = f"references/{ref.name}"
                attrs = (
                    f"name={_xml_escape_attr(ref.name)} "
                    f'path={_xml_escape_attr(relative_path)} size="{ref.size}"'
                )
                if ref.truncated:
                    attrs += ' truncated="true"'
                if ref.skipped:
                    safe_reason = _xml_escape_attr(ref.skipped_reason or "unknown")
                    attrs += f' skipped="true" reason={safe_reason}'
                ref_xml_parts.append(f"  <reference {attrs}/>")
            ref_xml_parts.append("</references>")
        resource_xml_parts = [
            *_render_resource_group("scripts", skill.scripts),
            *_render_resource_group("assets", skill.assets),
        ]
        resource_suffix = "\n".join([*ref_xml_parts, *resource_xml_parts])

        activation_state = json.dumps({"name": name}, ensure_ascii=False)
        state_line = (
            f"<{SKILL_ACTIVATION_STATE_TAG}>{activation_state}"
            f"</{SKILL_ACTIVATION_STATE_TAG}>"
        )
        activation_context = _render_activation_frontmatter(skill)
        activation_lines = "\n".join(
            line for line in (state_line, activation_context) if line
        )
        if skill.content:
            return (
                f'<skill name={safe_name} root={safe_root} activated="true">\n'
                f"{activation_lines}\n{skill.content}{resource_suffix}\n</skill>"
            )
        return (
            f'<skill name={safe_name} root={safe_root} activated="true">\n'
            f"{activation_lines}\n{skill.description}{resource_suffix}\n</skill>"
        )

    available_names = registry.available_names()
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
                    "enum": list(available_names),
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


def _render_resource_group(
    group_name: str,
    resources: tuple[SkillResource, ...],
) -> list[str]:
    """渲染 scripts/assets 相对路径元数据。"""
    if not resources:
        return []
    lines = ["", f"<{group_name}>"]
    for resource in resources:
        attrs = f'path={_xml_escape_attr(resource.path)} size="{resource.size}"'
        if resource.skipped:
            reason = _xml_escape_attr(resource.skipped_reason or "unknown")
            attrs += f' skipped="true" reason={reason}'
        lines.append(f"  <resource {attrs}/>")
    lines.append(f"</{group_name}>")
    return lines


def _render_activation_frontmatter(skill: SkillDef) -> str:
    """向模型披露兼容性和 advisory allowed-tools 信息。"""
    lines: list[str] = []
    if skill.compatibility:
        lines.append(
            f"<compatibility>{_xml_escape_content(skill.compatibility)}</compatibility>"
        )
    if skill.allowed_tools:
        lines.append(
            '<allowed-tools advisory="true" permission-bypass="false">'
            f"{_xml_escape_content(skill.allowed_tools)}</allowed-tools>"
        )
    return "\n".join(lines)


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
        lines: list[str] = [
            "<skill-activation>",
            (
                "When the user task clearly matches a skill description below, "
                "call load_skill with that exact name before performing the task. "
                "Do not load a skill when no description clearly matches."
            ),
            "</skill-activation>",
            "<available-skills>",
        ]
        for s in summaries:
            safe_name = _xml_escape_attr(s.name)
            description = s.description
            if len(description) > _CATALOG_DESCRIPTION_MAX_CHARS:
                description = f"{description[: _CATALOG_DESCRIPTION_MAX_CHARS - 3]}..."
            safe_description = _xml_escape_content(description)
            lines.append(f"  <skill name={safe_name}>{safe_description}</skill>")
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
