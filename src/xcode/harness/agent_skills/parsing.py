"""YAML frontmatter 解析与校验。

提供从 SKILL.md 文本中提取和规范化 YAML frontmatter 的纯函数。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SKILL_NAME_MAX_CHARS = 64
_SKILL_DESCRIPTION_MAX_CHARS = 1024
_SKILL_COMPATIBILITY_MAX_CHARS = 500
_COLON_REPAIR_FIELDS = frozenset(
    {"description", "license", "compatibility", "allowed-tools"}
)


def parse_frontmatter(
    text: str,
    *,
    skill_directory_name: str | None = None,
    source: Path | None = None,
) -> dict[str, Any] | None:
    """解析 SKILL.md 的 YAML frontmatter。

    从 --- 分隔符之间提取 YAML，并按规范策略校验字段。
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


def find_body_start(text: str) -> str | None:
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

    raw_disable = parsed.get("disable-model-invocation", False)
    if not isinstance(raw_disable, bool):
        logger.warning(
            "Skill frontmatter 'disable-model-invocation' must be a boolean in %s; "
            "using false",
            source_label,
        )
        raw_disable = False

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
    if raw_disable:
        logger.info(
            "Skill %r in %s has model invocation disabled",
            name,
            source_label,
        )

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
        "disable_model_invocation": raw_disable,
        "license": license_value,
        "compatibility": compatibility,
        "metadata": metadata,
        "allowed-tools": allowed_tools,
    }


def frontmatter_optional_string(
    frontmatter: Mapping[str, object],
    field_name: str,
) -> str | None:
    """读取已规范化 frontmatter 中的可选字符串。"""
    value = frontmatter.get(field_name)
    return value if isinstance(value, str) else None


def frontmatter_metadata(
    frontmatter: Mapping[str, object],
) -> dict[str, str]:
    """读取已规范化 frontmatter metadata 的独立副本。"""
    value = frontmatter.get("metadata")
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


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
