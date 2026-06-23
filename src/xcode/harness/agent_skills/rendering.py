"""XML 格式化与 SkillIndexCollector。

提供技能摘要的 pi 风格 XML 化，以及向 agent 上下文注入技能目录的收集器。
"""

from __future__ import annotations

import xml.sax.saxutils as _xml
from typing import TYPE_CHECKING

from xcode.agent.context_assembly import (
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextPriority,
)
from xcode.harness.agent_skills.models import SkillSummary

if TYPE_CHECKING:
    from xcode.harness.agent_skills.registry import SkillRegistry

_CATALOG_DESCRIPTION_MAX_CHARS = 768


def format_skill_summary_xml(summary: SkillSummary) -> str:
    """将单个技能摘要格式化为 <skill> 块（pi 风格）。"""
    name = summary.name
    desc = summary.description
    if len(desc) > _CATALOG_DESCRIPTION_MAX_CHARS:
        desc = f"{desc[: _CATALOG_DESCRIPTION_MAX_CHARS - 3]}..."
    return (
        f"<skill>\n"
        f"  <name>{xml_escape_content(name)}</name>\n"
        f"  <description>{xml_escape_content(desc)}</description>\n"
        f"</skill>"
    )


def format_skill_list_xml(summaries: list[SkillSummary]) -> str:
    """格式化 <available-skills> 块（pi 风格）。"""
    if not summaries:
        return ""
    parts: list[str] = [
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
        parts.append(f"  {format_skill_summary_xml(s)}")
    parts.append("</available-skills>")
    return "\n".join(parts)


def xml_escape_content(text: str) -> str:
    """对 XML 元素内容进行转义。"""
    return _xml.escape(_xml_safe_text(text))


def xml_escape_attr(text: str) -> str:
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
        body = format_skill_list_xml(summaries)
        return [
            ContextBlock(
                source=ContextBlockSource.SKILL,
                target=ContextBlockTarget.USER_CONTEXT,
                priority=ContextPriority.MEDIUM,
                content=body,
            )
        ]
