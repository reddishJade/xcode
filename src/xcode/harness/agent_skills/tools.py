"""load_skill 工具构建。

将 SkillRegistry 包装为 ToolSpec，供 Agent 按名称加载技能。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from xcode.harness.skills import ToolInput, ToolSpec
from xcode.harness.skill_activation import SKILL_ACTIVATION_STATE_TAG
from xcode.harness.agent_skills.models import SkillDef, SkillResource
from xcode.harness.agent_skills.rendering import xml_escape_attr, xml_escape_content

if TYPE_CHECKING:
    from xcode.harness.agent_skills.registry import SkillRegistry


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
            safe_name = xml_escape_attr(name)
            safe_ref = xml_escape_attr(ref_name)
            safe_content = xml_escape_content(ref_content)
            return (
                f"<skill name={safe_name} reference={safe_ref}>\n"
                f"{safe_content}\n"
                f"</skill>"
            )

        activated_skill, already_activated = registry.activate(name)
        assert activated_skill is not None
        skill = activated_skill
        safe_name = xml_escape_attr(name)
        safe_root = xml_escape_attr(str(skill.file_path.parent.resolve()))
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
                    f"name={xml_escape_attr(ref.name)} "
                    f'path={xml_escape_attr(relative_path)} size="{ref.size}"'
                )
                if ref.truncated:
                    attrs += ' truncated="true"'
                if ref.skipped:
                    safe_reason = xml_escape_attr(ref.skipped_reason or "unknown")
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
        attrs = f'path={xml_escape_attr(resource.path)} size="{resource.size}"'
        if resource.skipped:
            reason = xml_escape_attr(resource.skipped_reason or "unknown")
            attrs += f' skipped="true" reason={reason}'
        lines.append(f"  <resource {attrs}/>")
    lines.append(f"</{group_name}>")
    return lines


def _render_activation_frontmatter(skill: SkillDef) -> str:
    """向模型披露兼容性和 advisory allowed-tools 信息。"""
    lines: list[str] = []
    if skill.compatibility:
        lines.append(
            f"<compatibility>{xml_escape_content(skill.compatibility)}</compatibility>"
        )
    if skill.allowed_tools:
        lines.append(
            '<allowed-tools advisory="true" permission-bypass="false">'
            f"{xml_escape_content(skill.allowed_tools)}</allowed-tools>"
        )
    return "\n".join(lines)
