"""REPL 显式技能激活解析与会话记录。"""

from __future__ import annotations

import re

from xcode.harness.skill_activation import ExplicitSkillActivationResult
from xcode.harness.config import ExecutionMode
from xcode.harness.session import SessionStore

_SKILL_INVOCATION_PATTERN = re.compile(
    r"^\$([A-Za-z0-9][A-Za-z0-9._-]*)(?:\s+(.*))?$",
    re.DOTALL,
)


def parse_skill_invocation(text: str) -> tuple[str, str] | None:
    """解析行首 `$skill-name` 及其后续用户任务。"""
    match = _SKILL_INVOCATION_PATTERN.fullmatch(text.strip())
    if match is None:
        return None
    return match.group(1), (match.group(2) or "").strip()


def available_skill_names(app: object) -> tuple[str, ...]:
    """从应用运行时读取可显式激活的技能名称。"""
    agent = getattr(app, "agent", None)
    provider = getattr(agent, "available_skill_names", None)
    if not callable(provider):
        return ()
    names = provider()
    if not isinstance(names, tuple):
        return ()
    return tuple(name for name in names if isinstance(name, str))


def activate_skill(
    app: object,
    store: SessionStore,
    skill_name: str,
    mode: ExecutionMode | None = None,
) -> ExplicitSkillActivationResult:
    """调用运行时激活技能，并将 canonical 工具事件写入会话。"""
    agent = getattr(app, "agent", None)
    activate = getattr(agent, "activate_skill", None)
    if not callable(activate):
        return ExplicitSkillActivationResult(
            name=skill_name,
            status="disabled",
            message="Skills are disabled for this runtime.",
        )

    result = activate(skill_name, mode=mode)
    if not isinstance(result, ExplicitSkillActivationResult):
        return ExplicitSkillActivationResult(
            name=skill_name,
            status="error",
            message="Skill activation returned an invalid runtime result.",
        )
    if result.status != "activated" or result.tool_call_id is None:
        return result

    store.append(
        "event",
        {
            "type": "tool_use",
            "data": {
                "id": result.tool_call_id,
                "name": "load_skill",
                "input": {"name": result.name},
            },
        },
    )
    store.append(
        "event",
        {
            "type": "tool_result",
            "data": {
                "tool_use_id": result.tool_call_id,
                "content": result.content,
                "status": "ok",
            },
        },
    )
    return result
