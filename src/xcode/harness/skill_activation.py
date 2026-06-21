"""Skill 激活状态标记的共享解析辅助。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal


SKILL_ACTIVATION_STATE_TAG = "skill-activation-state"
_ACTIVATION_STATE_PATTERN = re.compile(
    rf"<{SKILL_ACTIVATION_STATE_TAG}>(.*?)</{SKILL_ACTIVATION_STATE_TAG}>"
)
type ExplicitSkillActivationStatus = Literal[
    "activated",
    "already_active",
    "unknown",
    "disabled",
    "blocked",
    "error",
]


@dataclass(frozen=True)
class ExplicitSkillActivationResult:
    """显式技能激活的稳定运行时结果。"""

    name: str
    status: ExplicitSkillActivationStatus
    message: str
    content: str = ""
    tool_call_id: str | None = None


def is_skill_activation_content(content: object) -> bool:
    """判断文本是否包含可恢复的技能激活状态。"""
    return f"<{SKILL_ACTIVATION_STATE_TAG}>" in str(content)


def activated_skill_names(content: object) -> tuple[str, ...]:
    """从文本中的状态标记提取技能名称。"""
    names: list[str] = []
    for match in _ACTIVATION_STATE_PATTERN.finditer(str(content)):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name") if isinstance(payload, dict) else None
        if isinstance(name, str):
            names.append(name)
    return tuple(names)
