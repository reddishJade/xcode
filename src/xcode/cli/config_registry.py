"""交互式配置注册表：声明 `/config` 可浏览和修改的设置项。

每个 `SettingSpec` 描述一行设置：点路径、展示名、值类型、说明与格式化/解析
规则。REPL、TUI 和 CLI 共用同一份注册表，写入统一经过
`XcodeRuntimeConfig` 校验后再落盘。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from enum import StrEnum
from operator import attrgetter
from pathlib import Path
from typing import Any

import questionary
from pydantic import ValidationError

from xcode.harness.config import XcodeRuntimeConfig, load_runtime_config

from .setup_wizard import _load_existing_config, _save_config


class SettingKind(StrEnum):
    """设置项的编辑形态。"""

    BOOL = "bool"
    ENUM = "enum"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    PATH = "path"
    STR_LIST = "str_list"
    INFO = "info"


@dataclass(frozen=True)
class SettingSpec:
    """单行设置的声明。"""

    key: str
    label: str
    kind: SettingKind
    description: str = ""
    choices: tuple[str, ...] = ()
    nullable: bool = False
    none_choice: str | None = None
    getter: Callable[[XcodeRuntimeConfig], Any] | None = None
    formatter: Callable[[Any], str] | None = None
    choice_descriptions: dict[str, str] = dataclasses_field(default_factory=dict)
    writer: Callable[[dict[str, Any], Any], None] | None = None

    def read(self, config: XcodeRuntimeConfig) -> Any:
        """从生效配置中读取当前值；INFO/复合行必须提供自定义 getter。"""
        if self.getter is not None:
            return self.getter(config)
        return attrgetter(self.key)(config)

    def describe_choice(self, token: str) -> str:
        """返回枚举选项的灰色说明文本；未声明时回退到整体说明。"""
        return self.choice_descriptions.get(token) or self.description


_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_CLEAR_TOKENS = frozenset({"none", "unset", "default", "unlimited"})


def _read_approval_choice(config: XcodeRuntimeConfig) -> str:
    """把 approval_policy + approval_router 组合读回三选项 token。"""
    if config.security.approval_policy == "never":
        return "always proceeds"
    router = config.security.approval_router
    if router == "auto":
        return "agent decides"
    if router == "user":
        return "asks for review"
    return (
        "agent decides"
        if config.execution_modes.default_mode == "build"
        else ("asks for review")
    )


def _write_approval_choice(raw: dict[str, Any], value: Any) -> None:
    """把三选项 token 落到 approval_policy + approval_router 两个字段。"""
    security = raw.setdefault("security", {})
    if not isinstance(security, dict):
        security = {}
        raw["security"] = security
    if value == "always proceeds":
        security["approval_policy"] = "never"
        security.pop("approval_router", None)
    elif value == "agent decides":
        security["approval_policy"] = "on-request"
        security["approval_router"] = "auto"
    else:
        security["approval_policy"] = "on-request"
        security["approval_router"] = "user"


SETTING_SPECS: tuple[SettingSpec, ...] = (
    SettingSpec(
        key="execution_modes.default_mode",
        label="Default Mode",
        kind=SettingKind.ENUM,
        choices=("act", "build", "plan"),
        description=(
            "Behavior for writes and shell commands in new sessions. "
            "'act': read allowed, writes and shell ask the user. "
            "'build': workspace mutations auto-run, boundary actions "
            "auto-reviewed. 'plan': research and plan without making changes."
        ),
        choice_descriptions={
            "act": "Maximizes interactivity: every write or shell command asks.",
            "build": (
                "Maximizes autonomy inside the workspace; risky actions are "
                "reviewed by a reviewer model."
            ),
            "plan": "Read-only research mode; nothing is written to disk.",
        },
    ),
    SettingSpec(
        key="security.approval_policy",
        label="Approval Policy",
        kind=SettingKind.ENUM,
        choices=("always proceeds", "agent decides", "asks for review"),
        description=(
            "What happens when the agent wants to edit files or run commands "
            "beyond the current mode's free tier. 'always proceeds': never "
            "asks (maximizes autonomy, risk of unsafe actions). "
            "'agent decides': a reviewer model reviews based on complexity. "
            "'asks for review': always asks for your confirmation."
        ),
        choice_descriptions={
            "always proceeds": (
                "Agent never asks; boundary actions run without review."
            ),
            "agent decides": (
                "Reviewer model approves low-risk actions and escalates risky ones."
            ),
            "asks for review": (
                "Every boundary action waits for explicit user approval."
            ),
        },
        getter=_read_approval_choice,
        writer=_write_approval_choice,
    ),
    SettingSpec(
        key="security.non_workspace_access",
        label="Non-Workspace Access",
        kind=SettingKind.BOOL,
        description=(
            "Whether directories outside the workspace may be accessed at "
            "all. When off, the external directory whitelist is ignored."
        ),
        choice_descriptions={
            "on": "External paths follow the security.external_directories whitelist.",
            "off": "Everything outside the workspace is denied.",
        },
    ),
    SettingSpec(
        key="tools.shell",
        label="Shell",
        kind=SettingKind.ENUM,
        choices=("auto", "bash", "zsh", "sh", "fish", "pwsh", "powershell", "cmd"),
        description=(
            "Shell used by the bash tool. 'auto' detects the login shell at "
            "runtime; pick an explicit value to pin behavior across machines."
        ),
    ),
)

_BY_KEY: dict[str, SettingSpec] = {spec.key: spec for spec in SETTING_SPECS}


def find_setting(query: str) -> SettingSpec | None:
    """按标签或点路径查找设置项；无歧义时返回，否则 None。"""
    needle = query.strip().lower()
    if not needle:
        return None
    exact = _BY_KEY.get(needle)
    if exact is not None:
        return exact
    matches = [
        spec
        for spec in SETTING_SPECS
        if needle in spec.label.lower() or needle in spec.key
    ]
    return matches[0] if len(matches) == 1 else None


def matching_settings(query: str) -> list[SettingSpec]:
    """返回所有标签或路径包含查询词的设置项。"""
    needle = query.strip().lower()
    if not needle:
        return []
    return [
        spec
        for spec in SETTING_SPECS
        if needle in spec.label.lower() or needle in spec.key
    ]


def setting_detail(spec: SettingSpec, config: XcodeRuntimeConfig) -> list[str]:
    """INFO 行展开的多行详情；无自定义详情时回退到值的字符串形式。"""
    return [f"  {spec.read(config)}"]


def format_setting(spec: SettingSpec, config: XcodeRuntimeConfig) -> str:
    """渲染当前生效值的展示文本。"""
    value = spec.read(config)
    if spec.formatter is not None:
        return spec.formatter(value)
    if value is None:
        return "(unset)"
    if spec.kind is SettingKind.BOOL:
        return "on" if value else "off"
    if isinstance(value, float | int):
        return f"{float(value):g}"
    if isinstance(value, tuple | list):
        items = list(value)
        return ", ".join(items) if items else "(none)"
    return str(value)


def parse_setting(spec: SettingSpec, text: str) -> Any:
    """把用户输入解析为目标 Python 值；非法输入抛 ValueError。"""
    stripped = text.strip()
    lowered = stripped.lower()
    clearable = spec.nullable or spec.none_choice is not None
    if lowered in _CLEAR_TOKENS and clearable:
        return None
    if spec.kind is SettingKind.BOOL:
        if lowered in _TRUE_TOKENS:
            return True
        if lowered in _FALSE_TOKENS:
            return False
        raise ValueError(f"Expected on/off, got: {text!r}")
    if spec.kind is SettingKind.ENUM:
        if lowered not in spec.choices:
            expected = "/".join(spec.choices)
            raise ValueError(f"Expected {expected}, got: {text!r}")
        return None if lowered == spec.none_choice else lowered
    if spec.kind is SettingKind.INT:
        return int(stripped)
    if spec.kind is SettingKind.FLOAT:
        return float(stripped)
    if spec.kind is SettingKind.PATH:
        if not stripped:
            raise ValueError("Path cannot be empty; use 'none' to clear.")
        return stripped
    if spec.kind is SettingKind.STR_LIST:
        parts = [part.strip() for part in stripped.split(",")]
        items = tuple(part for part in parts if part)
        return items or None
    if spec.kind is SettingKind.STR:
        if not stripped:
            raise ValueError("Value cannot be empty.")
        return stripped
    raise ValueError(f"Setting '{spec.key}' does not accept text input.")


def apply_setting(raw: dict[str, Any], spec: SettingSpec, value: Any) -> None:
    """把解析后的值写进原始 dict；None 表示清除该键以恢复默认。

    复合行（writer 非空）由 writer 决定写入哪些字段。
    """
    if spec.writer is not None:
        spec.writer(raw, value)
        return
    parts = spec.key.split(".")
    node = raw
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    leaf = parts[-1]
    restores_default = value is None or (
        spec.kind is SettingKind.STR_LIST and not value
    )
    if restores_default:
        node.pop(leaf, None)
    else:
        node[leaf] = value


def _first_validation_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    path = ".".join(str(part) for part in error["loc"])
    return f"{path}: {error['msg']}" if path else error["msg"]


def commit_setting_value(
    config_path: Path, spec: SettingSpec, value: Any
) -> tuple[bool, str]:
    """写入并校验；校验失败时不落盘。返回 (是否成功, 反馈消息)。"""
    try:
        raw = _load_existing_config(config_path)
    except (json.JSONDecodeError, ValueError) as exc:
        return False, f"Cannot read {config_path.name}: {exc}"

    apply_setting(raw, spec, value)
    try:
        XcodeRuntimeConfig.model_validate(raw)
    except ValidationError as exc:
        reason = _first_validation_error(exc)
        return False, f"Rejected {spec.label}: {reason}"

    _save_config(raw, config_path)
    display = format_setting(spec, XcodeRuntimeConfig.model_validate(raw))
    return True, f"{spec.label} set to {display} (saved to {config_path.name})"


def save_setting_text(
    config_path: Path, spec: SettingSpec, text: str
) -> tuple[bool, str]:
    """解析用户输入后提交；返回 (是否成功, 反馈消息)。"""
    if spec.kind is SettingKind.INFO:
        return False, f"'{spec.label}' is read-only here; edit JSON directly."
    try:
        value = parse_setting(spec, text)
    except ValueError as exc:
        return False, f"Invalid value for {spec.label}: {exc}"
    return commit_setting_value(config_path, spec, value)


def load_effective_config(config_path: Path) -> XcodeRuntimeConfig:
    """加载用于展示的生效配置（文件内容 + 环境覆盖，缺失用默认值）。"""
    return load_runtime_config(config_path)


def _row_title(spec: SettingSpec, config: XcodeRuntimeConfig) -> str:
    return f"{spec.label:<28}{format_setting(spec, config)}"


def _select_option(spec: SettingSpec, current_display: str) -> str | None:
    """枚举/布尔项的选择菜单；返回去掉标记的原始文本。"""
    if spec.kind is SettingKind.BOOL:
        tokens = ("on", "off")
    else:
        tokens = tuple(spec.choices)
    titles = [
        f"{token} (current)" if token == current_display.lower() else token
        for token in tokens
    ]
    picked = questionary.select(f"{spec.label}:", choices=titles).ask()
    if picked is None:
        return None
    return picked.removesuffix(" (current)")


def edit_setting_interactive(
    config_path: Path, spec: SettingSpec, config: XcodeRuntimeConfig
) -> None:
    """单个设置项的编辑流程：菜单选值或文本输入。"""
    current = format_setting(spec, config)
    if spec.kind is SettingKind.INFO:
        print(f"{spec.label} ({current}):")
        if spec.description:
            print(f"  {spec.description}")
        for line in setting_detail(spec, config):
            print(line)
        return

    print(f"  {spec.key} = {current}")
    if spec.description:
        print(f"  {spec.description}")

    if spec.kind in (SettingKind.BOOL, SettingKind.ENUM):
        text = _select_option(spec, current)
    else:
        hint = "Type value, enter to save, esc to cancel"
        if spec.nullable:
            hint += "; 'none' clears"
        text = questionary.text(
            f"{spec.label} — {hint}:",
            default="",
        ).ask()
        if text is not None and not text.strip():
            return

    if text is None:
        return
    saved, message = save_setting_text(config_path, spec, text)
    prefix = "  " if saved else "  ! "
    print(f"{prefix}{message}")


def run_config_browser(config_path: Path) -> None:
    """交互式配置浏览器：回车进入编辑，esc 返回上级或退出。"""
    while True:
        config = load_effective_config(config_path)
        choices = [
            questionary.Choice(title=_row_title(spec, config), value=spec)
            for spec in SETTING_SPECS
        ]
        choices.append(questionary.Choice(title="Exit", value=None))
        selected = questionary.select(
            f"Config ({config_path.name}) — enter to change, esc to exit:",
            choices=choices,
        ).ask()
        if selected is None:
            return
        edit_setting_interactive(config_path, selected, config)


__all__ = [
    "SETTING_SPECS",
    "SettingKind",
    "SettingSpec",
    "apply_setting",
    "commit_setting_value",
    "edit_setting_interactive",
    "find_setting",
    "format_setting",
    "load_effective_config",
    "matching_settings",
    "parse_setting",
    "run_config_browser",
    "save_setting_text",
    "setting_detail",
]
