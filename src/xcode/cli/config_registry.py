"""交互式配置注册表：声明 `/config` 可浏览和修改的设置项。

每个 `SettingSpec` 描述一行设置：点路径、展示名、值类型、说明与格式化/解析
规则。REPL、TUI 和 CLI 共用同一份注册表，写入统一经过
`XcodeRuntimeConfig` 校验后再落盘。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
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

    def read(self, config: XcodeRuntimeConfig) -> Any:
        """从生效配置中读取当前值；INFO 行必须提供自定义 getter。"""
        if self.getter is not None:
            return self.getter(config)
        return attrgetter(self.key)(config)


_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_CLEAR_TOKENS = frozenset({"none", "unset", "default", "unlimited"})


def _fmt_bool(value: Any) -> str:
    return "on" if value else "off"


def _fmt_float(value: Any) -> str:
    return f"{float(value):g}"


def _fmt_seconds(value: Any) -> str:
    return f"{float(value):g}s"


def _fmt_bytes(value: Any) -> str:
    return f"{int(value):,}B"


def _fmt_str_list(value: Any) -> str:
    items = list(value) if value is not None else []
    return ", ".join(items) if items else "(none)"


def _fmt_unlimited(value: Any) -> str:
    return "unlimited" if value is None else str(value)


def _fmt_default(value: Any) -> str:
    return "default" if value is None else str(value)


def _fmt_optional_path(fallback: str) -> Callable[[Any], str]:
    def _format(value: Any) -> str:
        return fallback if value is None else str(value)

    return _format


def _count_getter(section: str) -> Callable[[XcodeRuntimeConfig], int]:
    return lambda config: len(attrgetter(section)(config))


def _hooks_detail(config: XcodeRuntimeConfig) -> list[str]:
    lines: list[str] = []
    for index, entry in enumerate(config.hooks.entries):
        state = "enabled" if entry.enabled else "disabled"
        command = " ".join(entry.command)
        matcher = f" matcher={entry.matcher}" if entry.matcher else ""
        lines.append(
            f"  [{index}] {entry.event}: {command}{matcher} ({state}, "
            f"{entry.failure_policy})"
        )
    return lines or ["  (none)"]


def _external_dirs_detail(config: XcodeRuntimeConfig) -> list[str]:
    entries = config.security.external_directories
    if not entries:
        return ["  (none)"]
    return [f"  {item.path} ({item.access})" for item in entries]


def _sensitive_overrides_detail(config: XcodeRuntimeConfig) -> list[str]:
    entries = config.security.sensitive_path_overrides
    if not entries:
        return ["  (none)"]
    return [f"  {item.path} ({item.access})" for item in entries]


def _instructions_detail(config: XcodeRuntimeConfig) -> list[str]:
    sources = config.prompt.instructions
    if not sources:
        return ["  AGENTS.md (fallback)"]
    lines: list[str] = []
    for source in sources:
        if source.type == "file":
            priority = f", {source.priority}" if source.priority else ""
            lines.append(f"  file: {source.path}{priority}")
        else:
            preview = " ".join(source.content.split())[:60]
            lines.append(f"  inline: {preview}")
    return lines


def _modules_detail(config: XcodeRuntimeConfig) -> list[str]:
    return ["  " + ", ".join(config.prompt.modules)]


SETTING_SPECS: tuple[SettingSpec, ...] = (
    SettingSpec(
        key="execution_modes.default_mode",
        label="Default Mode",
        kind=SettingKind.ENUM,
        choices=("act", "build", "plan"),
        description=(
            "'act': read allowed; writes and shell ask the user. "
            "'build': workspace mutations auto-run, boundary actions reviewed. "
            "'plan': research and plan without making changes."
        ),
    ),
    SettingSpec(
        key="security.approval_policy",
        label="Approval Policy",
        kind=SettingKind.ENUM,
        choices=("on-request", "never"),
        description=(
            "'on-request': rule-generated asks enter the approval flow. "
            "'never': actions without an existing grant are denied."
        ),
    ),
    SettingSpec(
        key="security.auto_review_timeout_seconds",
        label="Auto Review Timeout",
        kind=SettingKind.FLOAT,
        description="Total deadline for the automatic reviewer, 0-300 seconds.",
        formatter=_fmt_seconds,
    ),
    SettingSpec(
        key="security.global_default",
        label="Global Default Decision",
        kind=SettingKind.ENUM,
        choices=("allow", "ask", "deny", "default"),
        none_choice="default",
        description=(
            "Decision when no static rule matches. 'default' clears the "
            "override and defers to mode fallback."
        ),
        formatter=_fmt_default,
    ),
    SettingSpec(
        key="security.restricted_dirs",
        label="Restricted Dirs",
        kind=SettingKind.STR_LIST,
        nullable=True,
        description="Directories the agent must never access. Comma separated.",
        formatter=_fmt_str_list,
    ),
    SettingSpec(
        key="agent.max_steps",
        label="Agent Max Steps",
        kind=SettingKind.INT,
        nullable=True,
        description="Max loop turns per task. 'unlimited' removes the cap.",
        formatter=_fmt_unlimited,
    ),
    SettingSpec(
        key="agent.compact_threshold",
        label="Compact Msg Threshold",
        kind=SettingKind.INT,
        description="Message-count trigger for compaction; 0 disables.",
    ),
    SettingSpec(
        key="agent.compact_token_threshold",
        label="Compact Token Threshold",
        kind=SettingKind.INT,
        description="Token trigger for compaction; 0 disables.",
    ),
    SettingSpec(
        key="agent.max_recent_messages",
        label="Max Recent Messages",
        kind=SettingKind.INT,
        description="Recent messages kept verbatim when compacting.",
    ),
    SettingSpec(
        key="agent.keep_recent_tokens",
        label="Keep Recent Tokens",
        kind=SettingKind.INT,
        description="Token budget preserved for recent raw messages.",
    ),
    SettingSpec(
        key="agent.reserve_tokens",
        label="Reserve Tokens",
        kind=SettingKind.INT,
        description="Token budget reserved for output and tool interaction.",
    ),
    SettingSpec(
        key="agent.compact_trigger_ratio",
        label="Compact Trigger Ratio",
        kind=SettingKind.FLOAT,
        description="Auto-compact trigger as a ratio of the context window.",
    ),
    SettingSpec(
        key="agent.tool_workers",
        label="Tool Workers",
        kind=SettingKind.INT,
        description="Max active tools inside one parallel batch.",
    ),
    SettingSpec(
        key="agent.tool_timeout_seconds",
        label="Tool Timeout",
        kind=SettingKind.FLOAT,
        description="Per-tool-call timeout in seconds.",
        formatter=_fmt_seconds,
    ),
    SettingSpec(
        key="agent.watchdog_repeated_tool_limit",
        label="Watchdog Repeat Limit",
        kind=SettingKind.INT,
        description="Consecutive identical tool calls before the watchdog fires.",
    ),
    SettingSpec(
        key="request_hygiene.enabled",
        label="Request Hygiene",
        kind=SettingKind.BOOL,
        description="Compress message history sent to the model.",
        formatter=_fmt_bool,
    ),
    SettingSpec(
        key="request_hygiene.max_tool_result_bytes",
        label="Tool Result Limit",
        kind=SettingKind.INT,
        description="Maximum tool_result size in bytes.",
        formatter=_fmt_bytes,
    ),
    SettingSpec(
        key="request_hygiene.max_tool_arg_length",
        label="Tool Arg Limit",
        kind=SettingKind.INT,
        description="Maximum completed tool-call argument length.",
    ),
    SettingSpec(
        key="request_hygiene.keep_head_lines",
        label="Hygiene Head Lines",
        kind=SettingKind.INT,
        description="Head lines kept when compressing a tool_result.",
    ),
    SettingSpec(
        key="request_hygiene.keep_tail_lines",
        label="Hygiene Tail Lines",
        kind=SettingKind.INT,
        description="Tail lines kept when compressing a tool_result.",
    ),
    SettingSpec(
        key="tools.shell",
        label="Shell",
        kind=SettingKind.ENUM,
        choices=("auto", "bash", "zsh", "sh", "fish", "pwsh", "powershell", "cmd"),
        description="Shell used by the bash tool. 'auto' detects at runtime.",
    ),
    SettingSpec(
        key="tools.subagent_extra_tools",
        label="Subagent Extra Tools",
        kind=SettingKind.STR_LIST,
        nullable=True,
        description=(
            "Main-agent tools additionally granted to subagents. "
            "'todowrite' is not inherited by default. Comma separated."
        ),
        formatter=_fmt_str_list,
    ),
    SettingSpec(
        key="skills.trust_project_skills",
        label="Trust Project Skills",
        kind=SettingKind.BOOL,
        description="Discover and expose project-local skill directories.",
        formatter=_fmt_bool,
    ),
    SettingSpec(
        key="paths.sessions_dir",
        label="Sessions Dir",
        kind=SettingKind.PATH,
        nullable=True,
        description="Session transcript directory. 'none' uses .xcode/sessions.",
        formatter=_fmt_optional_path(".xcode/sessions"),
    ),
    SettingSpec(
        key="paths.skills_dir",
        label="Skills Dir",
        kind=SettingKind.PATH,
        nullable=True,
        description="Highest-priority skill scan directory. 'none' for defaults.",
        formatter=_fmt_optional_path("user defaults"),
    ),
    SettingSpec(
        key="observability.audit_path",
        label="Audit Log",
        kind=SettingKind.PATH,
        nullable=True,
        description="Audit log path. 'none' disables audit logging.",
        formatter=_fmt_optional_path("off"),
    ),
    SettingSpec(
        key="security.external_directories",
        label="External Dirs",
        kind=SettingKind.INFO,
        description="Workspace-external directories whitelisted for access.",
        getter=_count_getter("security.external_directories"),
        formatter=lambda v: f"{v} allowed",
    ),
    SettingSpec(
        key="security.sensitive_path_overrides",
        label="Sensitive Overrides",
        kind=SettingKind.INFO,
        description="Exact-path exceptions for sensitive files.",
        getter=_count_getter("security.sensitive_path_overrides"),
        formatter=lambda v: f"{v} entries",
    ),
    SettingSpec(
        key="prompt.instructions",
        label="Instructions",
        kind=SettingKind.INFO,
        description="Instruction sources injected into the system prompt.",
        getter=_count_getter("prompt.instructions"),
        formatter=lambda v: f"{v} sources",
    ),
    SettingSpec(
        key="prompt.modules",
        label="Prompt Modules",
        kind=SettingKind.INFO,
        description="Modules joined into the system prompt, in order.",
        getter=_count_getter("prompt.modules"),
        formatter=lambda v: f"{v} modules",
    ),
    SettingSpec(
        key="hooks.entries",
        label="Hooks",
        kind=SettingKind.INFO,
        description="Trusted external-command hooks. Edit JSON to change.",
        getter=_count_getter("hooks.entries"),
        formatter=lambda v: f"{v} entries",
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
    """INFO 行展开的多行详情。"""
    detail_getters: dict[str, Callable[[XcodeRuntimeConfig], list[str]]] = {
        "hooks.entries": _hooks_detail,
        "security.external_directories": _external_dirs_detail,
        "security.sensitive_path_overrides": _sensitive_overrides_detail,
        "prompt.instructions": _instructions_detail,
        "prompt.modules": _modules_detail,
    }
    getter = detail_getters.get(spec.key)
    if getter is None:
        value = spec.read(config)
        return [f"  {value}"]
    return getter(config)


def format_setting(spec: SettingSpec, config: XcodeRuntimeConfig) -> str:
    """渲染当前生效值的展示文本。"""
    value = spec.read(config)
    if spec.formatter is not None:
        return spec.formatter(value)
    if value is None:
        return "(unset)"
    if spec.kind is SettingKind.BOOL:
        return _fmt_bool(value)
    if isinstance(value, float | int):
        return _fmt_float(value)
    if isinstance(value, tuple | list):
        return _fmt_str_list(value)
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
    """把解析后的值写进原始 dict；None 表示清除该键以恢复默认。"""
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
