"""安全策略与 ruleset 辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from xcode.harness.config import (
    ModeRuleRuntimeConfig,
    SecurityRuntimeConfig,
    XcodeRuntimeConfig,
)
from xcode.harness.security import (
    PermissionDecision,
    PermissionPolicy,
    StaticPermission,
)
from xcode.harness.security.permission_model import (
    ExternalDirectory,
    Rule,
    SensitivePathOverride,
)


# 用户配置中的权限名。这里刻意不复用内部 capability，避免把 webfetch 等
# 网络工具意外包含到文件读取权限中。
_PERMISSION_TOOLS: dict[str, tuple[str, ...]] = {
    "read": (
        "read_file",
        "glob_files",
        "grep_search",
        "find_files",
        "list_dir",
    ),
    "edit": ("write_file", "edit_file", "apply_patch"),
    "shell": ("bash", "shell"),
    "web": ("websearch", "webfetch"),
    "subagent": ("subagent",),
    "skill": ("load_skill",),
}


def _rule_from_runtime_config(rule: ModeRuleRuntimeConfig) -> Rule:
    return Rule(
        action=rule.action,
        effect=rule.effect,
        command=rule.command,
        subcommand=rule.subcommand,
        subcommand_in=set(rule.subcommand_in)
        if rule.subcommand_in is not None
        else None,
        flags_any=set(rule.flags_any) if rule.flags_any is not None else None,
        flags_all=set(rule.flags_all) if rule.flags_all is not None else None,
        resource_pattern=rule.resource_pattern,
    )


def mode_rulesets_from_runtime_config(
    runtime_config: XcodeRuntimeConfig,
) -> dict[str, tuple[Rule, ...]]:
    modes = runtime_config.execution_modes
    result: dict[str, tuple[Rule, ...]] = {}
    for mode_name, ruleset in (
        ("plan", modes.plan),
        ("build", modes.build),
        ("act", modes.act),
    ):
        if ruleset.rules:
            result[mode_name] = tuple(
                _rule_from_runtime_config(rule) for rule in ruleset.rules
            )
    return result


def external_directories_from_security(
    security: SecurityRuntimeConfig,
) -> tuple[ExternalDirectory, ...]:
    dirs: list[ExternalDirectory] = [
        ExternalDirectory(path=Path(ed.path), access=ed.access)
        for ed in security.external_directories
    ]
    home = Path.home()
    for p in (home / ".xcode", home / ".agents"):
        if p.is_dir():
            ext = ExternalDirectory(path=p, access="read")
            if ext not in dirs:
                dirs.append(ext)
    return tuple(dirs)


def sensitive_path_overrides_from_security(
    security: SecurityRuntimeConfig,
    project_root: Path,
) -> tuple[SensitivePathOverride, ...]:
    """将用户配置转换为规范化的精确敏感路径例外。"""
    overrides: list[SensitivePathOverride] = []
    for item in security.sensitive_path_overrides:
        path = Path(item.path)
        if not path.is_absolute():
            path = project_root / path
        overrides.append(SensitivePathOverride(path=path, access=item.access))
    return tuple(overrides)


def permission_policy_from_security(
    security: SecurityRuntimeConfig,
) -> PermissionPolicy | None:
    rules: list[StaticPermission] = []

    # 权限名称先展开为具体工具；具体工具配置随后追加，从而覆盖权限名称。
    for permission, decision in security.permissions.items():
        for tool in _PERMISSION_TOOLS.get(permission, ()):
            rules.append(StaticPermission(tool=tool, decision=decision))

    for tool, decision in security.tools.items():
        rules.append(StaticPermission(tool=tool, decision=decision))

    global_default: str | None = security.global_default
    if global_default is None and security.resolve_approval_policy() == "always":
        global_default = "ask"
    if not rules and global_default is None:
        return None
    return PermissionPolicy(
        tuple(rules), global_default=cast(PermissionDecision, global_default)
    )
