"""安全策略与 ruleset 辅助函数。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import cast

from xcode.harness.config import (
    ModeRuleRuntimeConfig,
    SecurityRuntimeConfig,
    XcodeRuntimeConfig,
)
from xcode.harness.execution_env import (
    LinuxBubblewrapSandbox,
    NetworkAccess,
    SandboxMode,
    SandboxPolicy,
    Shell,
    SubprocessShell,
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


def build_shell_from_security(
    project_root: Path,
    security: SecurityRuntimeConfig,
) -> Shell:
    """按运行时安全配置构造 Agent shell。"""
    sandbox = security.sandbox
    if sys.platform != "linux":
        return SubprocessShell()
    if (
        sandbox.mode is SandboxMode.DANGER_FULL_ACCESS
        and sandbox.network_access is NetworkAccess.ALLOW
    ):
        return SubprocessShell()
    policy = sandbox_policy_from_security(project_root, security)
    return SubprocessShell(sandbox=LinuxBubblewrapSandbox(policy))


def sandbox_policy_from_security(
    project_root: Path,
    security: SecurityRuntimeConfig,
) -> SandboxPolicy:
    """把用户配置转换为 Linux bubblewrap 文件和网络策略。"""
    writable_roots: list[Path] = []
    if security.sandbox.mode is SandboxMode.WORKSPACE_WRITE:
        temp_root = Path(tempfile.gettempdir())
        if temp_root.is_dir():
            writable_roots.append(temp_root)
        if security.non_workspace_access:
            for item in security.external_directories:
                if item.access not in {"write", "read_write"}:
                    continue
                path = Path(item.path).expanduser()
                if not path.is_absolute():
                    path = project_root / path
                if path.is_dir():
                    writable_roots.append(path)
    return SandboxPolicy(
        project_root=project_root,
        mode=security.sandbox.mode,
        network_access=security.sandbox.network_access,
        writable_roots=tuple(writable_roots),
    )


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
    """装配外部目录白名单；non_workspace_access 关闭时忽略用户白名单。

    ~/.xcode 与 ~/.agents 属于 xcode 自身基础设施（记忆、技能），不受该
    开关影响，始终保留只读授权。
    """
    dirs: list[ExternalDirectory] = (
        [
            ExternalDirectory(path=Path(ed.path), access=ed.access)
            for ed in security.external_directories
        ]
        if security.non_workspace_access
        else []
    )
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

    # approval_policy 只决定 ask 是否进入审批。
    global_default: str | None = security.global_default
    if not rules and global_default is None:
        return None
    return PermissionPolicy(
        tuple(rules), global_default=cast(PermissionDecision, global_default)
    )
