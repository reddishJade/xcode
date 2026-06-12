"""共享的安全策略常量、超时配置与风险评估。"""

from __future__ import annotations

import shlex

from xcode.harness.observability.permissions import PermissionDecision

# 命令执行超时配置
DEFAULT_TIMEOUT_SECONDS: int = 30
MAX_TIMEOUT_SECONDS: int = 120

# 危险命令模式（拒绝执行）
DANGEROUS_PATTERNS: list[str] = [
    "mkfs.",
    "> /dev/sda",
    "> /dev/sdb",
    "> /dev/nvme",
    "dd if=",
]

# 高风险写操作命令前缀（需要用户确认）
HIGH_RISK_WRITE_COMMANDS: list[str] = [
    "chmod -r ",
    "chmod -r\t",
    "chown -r ",
    "chown -r\t",
    "rm ",
    "rm\t",
    "mv ",
    "mv\t",
    "git reset --hard",
    "git clean -f",
    "git push --force",
    "git push -f",
]


def evaluate_command_risk(command: str) -> PermissionDecision:
    """评估 shell 命令的风险级别。

    返回：
    - "deny"  — 危险命令，直接拒绝
    - "ask"   — 高风险写操作，需要用户确认
    - "allow" — 普通命令，放行
    """
    normalized = command.strip().lower()
    if _deletes_root_path(normalized):
        return "deny"
    if _touches_system_path_with_recursive_mutation(normalized):
        return "deny"
    for pattern in DANGEROUS_PATTERNS:
        if pattern in normalized:
            return "deny"
    if _is_recursive_permission_change(normalized):
        return "ask"
    for prefix in HIGH_RISK_WRITE_COMMANDS:
        if normalized.startswith(prefix):
            return "ask"
    return "allow"


def _deletes_root_path(command: str) -> bool:
    """识别真正指向根目录或根通配的 rm 目标。"""
    tokens = _shell_tokens(command)
    if not tokens or tokens[0] != "rm":
        return False

    has_recursive = False
    for token in tokens[1:]:
        if token == "--":
            continue
        if token.startswith("-"):
            has_recursive = has_recursive or "r" in token
            continue
        if has_recursive and token in {"/", "/*"}:
            return True
    return False


def _touches_system_path_with_recursive_mutation(command: str) -> bool:
    """识别递归修改系统目录的命令。"""
    tokens = _shell_tokens(command)
    if len(tokens) < 3 or tokens[0] not in {"rm", "mv", "chmod", "chown"}:
        return False
    if not _has_recursive_flag(tokens):
        return False
    return any(_is_protected_system_target(token) for token in tokens[1:])


def _is_recursive_permission_change(command: str) -> bool:
    """递归权限和属主修改需要人工确认。"""
    tokens = _shell_tokens(command)
    return (
        bool(tokens) and tokens[0] in {"chmod", "chown"} and _has_recursive_flag(tokens)
    )


def _has_recursive_flag(tokens: list[str]) -> bool:
    """判断 shell token 中是否包含递归选项。"""
    for token in tokens[1:]:
        if token == "--":
            return False
        if token.startswith("-") and "r" in token:
            return True
    return False


def _is_protected_system_target(token: str) -> bool:
    """判断目标是否为不应由 agent 触碰的系统级路径。"""
    protected_targets = {
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/sys",
        "/usr",
        "/var",
    }
    cleaned = token.rstrip("/")
    if cleaned in {"", "/", "/*"}:
        return True
    return cleaned in protected_targets


def _shell_tokens(command: str) -> list[str]:
    """解析 shell 命令，解析失败时返回空列表。"""
    try:
        return shlex.split(command)
    except ValueError:
        return []
