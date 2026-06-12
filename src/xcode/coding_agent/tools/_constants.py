"""共享的安全策略常量、超时配置与风险评估。"""

from __future__ import annotations

from xcode.harness.observability.permissions import PermissionDecision

# 命令执行超时配置
DEFAULT_TIMEOUT_SECONDS: int = 30
MAX_TIMEOUT_SECONDS: int = 120

# 危险命令模式（拒绝执行）
DANGEROUS_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs.",
    "> /dev/sda",
    "dd if=",
    "chmod -R 777",
    "chown -R root",
]

# 高风险写操作命令前缀（需要用户确认）
HIGH_RISK_WRITE_COMMANDS: list[str] = [
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
    for pattern in DANGEROUS_PATTERNS:
        if pattern in normalized:
            return "deny"
    for prefix in HIGH_RISK_WRITE_COMMANDS:
        if normalized.startswith(prefix):
            return "ask"
    return "allow"
