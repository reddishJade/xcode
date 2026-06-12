"""共享的安全策略常量与超时配置。"""

from __future__ import annotations

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
