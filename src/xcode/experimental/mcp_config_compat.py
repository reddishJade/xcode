"""MCP 配置兼容性工具：从 Claude Code 格式导入。

纯工具函数，不自动接入配置加载。
"""

from __future__ import annotations


# ── 简单的 MCP 服务器配置结构 ──


class McpCompatConfig:
    """转换后的兼容 MCP 服务器配置。"""

    def __init__(
        self,
        name: str,
        command: tuple[str, ...],
        env: dict[str, str] | None = None,
        enabled: bool = True,
        timeout: float | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.env = env
        self.enabled = enabled
        self.timeout = timeout


# ── 敏感字段辅助 ──

_SENSITIVE_HEADERS: tuple[str, ...] = (
    "authorization",
    "token",
    "api_key",
    "apikey",
    "key",
    "secret",
    "password",
    "credential",
)

_LOCAL_TRANSPORTS: set[str] = {"stdio", "local"}
_REMOTE_TRANSPORTS: set[str] = {"http", "streamable-http", "remote", "sse"}


def _is_record(value: object) -> bool:
    return isinstance(value, dict)


def _string_record(value: object) -> dict[str, str] | None:
    if not _is_record(value):
        return None
    result: dict[str, str] = {}
    for k, v in dict(value).items():  # type: ignore[arg-type]
        if isinstance(k, str) and isinstance(v, str):
            result[k] = v
    return result if result else None


def from_claude_config(
    name: str, raw: object
) -> tuple[McpCompatConfig | None, list[str]]:
    """将 Claude Code 格式的 MCP 服务器配置转换为 Xcode 兼容格式。

    参数:
        name: 服务器名称
        raw: Claude Code 配置值

    返回:
        (config_or_None, warnings)
    """
    warnings: list[str] = []

    if not _is_record(raw):
        warnings.append(
            f'skipped Claude Code MCP server "{name}"; server config is not an object.'
        )
        return None, warnings

    data: dict[str, object] = raw  # type: ignore[assignment]

    if data.get("type") == "sse":
        warnings.append(
            f'skipped Claude Code MCP server "{name}"; unsupported transport "sse".'
        )
        return None, warnings

    args_raw = data.get("args")
    args: list[str] = []
    if args_raw is not None:
        if not isinstance(args_raw, list):
            warnings.append(
                f'skipped Claude Code MCP server "{name}"; args is not an array.'
            )
            return None, warnings
        for item in args_raw:
            if not isinstance(item, str):
                warnings.append(
                    f'skipped Claude Code MCP server "{name}"; '
                    f"args must contain only strings."
                )
                return None, warnings
        args = list(args_raw)

    disabled = data.get("disabled", False)
    enabled_raw = data.get("enabled", True)
    if isinstance(disabled, bool) and disabled:
        enabled = False
    elif isinstance(enabled_raw, bool):
        enabled = enabled_raw
    else:
        enabled = True

    env_data = data.get("environment")
    if not _is_record(env_data):
        env_data = data.get("env")
    environment = _string_record(env_data)

    timeout_raw = data.get("timeout")
    timeout: float | None = None
    if isinstance(timeout_raw, (int, float)):
        timeout = float(timeout_raw)

    transport_type = data.get("type")
    if not isinstance(transport_type, str):
        transport_type = None

    command_raw = data.get("command")
    url_raw = data.get("url")

    if isinstance(command_raw, str) and (
        transport_type is None or transport_type in _LOCAL_TRANSPORTS
    ):
        command = tuple([command_raw] + args)
        return (
            McpCompatConfig(
                name=name,
                command=command,
                env=environment,
                enabled=enabled,
                timeout=timeout,
            ),
            warnings,
        )

    if command_raw is not None:
        warnings.append(
            f'skipped Claude Code MCP server "{name}"; command is not a string.'
        )
        return None, warnings

    if isinstance(url_raw, str) and (
        transport_type is None or transport_type in _REMOTE_TRANSPORTS
    ):
        warnings.append(
            f'skipped Claude Code MCP server "{name}"; '
            f"remote MCP servers are not supported in Step 9."
        )
        return None, warnings

    if url_raw is not None:
        warnings.append(
            f'skipped Claude Code MCP server "{name}"; url is not a string.'
        )
        return None, warnings

    if transport_type is not None and transport_type not in _LOCAL_TRANSPORTS:
        warnings.append(
            f'skipped Claude Code MCP server "{name}"; '
            f'unsupported transport "{transport_type}".'
        )
        return None, warnings

    warnings.append(f'skipped Claude Code MCP server "{name}"; missing command or url.')
    return None, warnings
