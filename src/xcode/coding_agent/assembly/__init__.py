"""应用装配工厂函数。

子包拆分：
  config.py   — 配置解析
  infra.py    — 共享基础设施构建
  registry.py — 工具注册表构建
  agent.py    — Agent 构建 + Hook Manager
  security.py — 安全策略与 ruleset 辅助函数
"""

from .agent import build_agent, build_hook_manager
from .config import ResolvedConfig, resolve_config
from .infra import SharedInfra, build_shared_infra
from .registry import (
    build_search_tools_tool,
    build_tool_registry,
)
from .security import (
    build_shell_from_security,
    external_directories_from_security,
    mode_rulesets_from_runtime_config,
    permission_policy_from_security,
    sensitive_path_overrides_from_security,
)

__all__ = [
    "ResolvedConfig",
    "SharedInfra",
    "build_agent",
    "build_hook_manager",
    "build_search_tools_tool",
    "build_shared_infra",
    "build_shell_from_security",
    "build_tool_registry",
    "external_directories_from_security",
    "mode_rulesets_from_runtime_config",
    "permission_policy_from_security",
    "resolve_config",
    "sensitive_path_overrides_from_security",
]
