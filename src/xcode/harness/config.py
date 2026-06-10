from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

ProviderTransport = Literal[
    "openai_chat",
    "anthropic_messages",
    "chatglm_chat",
    "deepseek_chat",
    "mimo_chat",
]
ExecutionMode = Literal["plan", "review", "act"]
PermissionMode = Literal["strict", "normal", "permissive"]
ApprovalPolicy = Literal["always", "high_risk_only", "never"]

PROFILE_MAIN = "main"
PROFILE_SUBAGENT = "subagent"
PROFILE_FALLBACK = "fallback"
DEFAULT_PROMPT_MODULES: tuple[str, ...] = (
    "identity",
    "instructions",
    "tool_discipline",
    "tools",
    "search_strategy",
    "environment",
    "cwd",
    "git_preflight",
    "contextual_retrieval",
    "skills",
    "notices",
)


class AgentConfig(BaseModel):
    max_steps: int = 20
    execution_mode: ExecutionMode = "act"
    compact_threshold: int = 0
    compact_token_threshold: int = 0
    max_recent_messages: int = 10
    tool_workers: int = 4
    watchdog_repeated_tool_limit: int = 3


class RequestHygieneConfig(BaseModel):
    enabled: bool = True
    max_tool_result_bytes: int = 8000
    max_tool_arg_length: int = 1000
    keep_head_lines: int = 50
    keep_tail_lines: int = 50


class ModelProfileRuntimeConfig(BaseModel):
    transport: ProviderTransport = "openai_chat"
    chat_model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    thinking: bool = True
    reasoning_effort: str | None = "high"
    clear_thinking: bool = False
    tool_stream: bool = True
    response_format: dict[str, Any] | None = None


class ProviderRuntimeConfig(BaseModel):
    model_profiles: dict[str, ModelProfileRuntimeConfig] = {
        PROFILE_MAIN: ModelProfileRuntimeConfig(),
        PROFILE_SUBAGENT: ModelProfileRuntimeConfig(),
        PROFILE_FALLBACK: ModelProfileRuntimeConfig(),
    }


class SecurityRuntimeConfig(BaseModel):
    permission_mode: PermissionMode = "normal"
    sandbox_mode: bool = False
    approval_policy: ApprovalPolicy = "high_risk_only"
    network_access: bool = True
    writable_roots: tuple[str, ...] = ()
    restricted_dirs: tuple[str, ...] = ()
    deny_tools: tuple[str, ...] = ()
    ask_tools: tuple[str, ...] = ()
    allow_tools: tuple[str, ...] = ()

    def resolve_approval_policy(self) -> ApprovalPolicy:
        mode_to_policy: dict[PermissionMode, ApprovalPolicy] = {
            "strict": "always",
            "normal": "high_risk_only",
            "permissive": "never",
        }
        if self.approval_policy == "high_risk_only":
            return mode_to_policy.get(self.permission_mode, "high_risk_only")
        return self.approval_policy

    def resolve_sandbox_mode(self) -> bool:
        if self.permission_mode == "strict":
            return True
        return self.sandbox_mode


class ToolsRuntimeConfig(BaseModel):
    enabled_groups: tuple[str, ...] = ("core",)
    shell: str = "auto"


class SkillsRuntimeConfig(BaseModel):
    auto_trigger: bool = True


class PromptRuntimeConfig(BaseModel):
    modules: tuple[str, ...] = DEFAULT_PROMPT_MODULES


class PathsRuntimeConfig(BaseModel):
    sessions_dir: Path | None = None
    skills_dir: Path | None = None


class ObservabilityRuntimeConfig(BaseModel):
    audit_path: Path | None = None


class DaemonRuntimeConfig(BaseModel):
    enabled: bool = False
    interval_seconds: int = 30


class XcodeRuntimeConfig(BaseModel):
    provider: ProviderRuntimeConfig = ProviderRuntimeConfig()
    agent: AgentConfig = AgentConfig()
    tools: ToolsRuntimeConfig = ToolsRuntimeConfig()
    skills: SkillsRuntimeConfig = SkillsRuntimeConfig()
    prompt: PromptRuntimeConfig = PromptRuntimeConfig()
    paths: PathsRuntimeConfig = PathsRuntimeConfig()
    observability: ObservabilityRuntimeConfig = ObservabilityRuntimeConfig()
    daemon: DaemonRuntimeConfig = DaemonRuntimeConfig()
    request_hygiene: RequestHygieneConfig = RequestHygieneConfig()
    security: SecurityRuntimeConfig = SecurityRuntimeConfig()


# 配置发现


def discover_runtime_config(
    project_root: Path, explicit_path: Path | None = None
) -> XcodeRuntimeConfig:
    global_config = _load_global_config()
    project_config_path = explicit_path or project_root / "xcode.config.json"
    project_config = _load_json_config(project_config_path)
    local_config_path = project_root / ".local" / "settings.json"
    local_config = _load_json_config(local_config_path)
    merged = _deep_merge_configs(global_config, project_config)
    merged = _deep_merge_configs(merged, local_config)
    merged = _apply_env_overrides(merged)
    return merged


def _load_global_config() -> XcodeRuntimeConfig:
    home = Path.home()
    return _load_json_config(home / ".xcode" / "settings.json")


def _load_json_config(path: Path | None) -> XcodeRuntimeConfig:
    if path is None or not path.exists():
        return XcodeRuntimeConfig()
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return _from_dict_preserving_profiles(data)


def _from_dict_preserving_profiles(data: dict[str, Any]) -> XcodeRuntimeConfig:
    """从 dict 构造 config，保留 model_profiles 的 profile 继承逻辑。"""
    provider = data.get("provider", {})
    raw_profiles = provider.get("model_profiles", {})
    if isinstance(raw_profiles, dict):
        provider = dict(provider)
        provider["model_profiles"] = _resolve_model_profiles(raw_profiles)
    return XcodeRuntimeConfig.model_validate({**data, "provider": provider})


def _resolve_model_profiles(
    raw_profiles: dict[str, Any],
) -> dict[str, Any]:
    """解析 model_profiles：继承 main profile 的默认值。"""
    main_raw = raw_profiles.get(PROFILE_MAIN, {})
    resolved: dict[str, Any] = {PROFILE_MAIN: main_raw}
    for name, raw in raw_profiles.items():
        if name == PROFILE_MAIN:
            continue
        if isinstance(raw, str):
            resolved[name] = {**main_raw, "chat_model": raw}
        elif isinstance(raw, dict):
            profile = dict(main_raw)
            profile.update(raw)
            resolved[name] = profile
    resolved.setdefault(PROFILE_SUBAGENT, resolved.get(PROFILE_MAIN, {}))
    resolved.setdefault(PROFILE_FALLBACK, resolved.get(PROFILE_MAIN, {}))
    return resolved


def _deep_merge_configs(
    base: XcodeRuntimeConfig, override: XcodeRuntimeConfig
) -> XcodeRuntimeConfig:
    """深度合并两个配置，override 的非默认字段覆盖 base。"""
    base_dict = base.model_dump()
    override_dict = override.model_dump()
    default_dict = XcodeRuntimeConfig().model_dump()
    merged = _merge_non_default(base_dict, override_dict, default_dict)
    return XcodeRuntimeConfig.model_validate(merged)


def _merge_non_default(base: dict, override: dict, default: dict) -> dict:
    """递归合并，只覆盖非默认值。"""
    result = dict(base)
    for key, val in override.items():
        if key not in base:
            result[key] = val
        elif isinstance(val, dict) and isinstance(base[key], dict):
            result[key] = _merge_non_default(base[key], val, default.get(key, {}))
        elif val != default.get(key):
            result[key] = val
    return result


def _apply_env_overrides(config: XcodeRuntimeConfig) -> XcodeRuntimeConfig:
    import os

    permission_mode = os.getenv("XCODE_PERMISSION_MODE")
    sandbox_mode = os.getenv("XCODE_SANDBOX_MODE")
    approval_policy = os.getenv("XCODE_APPROVAL_POLICY")

    security_updates: dict[str, Any] = {}

    if permission_mode in ("strict", "normal", "permissive"):
        security_updates["permission_mode"] = permission_mode

    if sandbox_mode in ("true", "false"):
        security_updates["sandbox_mode"] = sandbox_mode == "true"

    if approval_policy in ("always", "high_risk_only", "never"):
        security_updates["approval_policy"] = approval_policy

    if security_updates:
        security = config.security.model_copy(update=security_updates)
        config = config.model_copy(update={"security": security})

    return config


def load_runtime_config(path: Path | None) -> XcodeRuntimeConfig:
    """加载单文件运行时配置（公开 API，用于测试）。"""
    return _load_json_config(path)


def resolve_config_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_root / path
