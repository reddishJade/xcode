from __future__ import annotations

import json
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Union, get_origin, get_type_hints

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


@dataclass
class AgentConfig:
    max_steps: int = 20
    execution_mode: ExecutionMode = "act"
    compact_threshold: int = 0
    compact_token_threshold: int = 0
    max_recent_messages: int = 10
    tool_workers: int = 4
    watchdog_repeated_tool_limit: int = 3


@dataclass
class RequestHygieneConfig:
    enabled: bool = True
    max_tool_result_bytes: int = 8000
    max_tool_arg_length: int = 1000
    keep_head_lines: int = 50
    keep_tail_lines: int = 50


@dataclass
class ModelProfileRuntimeConfig:
    transport: ProviderTransport = "openai_chat"
    chat_model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    thinking: bool = True
    reasoning_effort: str | None = "high"
    clear_thinking: bool = False
    tool_stream: bool = True
    response_format: dict[str, Any] | None = None


@dataclass
class ProviderRuntimeConfig:
    model_profiles: dict[str, ModelProfileRuntimeConfig] = field(
        default_factory=lambda: {
            PROFILE_MAIN: ModelProfileRuntimeConfig(),
            PROFILE_SUBAGENT: ModelProfileRuntimeConfig(),
            PROFILE_FALLBACK: ModelProfileRuntimeConfig(),
        }
    )


@dataclass
class SecurityRuntimeConfig:
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


@dataclass
class ToolsRuntimeConfig:
    enabled_groups: tuple[str, ...] = ("core",)
    shell: str = "auto"


@dataclass
class SkillsRuntimeConfig:
    auto_trigger: bool = True


@dataclass
class PromptRuntimeConfig:
    modules: tuple[str, ...] = DEFAULT_PROMPT_MODULES


@dataclass
class PathsRuntimeConfig:
    sessions_dir: Path | None = None
    skills_dir: Path | None = None


@dataclass
class ObservabilityRuntimeConfig:
    audit_path: Path | None = None


@dataclass
class DaemonRuntimeConfig:
    enabled: bool = False
    interval_seconds: int = 30


@dataclass
class XcodeRuntimeConfig:
    provider: ProviderRuntimeConfig = field(default_factory=ProviderRuntimeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsRuntimeConfig = field(default_factory=ToolsRuntimeConfig)
    skills: SkillsRuntimeConfig = field(default_factory=SkillsRuntimeConfig)
    prompt: PromptRuntimeConfig = field(default_factory=PromptRuntimeConfig)
    paths: PathsRuntimeConfig = field(default_factory=PathsRuntimeConfig)
    observability: ObservabilityRuntimeConfig = field(
        default_factory=ObservabilityRuntimeConfig
    )
    daemon: DaemonRuntimeConfig = field(default_factory=DaemonRuntimeConfig)
    request_hygiene: RequestHygieneConfig = field(default_factory=RequestHygieneConfig)
    security: SecurityRuntimeConfig = field(default_factory=SecurityRuntimeConfig)


# ── 序列化 / 反序列化 ──


def _config_to_dict(config: XcodeRuntimeConfig) -> dict[str, Any]:
    """将 XcodeRuntimeConfig 转为可 JSON 序列化的 dict。"""

    def _to_dict(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return {
                f.name: _to_dict(getattr(obj, f.name))
                for f in obj.__dataclass_fields__.values()
            }
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(item) for item in obj]
        return obj

    return _to_dict(config)


def _config_from_dict(data: dict[str, Any]) -> XcodeRuntimeConfig:
    """从 dict 递归构建 XcodeRuntimeConfig。"""

    def _build(cls: type, d: dict[str, Any]) -> Any:
        field_types = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for name, ftype in field_types.items():
            if name not in d:
                continue
            val = d[name]
            kwargs[name] = _resolve_value(ftype, val)
        return cls(**kwargs)

    def _resolve_value(ftype: Any, val: Any) -> Any:
        if val is None:
            return None

        origin = get_origin(ftype)
        args = getattr(ftype, "__args__", ())

        # 处理 Union 类型 (如 Path | None, PathsRuntimeConfig | None)
        if origin is types.UnionType or origin is Union:
            non_none_args = [a for a in args if a is not type(None)]
            if non_none_args:
                return _resolve_value(non_none_args[0], val)
            return val

        # 处理嵌套 dataclass
        if hasattr(ftype, "__dataclass_fields__"):
            return _build(ftype, val)

        # 处理 Path
        if ftype is Path or (isinstance(ftype, type) and issubclass(ftype, Path)):
            return Path(val) if val else None

        # 处理元组
        if origin is tuple:
            if isinstance(val, list):
                return tuple(val)
            return val

        # 处理字典
        if origin is dict:
            if args and len(args) >= 2:
                vtype = args[1]
                return {k: _resolve_value(vtype, v) for k, v in val.items()}
            return val

        return val

    return _build(XcodeRuntimeConfig, data)


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
    provider = data.get("provider", {})
    raw_profiles = provider.get("model_profiles", {})
    if isinstance(raw_profiles, dict):
        provider = dict(provider)
        provider["model_profiles"] = _resolve_model_profiles(raw_profiles)
    return _config_from_dict({**data, "provider": provider})


def _resolve_model_profiles(
    raw_profiles: dict[str, Any],
) -> dict[str, Any]:
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
    base_dict = _config_to_dict(base)
    override_dict = _config_to_dict(override)
    default_dict = _config_to_dict(XcodeRuntimeConfig())
    merged = _merge_non_default(base_dict, override_dict, default_dict)
    return _config_from_dict(merged)


def _merge_non_default(base: dict, override: dict, default: dict) -> dict:
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
        security = replace(config.security, **security_updates)
        config = replace(config, security=security)

    return config


def load_runtime_config(path: Path | None) -> XcodeRuntimeConfig:
    return _load_json_config(path)


def resolve_config_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_root / path
