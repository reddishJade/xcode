from __future__ import annotations

import json
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Union, get_origin, get_type_hints

ProviderTransport = Literal[
    "openai_chat",
    "chatglm_chat",
    "deepseek_chat",
    "mimo_chat",
]
ExecutionMode = Literal["plan", "build", "act"]
PermissionMode = Literal["strict", "normal", "permissive"]
ApprovalPolicy = Literal["always", "never"]
HookEventName = Literal[
    "pre_tool",
    "post_tool",
    "on_error",
    "on_compact",
    "before_agent_start",
    "before_provider_request",
]
HookFailurePolicy = Literal["ignore", "warn", "fail"]

PROFILE_MAIN = "main"
PROFILE_SUBAGENT = "subagent"
PROFILE_FALLBACK = "fallback"
DEFAULT_PROMPT_MODULES: tuple[str, ...] = (
    "identity",
    "tool_discipline",
    "tools",
    "search_strategy",
    "environment",
    "cwd",
    "git_preflight",
    "contextual_retrieval",
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


_LEGACY_SECURITY_FIELDS: frozenset[str] = frozenset(
    {
        "deny_tools",
        "ask_tools",
        "allow_tools",
    }
)


@dataclass
class SecurityRuntimeConfig:
    permission_mode: PermissionMode = "normal"
    sandbox_mode: bool = False
    approval_policy: str = "never"
    network_access: bool = True
    writable_roots: tuple[str, ...] = ()
    restricted_dirs: tuple[str, ...] = ()
    rules: tuple[dict[str, Any], ...] = ()
    global_default: str | None = None
    external_directories: tuple[dict[str, Any], ...] = ()

    def resolve_approval_policy(self) -> str:
        if self.approval_policy in ("always", "never"):
            return self.approval_policy
        return "never"

    def resolve_sandbox_mode(self) -> bool:
        if self.permission_mode == "strict":
            return True
        return self.sandbox_mode


def _validate_legacy_security_fields(raw: dict[str, Any]) -> None:
    """Fail-fast: legacy deny_tools/ask_tools/allow_tools detected."""
    security = raw.get("security")
    if not isinstance(security, dict):
        return
    found = _LEGACY_SECURITY_FIELDS & set(security)
    if found:
        raise ValueError(
            "Security config contains deprecated fields: "
            f"{', '.join(sorted(found))}. "
            "Migrate to 'security.rules' and 'security.global_default'."
        )


def _validate_external_directories(raw: dict[str, Any]) -> None:
    """Fail-fast: invalid external_directory entries."""
    security = raw.get("security")
    if not isinstance(security, dict):
        return
    ext = security.get("external_directories")
    if not isinstance(ext, list):
        return
    for i, entry in enumerate(ext):
        if not isinstance(entry, dict):
            raise ValueError(f"security.external_directories[{i}]: must be an object")
        path_val = entry.get("path")
        if not isinstance(path_val, str) or not path_val.strip():
            raise ValueError(
                f"security.external_directories[{i}]: 'path' is required "
                "and must be a non-empty string"
            )
        access_val = entry.get("access", "read")
        if access_val not in ("read", "write", "read_write"):
            raise ValueError(
                f"security.external_directories[{i}]: 'access' must be one of "
                "'read', 'write', 'read_write'"
            )


_INSTRUCTION_PRIORITIES: frozenset[str] = frozenset(
    {"critical", "high", "medium", "low"}
)


def _validate_instruction_sources(raw: dict[str, Any]) -> None:
    """Fail-fast: invalid prompt.instructions entries."""
    prompt = raw.get("prompt")
    if not isinstance(prompt, dict):
        return
    instructions = prompt.get("instructions")
    if not isinstance(instructions, list):
        return
    for i, entry in enumerate(instructions):
        if not isinstance(entry, dict):
            raise ValueError(f"prompt.instructions[{i}]: must be an object")
        typ = entry.get("type")
        if typ not in ("file", "inline"):
            raise ValueError(
                f"prompt.instructions[{i}]: type must be 'file' or 'inline'"
            )
        if typ == "file":
            path_raw = entry.get("path")
            if not isinstance(path_raw, str) or not path_raw.strip():
                raise ValueError(
                    f"prompt.instructions[{i}]: file path must be a non-empty string"
                )
            norm = path_raw.replace("\\", "/")
            if norm.startswith("/"):
                raise ValueError(
                    f"prompt.instructions[{i}]: absolute path not allowed: {path_raw}"
                )
            if len(norm) >= 2 and norm[1] == ":":
                raise ValueError(
                    f"prompt.instructions[{i}]: absolute path not allowed: {path_raw}"
                )
            if norm.startswith("~"):
                raise ValueError(
                    f"prompt.instructions[{i}]: home-relative path not allowed: "
                    f"{path_raw}"
                )
            for segment in norm.split("/"):
                if segment == "..":
                    raise ValueError(
                        f"prompt.instructions[{i}]: traversal path not allowed: "
                        f"{path_raw}"
                    )
        if typ == "inline":
            content = entry.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"prompt.instructions[{i}]: inline content must be a "
                    "non-empty string"
                )
        priority_raw = entry.get("priority")
        if priority_raw is not None:
            if (
                not isinstance(priority_raw, str)
                or priority_raw.lower() not in _INSTRUCTION_PRIORITIES
            ):
                raise ValueError(
                    f"prompt.instructions[{i}]: invalid priority '{priority_raw}'. "
                    "Must be one of: critical, high, medium, low"
                )


@dataclass
class ToolsRuntimeConfig:
    enabled_groups: tuple[str, ...] = ("core", "skills")
    subagent_tool_allowlist: tuple[str, ...] = ()
    shell: str = "auto"


@dataclass
class SkillsRuntimeConfig:
    trust_project_skills: bool = False


@dataclass
class PromptRuntimeConfig:
    modules: tuple[str, ...] = DEFAULT_PROMPT_MODULES
    instructions: tuple[dict, ...] = ()


@dataclass
class PathsRuntimeConfig:
    sessions_dir: Path | None = None
    skills_dir: Path | None = None


@dataclass
class ObservabilityRuntimeConfig:
    audit_path: Path | None = None


@dataclass(frozen=True)
class ExternalHookRuntimeConfig:
    """单个受信任外部命令 hook 声明。"""

    event: HookEventName
    command: tuple[str, ...]
    matcher: str | None = None
    timeout: float = 10.0
    enabled: bool = True
    failure_policy: HookFailurePolicy = "warn"
    inherit_to_subagents: bool = False
    source: str = ""


@dataclass
class HooksRuntimeConfig:
    """外部命令 hook 配置集合。"""

    entries: tuple[ExternalHookRuntimeConfig, ...] = ()


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
    hooks: HooksRuntimeConfig = field(default_factory=HooksRuntimeConfig)
    daemon: DaemonRuntimeConfig = field(default_factory=DaemonRuntimeConfig)
    request_hygiene: RequestHygieneConfig = field(default_factory=RequestHygieneConfig)
    security: SecurityRuntimeConfig = field(default_factory=SecurityRuntimeConfig)


# ── 序列化 / 反序列化 ──


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

        # 处理 Union 类型 (如 Path | None, str | int, PathsRuntimeConfig | None)
        if origin is types.UnionType or origin is Union:
            non_none_args = [a for a in args if a is not type(None)]
            for candidate in non_none_args:
                try:
                    return _resolve_value(candidate, val)
                except (TypeError, ValueError):
                    continue
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
                item_type = args[0] if args else object
                return tuple(_resolve_value(item_type, item) for item in val)
            return val

        # 处理字典
        if origin is dict:
            if args and len(args) >= 2:
                vtype = args[1]
                return {k: _resolve_value(vtype, v) for k, v in val.items()}
            return val

        return val

    return _build(XcodeRuntimeConfig, data)


# ── 配置发现（基于 raw dict 合并，仅覆盖显式指定的键）──


def discover_runtime_config(
    project_root: Path, explicit_path: Path | None = None
) -> XcodeRuntimeConfig:
    global_path = Path.home() / ".xcode" / "settings.json"
    project_path = explicit_path or project_root / "xcode.config.json"
    local_path = project_root / ".local" / "settings.json"
    global_raw = _load_raw_config(global_path)
    project_raw = _load_raw_config(project_path)
    local_raw = _load_raw_config(local_path)

    global_raw = _resolve_profiles_in_raw(global_raw)
    project_raw = _resolve_profiles_in_raw(project_raw)
    local_raw = _resolve_profiles_in_raw(local_raw)
    global_raw = _annotate_hook_sources(global_raw, global_path)
    project_raw = _annotate_hook_sources(project_raw, project_path)
    local_raw = _annotate_hook_sources(local_raw, local_path)

    merged = _deep_merge_raw(global_raw, project_raw)
    merged = _deep_merge_raw(merged, local_raw)

    _validate_legacy_security_fields(merged)
    _validate_external_directories(merged)
    _validate_instruction_sources(merged)
    _validate_external_hooks(merged)
    config = _config_from_dict(merged)
    config = _apply_env_overrides(config)
    return config


def _load_raw_config(path: Path | None) -> dict[str, Any]:
    """读取 JSON 配置为原始 dict；文件不存在返回空 dict。"""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"runtime config must be a JSON object: {path}")
    return data


def _annotate_hook_sources(
    data: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    """为当前配置层的 hook 声明附加来源路径。"""
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return data
    entries = hooks.get("entries")
    if not isinstance(entries, list):
        return data

    annotated_entries: list[object] = []
    for entry in entries:
        if not isinstance(entry, dict):
            annotated_entries.append(entry)
            continue
        annotated = dict(entry)
        annotated["source"] = str(source_path)
        annotated_entries.append(annotated)

    result = dict(data)
    result["hooks"] = dict(hooks)
    result["hooks"]["entries"] = annotated_entries
    return result


_HOOK_EVENTS: frozenset[str] = frozenset(
    {
        "pre_tool",
        "post_tool",
        "on_error",
        "on_compact",
        "before_agent_start",
        "before_provider_request",
    }
)
_HOOK_FAILURE_POLICIES: frozenset[str] = frozenset({"ignore", "warn", "fail"})


def _validate_external_hooks(raw: dict[str, Any]) -> None:
    """Fail-fast 校验 hooks.entries 声明。"""
    hooks = raw.get("hooks")
    if hooks is None:
        return
    if not isinstance(hooks, dict):
        raise ValueError("hooks must be an object")
    entries = hooks.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("hooks.entries must be an array")
    for index, entry in enumerate(entries):
        _validate_external_hook_entry(entry, index)


def _validate_external_hook_entry(entry: object, index: int) -> None:
    """校验单个外部 hook 声明。"""
    prefix = f"hooks.entries[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{prefix}: must be an object")
    event = entry.get("event")
    if event not in _HOOK_EVENTS:
        allowed = ", ".join(sorted(_HOOK_EVENTS))
        raise ValueError(f"{prefix}.event: must be one of {allowed}")

    command = entry.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"{prefix}.command: must be a non-empty argv array")
    if any(not isinstance(item, str) or not item.strip() for item in command):
        raise ValueError(f"{prefix}.command: argv items must be non-empty strings")

    matcher = entry.get("matcher")
    if matcher is not None and (
        not isinstance(matcher, str) or not matcher.strip()
    ):
        raise ValueError(f"{prefix}.matcher: must be a non-empty string")

    timeout = entry.get("timeout", 10.0)
    if (
        not isinstance(timeout, int | float)
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise ValueError(f"{prefix}.timeout: must be a positive number")

    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{prefix}.enabled: must be a boolean")

    failure_policy = entry.get("failure_policy", "warn")
    if failure_policy not in _HOOK_FAILURE_POLICIES:
        allowed = ", ".join(sorted(_HOOK_FAILURE_POLICIES))
        raise ValueError(f"{prefix}.failure_policy: must be one of {allowed}")

    inherit = entry.get("inherit_to_subagents", False)
    if not isinstance(inherit, bool):
        raise ValueError(f"{prefix}.inherit_to_subagents: must be a boolean")


def _resolve_profiles_in_raw(data: dict[str, Any]) -> dict[str, Any]:
    """在原始 dict 中展开 model_profiles 继承。"""
    if not data:
        return data
    provider = data.get("provider")
    if not isinstance(provider, dict):
        return data
    raw_profiles = provider.get("model_profiles")
    if not isinstance(raw_profiles, dict):
        return data
    result = dict(data)
    result["provider"] = dict(provider)
    result["provider"]["model_profiles"] = _resolve_model_profiles(raw_profiles)
    return result


def _resolve_model_profiles(
    raw_profiles: dict[str, object],
) -> dict[str, object]:
    """在原始 dict 中展开 model_profiles 继承。"""
    main_raw: dict[str, object] = {}
    main_entry = raw_profiles.get(PROFILE_MAIN)
    if isinstance(main_entry, dict):
        main_raw = {str(k): v for k, v in main_entry.items()}
    resolved: dict[str, object] = {PROFILE_MAIN: main_raw}
    main_transport: ProviderTransport = "openai_chat"
    main_transport_raw = main_raw.get("transport")
    if isinstance(main_transport_raw, str):
        main_transport = _load_provider_transport(main_transport_raw, main_transport)
        main_raw["transport"] = main_transport
    for name, raw in raw_profiles.items():
        if name == PROFILE_MAIN:
            continue
        if isinstance(raw, str):
            resolved[name] = {**main_raw, "chat_model": raw}
        elif isinstance(raw, dict):
            profile: dict[str, object] = dict(main_raw)
            profile.update({str(k): v for k, v in raw.items()})
            resolved[name] = profile
    resolved.setdefault(PROFILE_SUBAGENT, resolved.get(PROFILE_MAIN, {}))
    resolved.setdefault(PROFILE_FALLBACK, resolved.get(PROFILE_MAIN, {}))
    return resolved


def _deep_merge_raw(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并 override 到 base；仅在 override 中显式存在的键覆盖 base。

    与旧 _merge_non_default 的区别：不比较 dataclass 默认值，
    仅依据键是否在 override dict 中出现。用户显式设回默认值的键也会正确保留。
    """
    if not override:
        return dict(base)
    result = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_raw(result[key], val)
        else:
            result[key] = val
    return result


def _apply_env_overrides(config: XcodeRuntimeConfig) -> XcodeRuntimeConfig:
    import os

    sandbox_mode = os.getenv("XCODE_SANDBOX_MODE")
    security_updates: dict[str, Any] = {}

    parsed_permission_mode = _parse_permission_mode(os.getenv("XCODE_PERMISSION_MODE"))
    if parsed_permission_mode is not None:
        security_updates["permission_mode"] = parsed_permission_mode

    if sandbox_mode in ("true", "false"):
        security_updates["sandbox_mode"] = sandbox_mode == "true"

    parsed_approval_policy = _parse_approval_policy(os.getenv("XCODE_APPROVAL_POLICY"))
    if parsed_approval_policy is not None:
        security_updates["approval_policy"] = parsed_approval_policy

    if security_updates:
        security = replace(config.security, **security_updates)
        config = replace(config, security=security)

    return config


def _parse_permission_mode(value: object) -> PermissionMode | None:
    if not isinstance(value, str):
        return None
    match value:
        case "strict":
            return "strict"
        case "normal":
            return "normal"
        case "permissive":
            return "permissive"
        case _:
            return None


def _parse_approval_policy(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if value in ("always", "never"):
        return value
    return "never"


def _load_provider_transport(
    value: object, default: ProviderTransport
) -> ProviderTransport:
    if not isinstance(value, str):
        return default
    match value:
        case "openai_chat":
            return "openai_chat"
        case "chatglm_chat":
            return "chatglm_chat"
        case "deepseek_chat":
            return "deepseek_chat"
        case "mimo_chat":
            return "mimo_chat"
        case _:
            raise ValueError(
                f"Unsupported provider transport: {value!r}. "
                "Supported transports: openai_chat, chatglm_chat, "
                "deepseek_chat, mimo_chat"
            )


def load_runtime_config(path: Path | None) -> XcodeRuntimeConfig:
    raw = _load_raw_config(path)
    raw = _resolve_profiles_in_raw(raw)
    if path is not None:
        raw = _annotate_hook_sources(raw, path)
    _validate_external_hooks(raw)
    return _config_from_dict(raw)


def resolve_config_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_root / path
