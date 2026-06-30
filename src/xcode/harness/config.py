from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
)

DirAccess = Literal["read", "write", "read_write"]

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
    "citations",
    "tools",
    "search_strategy",
    "environment",
    "cwd",
    "git_preflight",
    "contextual_retrieval",
    "notices",
)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: StrictInt = 20
    execution_mode: ExecutionMode = "act"
    compact_threshold: StrictInt = 0
    compact_token_threshold: StrictInt = 0
    max_recent_messages: StrictInt = 10
    reserve_tokens: StrictInt = 16384
    tool_workers: StrictInt = 4
    tool_timeout_seconds: StrictFloat | StrictInt = 120.0
    subagent_workers: StrictInt = 4
    watchdog_repeated_tool_limit: StrictInt = 3


class RequestHygieneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool = True
    max_tool_result_bytes: StrictInt = 8000
    max_tool_arg_length: StrictInt = 1000
    keep_head_lines: StrictInt = 50
    keep_tail_lines: StrictInt = 50


class ModelProfileRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transport: ProviderTransport = "openai_chat"
    chat_model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    thinking: StrictBool = True
    reasoning_effort: str | None = "high"
    clear_thinking: StrictBool = False
    tool_stream: StrictBool = True
    response_format: dict[str, Any] | None = None


class ProviderRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_profiles: dict[str, ModelProfileRuntimeConfig] = Field(
        default_factory=lambda: {
            PROFILE_MAIN: ModelProfileRuntimeConfig(),
            PROFILE_SUBAGENT: ModelProfileRuntimeConfig(),
            PROFILE_FALLBACK: ModelProfileRuntimeConfig(),
        }
    )


class SecurityExternalDirectory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    access: DirAccess = "read"


class SecurityRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    permission_mode: PermissionMode = "normal"
    sandbox_mode: StrictBool = False
    approval_policy: ApprovalPolicy = "never"
    network_access: StrictBool = True
    writable_roots: tuple[str, ...] = ()
    restricted_dirs: tuple[str, ...] = ()
    rules: tuple[dict[str, Any], ...] = ()
    global_default: str | None = None
    external_directories: tuple[SecurityExternalDirectory, ...] = ()

    def resolve_approval_policy(self) -> str:
        return self.approval_policy

    def resolve_sandbox_mode(self) -> bool:
        if self.permission_mode == "strict":
            return True
        return self.sandbox_mode


_INSTRUCTION_PRIORITIES: frozenset[str] = frozenset(
    {"critical", "high", "medium", "low"}
)


class FileInstructionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["file"] = "file"
    path: str
    priority: str | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        norm = v.replace("\\", "/")
        if norm.startswith("/"):
            raise ValueError(f"absolute path not allowed: {v}")
        if len(norm) >= 2 and norm[1] == ":":
            raise ValueError(f"absolute path not allowed: {v}")
        if norm.startswith("~"):
            raise ValueError(f"home-relative path not allowed: {v}")
        for segment in norm.split("/"):
            if segment == "..":
                raise ValueError(f"traversal path not allowed: {v}")
        return v

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: str | None) -> str | None:
        if v is not None and v.lower() not in _INSTRUCTION_PRIORITIES:
            raise ValueError(
                f"invalid priority '{v}'. Must be one of: critical, high, medium, low"
            )
        return v


class InlineInstructionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["inline"] = "inline"
    content: str = Field(min_length=1)
    priority: str | None = None

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: str | None) -> str | None:
        if v is not None and v.lower() not in _INSTRUCTION_PRIORITIES:
            raise ValueError(
                f"invalid priority '{v}'. Must be one of: critical, high, medium, low"
            )
        return v


InstructionSource = FileInstructionSource | InlineInstructionSource


class ToolsRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subagent_tool_allowlist: tuple[str, ...] = ()
    shell: str = "auto"


class SkillsRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trust_project_skills: bool = False


class PromptRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modules: tuple[str, ...] = DEFAULT_PROMPT_MODULES
    instructions: tuple[InstructionSource, ...] = ()


class PathsRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sessions_dir: Path | None = None
    skills_dir: Path | None = None
    progress_summary: Path = Path(".local/progress_summary.md")


class ObservabilityRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audit_path: Path | None = None


class ExternalHookRuntimeConfig(BaseModel):
    """单个受信任外部命令 hook 声明。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    event: HookEventName
    command: tuple[str, ...]
    matcher: str | None = None
    timeout: StrictFloat | StrictInt = 10.0
    enabled: StrictBool = True
    failure_policy: HookFailurePolicy = "warn"
    inherit_to_subagents: StrictBool = False
    source: str = ""

    @field_validator("command")
    @classmethod
    def _validate_command(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v or not v[0].strip():
            raise ValueError("command must be a non-empty argv array")
        return v

    @field_validator("matcher")
    @classmethod
    def _validate_matcher(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("matcher must be a non-empty string")
        return v

    @field_validator("timeout")
    @classmethod
    def _validate_timeout(cls, v: float | int) -> float | int:
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v


class HooksRuntimeConfig(BaseModel):
    """外部命令 hook 配置集合。"""

    model_config = ConfigDict(extra="forbid")
    entries: tuple[ExternalHookRuntimeConfig, ...] = ()


class DaemonRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool = False
    interval_seconds: StrictInt = 30


class ExperimentalRuntimeConfig(BaseModel):
    """默认关闭的实验性能力开关。"""

    model_config = ConfigDict(extra="forbid")
    tasks: StrictBool = False
    mailbox: StrictBool = False
    progress: StrictBool = False


class XcodeRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: ProviderRuntimeConfig = Field(default_factory=ProviderRuntimeConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsRuntimeConfig = Field(default_factory=ToolsRuntimeConfig)
    skills: SkillsRuntimeConfig = Field(default_factory=SkillsRuntimeConfig)
    prompt: PromptRuntimeConfig = Field(default_factory=PromptRuntimeConfig)
    paths: PathsRuntimeConfig = Field(default_factory=PathsRuntimeConfig)
    observability: ObservabilityRuntimeConfig = Field(
        default_factory=ObservabilityRuntimeConfig
    )
    hooks: HooksRuntimeConfig = Field(default_factory=HooksRuntimeConfig)
    daemon: DaemonRuntimeConfig = Field(default_factory=DaemonRuntimeConfig)
    experimental: ExperimentalRuntimeConfig = Field(
        default_factory=ExperimentalRuntimeConfig
    )
    request_hygiene: RequestHygieneConfig = Field(default_factory=RequestHygieneConfig)
    security: SecurityRuntimeConfig = Field(default_factory=SecurityRuntimeConfig)


# ── 序列化 / 反序列化 ──


def _config_from_dict(
    data: dict[str, Any],
    *,
    source_hint: Callable[[tuple[str | int, ...]], str | None] | None = None,
) -> XcodeRuntimeConfig:
    """从 dict 使用 Pydantic 模型校验构建 XcodeRuntimeConfig。

    Pydantic 自动校验类型、枚举值、嵌套结构，并拒绝未知字段。
    """
    try:
        return XcodeRuntimeConfig.model_validate(data)
    except ValidationError as e:
        lines = ["配置校验失败，请检查字段路径和来源配置层:"]
        for err in e.errors():
            path = ".".join(str(p) for p in err["loc"])
            line = f"  {path}: {err['msg']} (type={err['type']})"
            if source_hint is not None:
                hint = source_hint(tuple(err["loc"]))
                if hint:
                    line += f" [source: {hint}]"
            lines.append(line)
        raise ValueError("\n".join(lines)) from e


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
    env_raw, env_sources = _build_env_override_raw()
    merged = _deep_merge_raw(merged, env_raw)

    return _config_from_dict(
        merged,
        source_hint=lambda loc: _resolve_config_source_hint(
            loc,
            (
                ("environment", env_raw, env_sources),
                (f"local config {local_path}", local_raw, {}),
                (f"project config {project_path}", project_raw, {}),
                (f"global config {global_path}", global_raw, {}),
            ),
        ),
    )


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


def _build_env_override_raw() -> tuple[
    dict[str, Any], dict[tuple[str | int, ...], str]
]:
    raw: dict[str, Any] = {}
    sources: dict[tuple[str | int, ...], str] = {}
    security: dict[str, Any] = {}

    permission_mode = os.getenv("XCODE_PERMISSION_MODE")
    if permission_mode is not None:
        security["permission_mode"] = permission_mode
        sources[("security", "permission_mode")] = (
            "environment variable XCODE_PERMISSION_MODE"
        )

    sandbox_mode = os.getenv("XCODE_SANDBOX_MODE")
    if sandbox_mode is not None:
        if sandbox_mode == "true":
            security["sandbox_mode"] = True
        elif sandbox_mode == "false":
            security["sandbox_mode"] = False
        else:
            security["sandbox_mode"] = sandbox_mode
        sources[("security", "sandbox_mode")] = (
            "environment variable XCODE_SANDBOX_MODE"
        )

    approval_policy = os.getenv("XCODE_APPROVAL_POLICY")
    if approval_policy is not None:
        security["approval_policy"] = approval_policy
        sources[("security", "approval_policy")] = (
            "environment variable XCODE_APPROVAL_POLICY"
        )

    if security:
        raw["security"] = security
        sources[("security",)] = "environment overrides"
    return raw, sources


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
    env_raw, env_sources = _build_env_override_raw()
    raw = _deep_merge_raw(raw, env_raw)
    file_source = f"config {path}" if path is not None else "in-memory defaults"
    return _config_from_dict(
        raw,
        source_hint=lambda loc: _resolve_config_source_hint(
            loc,
            (
                ("environment", env_raw, env_sources),
                (file_source, raw, {}),
            ),
        ),
    )


def _resolve_config_source_hint(
    loc: tuple[str | int, ...],
    layers: tuple[tuple[str, dict[str, Any], dict[tuple[str | int, ...], str]], ...],
) -> str | None:
    for default_label, raw, path_labels in layers:
        if not raw:
            continue
        for size in range(len(loc), 0, -1):
            prefix = loc[:size]
            if not _path_exists(raw, prefix):
                continue
            for label_size in range(size, -1, -1):
                label = path_labels.get(prefix[:label_size])
                if label is not None:
                    return label
            return default_label
    return None


def _path_exists(data: object, path: tuple[str | int, ...]) -> bool:
    cursor = data
    for part in path:
        if isinstance(cursor, dict) and isinstance(part, str) and part in cursor:
            cursor = cursor[part]
            continue
        if (
            isinstance(cursor, list)
            and isinstance(part, int)
            and 0 <= part < len(cursor)
        ):
            cursor = cursor[part]
            continue
        return False
    return True


def resolve_config_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_root / path
