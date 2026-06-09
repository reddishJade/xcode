from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, cast

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


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 20
    execution_mode: ExecutionMode = "act"
    compact_threshold: int = 0
    compact_token_threshold: int = 0
    max_recent_messages: int = 10
    tool_workers: int = 4
    watchdog_repeated_tool_limit: int = 3


@dataclass(frozen=True)
class RequestHygieneConfig:
    """请求 hygiene 配置。

    控制发给模型的消息历史压缩策略，不影响磁盘/session 保存的完整历史。
    """

    enabled: bool = True
    max_tool_result_bytes: int = 8000
    max_tool_arg_length: int = 1000
    keep_head_lines: int = 50
    keep_tail_lines: int = 50


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    model_profiles: dict[str, ModelProfileRuntimeConfig] = field(
        default_factory=lambda: {
            PROFILE_MAIN: ModelProfileRuntimeConfig(),
            PROFILE_SUBAGENT: ModelProfileRuntimeConfig(),
            PROFILE_FALLBACK: ModelProfileRuntimeConfig(),
        }
    )


@dataclass(frozen=True)
class SecurityRuntimeConfig:
    """三层权限模型配置。

    Layer 1: permission_mode (用户入口简化配置)
    Layer 2: 底层安全能力 (sandbox_mode, approval_policy, network_access, writable_roots, restricted_dirs)
    Layer 3: 工具规则 (deny_tools, ask_tools, allow_tools)

    优先级：deny > ask > allow

    permission_mode 映射规则：
    - strict: sandbox_mode=True, approval_policy=always, 所有工具默认需审批
    - normal: sandbox_mode=False, approval_policy=high_risk_only, 高风险工具需审批
    - permissive: sandbox_mode=False, approval_policy=never, 仅 deny_tools 阻断
    """

    # Layer 1: 简化模式
    permission_mode: PermissionMode = "normal"

    # Layer 2: 底层安全能力
    sandbox_mode: bool = False
    approval_policy: ApprovalPolicy = "high_risk_only"
    network_access: bool = True
    writable_roots: tuple[str, ...] = ()
    restricted_dirs: tuple[str, ...] = ()

    # Layer 3: 工具规则 (deny > ask > allow)
    deny_tools: tuple[str, ...] = ()
    ask_tools: tuple[str, ...] = ()
    allow_tools: tuple[str, ...] = ()

    def resolve_approval_policy(self) -> ApprovalPolicy:
        """根据 permission_mode 解析最终的 approval_policy。

        显式设置的 approval_policy 优先于 permission_mode 映射。
        """
        # permission_mode 映射表
        mode_to_policy: dict[PermissionMode, ApprovalPolicy] = {
            "strict": "always",
            "normal": "high_risk_only",
            "permissive": "never",
        }

        # 如果 approval_policy 是默认值，使用 permission_mode 映射
        if self.approval_policy == "high_risk_only":
            return mode_to_policy.get(self.permission_mode, "high_risk_only")

        return self.approval_policy

    def resolve_sandbox_mode(self) -> bool:
        """根据 permission_mode 解析最终的 sandbox_mode。"""
        if self.permission_mode == "strict":
            return True
        return self.sandbox_mode


@dataclass(frozen=True)
class ToolsRuntimeConfig:
    enabled_groups: tuple[str, ...] = ("core",)
    shell: str = "auto"


@dataclass(frozen=True)
class SkillsRuntimeConfig:
    auto_trigger: bool = True


@dataclass(frozen=True)
class PromptRuntimeConfig:
    modules: tuple[str, ...] = DEFAULT_PROMPT_MODULES


@dataclass(frozen=True)
class PathsRuntimeConfig:
    sessions_dir: Path | None = None
    skills_dir: Path | None = None


@dataclass(frozen=True)
class ObservabilityRuntimeConfig:
    audit_path: Path | None = None


@dataclass(frozen=True)
class DaemonRuntimeConfig:
    enabled: bool = False
    interval_seconds: int = 30


@dataclass(frozen=True)
class XcodeRuntimeConfig:
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


# ── 配置发现 ──


def discover_runtime_config(
    project_root: Path, explicit_path: Path | None = None
) -> XcodeRuntimeConfig:
    """发现并加载运行时配置，实现完整覆盖链。

    配置覆盖优先级（从高到低）：
    1. CLI 参数（由调用方传入）
    2. 环境变量（XCODE_* 前缀）
    3. .local/settings.json（本机私有覆盖）
    4. xcode.config.json（项目共享配置）
    5. ~/.xcode/settings.json（全局用户配置）
    6. 内置默认值
    """
    # 基础配置：全局用户配置 → 项目配置 → 本地覆盖
    global_config = _load_global_config()
    project_config_path = explicit_path or project_root / "xcode.config.json"
    project_config = load_runtime_config(project_config_path)
    local_config_path = project_root / ".local" / "settings.json"
    local_config = load_runtime_config(local_config_path)

    # 合并配置（project 覆盖 global，local 覆盖 project）
    merged = _merge_configs(global_config, project_config)
    merged = _merge_configs(merged, local_config)

    # 应用环境变量覆盖
    merged = _apply_env_overrides(merged)

    return merged


def _load_global_config() -> XcodeRuntimeConfig:
    """加载 ~/.xcode/settings.json 全局配置。"""
    home = Path.home()
    global_config_path = home / ".xcode" / "settings.json"
    return load_runtime_config(global_config_path)


def _merge_configs(
    base: XcodeRuntimeConfig, override: XcodeRuntimeConfig
) -> XcodeRuntimeConfig:
    """合并两个配置，override 非默认字段覆盖 base。

    实现字段级深度合并，保持 dataclass 不可变性。
    """
    from dataclasses import fields, replace, MISSING

    # 如果 override 完全是默认值，返回 base
    if override == XcodeRuntimeConfig():
        return base

    # 逐字段合并
    merged_fields = {}

    for config_field in fields(XcodeRuntimeConfig):
        base_value = getattr(base, config_field.name)
        override_value = getattr(override, config_field.name)

        # 获取默认值
        if config_field.default is not MISSING:
            default_value = config_field.default
        elif config_field.default_factory is not MISSING:
            default_value = config_field.default_factory()
        else:
            # 无默认值的字段，override 优先
            merged_fields[config_field.name] = override_value
            continue

        # 如果 override 是默认值，使用 base
        if override_value == default_value:
            merged_fields[config_field.name] = base_value
        else:
            merged_fields[config_field.name] = override_value

    return replace(base, **merged_fields)


def _apply_env_overrides(config: XcodeRuntimeConfig) -> XcodeRuntimeConfig:
    """应用环境变量覆盖。

    支持的环境变量：
    - XCODE_PERMISSION_MODE: strict/normal/permissive
    - XCODE_SANDBOX_MODE: true/false
    - XCODE_APPROVAL_POLICY: always/high_risk_only/never
    """
    import os

    permission_mode = os.getenv("XCODE_PERMISSION_MODE")
    sandbox_mode = os.getenv("XCODE_SANDBOX_MODE")
    approval_policy = os.getenv("XCODE_APPROVAL_POLICY")

    security = config.security

    if permission_mode in ("strict", "normal", "permissive"):
        security = dataclass_replace(
            security, permission_mode=cast(PermissionMode, permission_mode)
        )

    if sandbox_mode in ("true", "false"):
        security = dataclass_replace(security, sandbox_mode=(sandbox_mode == "true"))

    if approval_policy in ("always", "high_risk_only", "never"):
        security = dataclass_replace(
            security, approval_policy=cast(ApprovalPolicy, approval_policy)
        )

    if security != config.security:
        config = dataclass_replace(config, security=security)

    return config


def dataclass_replace(obj: Any, **changes: Any) -> Any:
    """替换 dataclass 字段值，保持不可变性。"""
    from dataclasses import replace

    return replace(obj, **changes)


def load_runtime_config(path: Path | None) -> XcodeRuntimeConfig:
    if path is None or not path.exists():
        return XcodeRuntimeConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    provider = data.get("provider", {})
    tools = data.get("tools", {})
    bash = tools.get("bash", {})
    skills = data.get("skills", {})
    prompt = data.get("prompt", {})
    agent = data.get("agent", {})
    paths = data.get("paths", {})
    observability = data.get("observability", {})
    daemon = data.get("daemon", {})
    request_hygiene = data.get("request_hygiene", {})
    security = data.get("security", {})
    return XcodeRuntimeConfig(
        provider=ProviderRuntimeConfig(
            model_profiles=_load_model_profiles(provider),
        ),
        agent=AgentConfig(
            max_steps=int(agent.get("max_steps", 20)),
            compact_threshold=int(agent.get("compact_threshold", 0)),
            compact_token_threshold=int(agent.get("compact_token_threshold", 0)),
            max_recent_messages=int(agent.get("max_recent_messages", 10)),
            tool_workers=int(agent.get("tool_workers", 4)),
            watchdog_repeated_tool_limit=int(
                agent.get("watchdog_repeated_tool_limit", 3)
            ),
        ),
        tools=ToolsRuntimeConfig(
            enabled_groups=tuple(tools.get("enabled_groups", ("core",))),
            shell=str(bash.get("shell", "auto")),
        ),
        skills=SkillsRuntimeConfig(
            auto_trigger=bool(skills.get("auto_trigger", True)),
        ),
        prompt=PromptRuntimeConfig(
            modules=tuple(prompt.get("modules", PromptRuntimeConfig().modules)),
        ),
        paths=PathsRuntimeConfig(
            sessions_dir=_optional_path(paths.get("sessions_dir")),
            skills_dir=_optional_path(paths.get("skills_dir")),
        ),
        observability=ObservabilityRuntimeConfig(
            audit_path=_optional_path(observability.get("audit_path")),
        ),
        daemon=DaemonRuntimeConfig(
            enabled=bool(daemon.get("enabled", False)),
            interval_seconds=int(daemon.get("interval_seconds", 30)),
        ),
        request_hygiene=RequestHygieneConfig(
            enabled=bool(request_hygiene.get("enabled", True)),
            max_tool_result_bytes=int(
                request_hygiene.get("max_tool_result_bytes", 8000)
            ),
            max_tool_arg_length=int(request_hygiene.get("max_tool_arg_length", 1000)),
            keep_head_lines=int(request_hygiene.get("keep_head_lines", 50)),
            keep_tail_lines=int(request_hygiene.get("keep_tail_lines", 50)),
        ),
        security=SecurityRuntimeConfig(
            permission_mode=_load_permission_mode(security.get("permission_mode")),
            sandbox_mode=bool(security.get("sandbox_mode", False)),
            approval_policy=_load_approval_policy(security.get("approval_policy")),
            network_access=bool(security.get("network_access", True)),
            writable_roots=tuple(security.get("writable_roots", ())),
            restricted_dirs=tuple(security.get("restricted_dirs", ())),
            deny_tools=tuple(security.get("deny_tools", ())),
            ask_tools=tuple(security.get("ask_tools", ())),
            allow_tools=tuple(security.get("allow_tools", ())),
        ),
    )


def resolve_config_path(project_root: Path, path: Path | None) -> Path | None:
    """解析配置路径，相对路径转为基于项目根的绝对路径。"""
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_root / path


def _optional_path(value: object) -> Path | None:
    """将配置值转为 Path 对象，空值返回 None。"""
    if value in (None, ""):
        return None
    return Path(str(value))


def _load_model_profiles(provider: dict) -> dict[str, ModelProfileRuntimeConfig]:
    """从配置文件加载 model profiles，支持字符串简写和完整配置。

    字符串简写设计原因：
    用户只需配置 {"subagent": "deepseek-v4-flash"} 即可快速切换模型，
    其余参数（transport/base_url/api_key）自动继承 main profile。
    这避免了为每个 profile 重复完整配置的冗余。

    三个固定 profile 的设计原因：
    - main: 主循环默认模型
    - subagent: 子任务模型（成本优化，未配置时继承 main）
    - fallback: 降级模型（可用性保障，未配置时继承 main）

    这三个 profile 是运行时约定，确保代码可以安全引用它们而不需要每次检查存在性。
    """
    profiles = {PROFILE_MAIN: ModelProfileRuntimeConfig()}
    raw_profiles = provider.get("model_profiles", {})
    if not isinstance(raw_profiles, dict):
        return profiles
    for name, raw in raw_profiles.items():
        profile_name = str(name).strip()
        if not profile_name:
            continue
        if isinstance(raw, str):
            profiles[profile_name] = ModelProfileRuntimeConfig(
                transport=profiles[PROFILE_MAIN].transport,
                chat_model=raw,
                base_url=profiles[PROFILE_MAIN].base_url,
                api_key=profiles[PROFILE_MAIN].api_key,
            )
            continue
        if not isinstance(raw, dict):
            continue
        profiles[profile_name] = ModelProfileRuntimeConfig(
            transport=cast(
                ProviderTransport,
                raw.get("transport", profiles[PROFILE_MAIN].transport),
            ),
            chat_model=raw.get("chat_model", profiles[PROFILE_MAIN].chat_model),
            base_url=raw.get("base_url", profiles[PROFILE_MAIN].base_url),
            api_key=raw.get("api_key", profiles[PROFILE_MAIN].api_key),
            thinking=bool(raw.get("thinking", profiles[PROFILE_MAIN].thinking)),
            reasoning_effort=raw.get(
                "reasoning_effort", profiles[PROFILE_MAIN].reasoning_effort
            ),
            clear_thinking=bool(
                raw.get("clear_thinking", profiles[PROFILE_MAIN].clear_thinking)
            ),
            tool_stream=bool(
                raw.get("tool_stream", profiles[PROFILE_MAIN].tool_stream)
            ),
            response_format=_optional_dict(
                raw.get("response_format", profiles[PROFILE_MAIN].response_format)
            ),
        )
    main = profiles[PROFILE_MAIN]
    profiles.setdefault(PROFILE_SUBAGENT, main)
    profiles.setdefault(PROFILE_FALLBACK, main)
    return profiles


def _optional_dict(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _load_permission_mode(value: object) -> PermissionMode:
    """加载 permission_mode，非法值默认为 normal。"""
    if value in ("strict", "normal", "permissive"):
        return cast(PermissionMode, value)
    return "normal"


def _load_approval_policy(value: object) -> ApprovalPolicy:
    """加载 approval_policy，非法值默认为 high_risk_only。"""
    if value in ("always", "high_risk_only", "never"):
        return cast(ApprovalPolicy, value)
    return "high_risk_only"
