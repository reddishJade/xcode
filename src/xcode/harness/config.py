from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, cast

ProviderTransport = Literal[
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "chatglm",
    "chatglm_chat",
    "deepseek_chat",
    "mimo_chat",
]
ExecutionMode = Literal["plan", "review", "act"]

PROFILE_MAIN = "main"
PROFILE_SUBAGENT = "subagent"
PROFILE_FALLBACK = "fallback"


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
class ToolsRuntimeConfig:
    network_commands: str = "ask"
    enabled_groups: tuple[str, ...] = ("core",)
    shell: str = "auto"


@dataclass(frozen=True)
class SkillsRuntimeConfig:
    auto_trigger: bool = True


@dataclass(frozen=True)
class PromptRuntimeConfig:
    modules: tuple[str, ...] = (
        "identity",
        "tool_discipline",
        "tools",
        "environment",
        "git_preflight",
        "cwd",
        "instructions",
        "notices",
    )


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


@dataclass
class HarnessConfig:
    """轻量 harness 配置。"""

    cwd: str = ""
    system_prompt: str = ""
    allow_dangerous_bash: bool = False
    bash_timeout: float = 120.0
    read_max_length: int = 10000
    max_tool_retries: int = 3
    session_id: str | None = None
    verbose: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ── 配置发现 ──


def discover_runtime_config(
    project_root: Path, explicit_path: Path | None = None
) -> XcodeRuntimeConfig:
    config_path = explicit_path or project_root / "xcode.config.json"
    return load_runtime_config(config_path)


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
            network_commands=bash.get("network_commands", "ask"),
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
    )


def to_agent_config(config: XcodeRuntimeConfig) -> AgentConfig:
    return config.agent


def resolve_config_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_root / path


def _optional_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _load_model_profiles(provider: dict) -> dict[str, ModelProfileRuntimeConfig]:
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
            transport=_normalize_transport(
                raw.get("transport", profiles[PROFILE_MAIN].transport)
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
            tool_stream=bool(raw.get("tool_stream", profiles[PROFILE_MAIN].tool_stream)),
            response_format=_optional_dict(
                raw.get("response_format", profiles[PROFILE_MAIN].response_format)
            ),
        )
    main = profiles[PROFILE_MAIN]
    profiles.setdefault(PROFILE_SUBAGENT, main)
    profiles.setdefault(PROFILE_FALLBACK, main)
    return profiles


def _normalize_transport(value: object) -> ProviderTransport:
    aliases = {
        "chat_completions": "openai_chat",
        "responses_stateful": "openai_responses",
        "chatglm": "chatglm_chat",
    }
    raw = str(value)
    return cast(ProviderTransport, aliases.get(raw, raw))


def _optional_dict(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None
