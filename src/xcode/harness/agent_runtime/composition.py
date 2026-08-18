"""发布前冻结的 agent 能力与策略组合。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from uuid import uuid4

from xcode.ai.providers.base import ModelProvider
from xcode.agent.request import RequestAssembler
from xcode.agent.types import ToolSpec
from xcode.harness.config import AgentConfig
from xcode.harness.security import PermissionPolicy
from xcode.harness.security.permission_model import Rule

from .config import GateConfig, RuntimeContextProvider


@dataclass(frozen=True)
class AgentComposition:
    """一次发布后不可原地修改的 agent generation。"""

    generation_id: str
    primary_provider: ModelProvider
    fallback_provider: ModelProvider | None
    registry: tuple[ToolSpec, ...]
    config: AgentConfig
    gate: GateConfig
    request_assembler: RequestAssembler
    runtime_context_provider: RuntimeContextProvider | None = None

    @classmethod
    def create(
        cls,
        *,
        primary_provider: ModelProvider,
        fallback_provider: ModelProvider | None,
        registry: tuple[ToolSpec, ...],
        config: AgentConfig,
        gate: GateConfig,
        request_assembler: RequestAssembler,
        runtime_context_provider: RuntimeContextProvider | None,
    ) -> AgentComposition:
        """规范化所有集合并发布新的 generation。"""
        return cls(
            generation_id=uuid4().hex,
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            registry=_freeze_registry(registry),
            config=config.model_copy(deep=True),
            gate=_freeze_gate(gate),
            request_assembler=request_assembler,
            runtime_context_provider=runtime_context_provider,
        )

    def with_primary_provider(self, provider: ModelProvider) -> AgentComposition:
        """以新 provider 发布新 generation，不修改运行中的组合。"""
        return replace(
            self,
            generation_id=uuid4().hex,
            primary_provider=provider,
        )

    def with_permission_policy(
        self,
        policy: PermissionPolicy | None,
    ) -> AgentComposition:
        """以新静态权限策略发布新 generation。"""
        return replace(
            self,
            generation_id=uuid4().hex,
            gate=replace(
                self.gate,
                permission_policy=_freeze_permission_policy(policy),
            ),
        )


def _freeze_gate(gate: GateConfig) -> GateConfig:
    """冻结 GateConfig 内部的映射，避免发布后被旁路修改。"""
    return replace(
        gate,
        permission_policy=_freeze_permission_policy(gate.permission_policy),
        user_rulesets=_freeze_rulesets(gate.user_rulesets),
        default_mode_rulesets=_freeze_rulesets(gate.default_mode_rulesets),
        mode_fallbacks=MappingProxyType(dict(gate.mode_fallbacks)),
        shell_unresolved_policies=MappingProxyType(
            dict(gate.shell_unresolved_policies)
        ),
        tool_path_extractors=MappingProxyType(dict(gate.tool_path_extractors)),
    )


def _freeze_permission_policy(
    policy: PermissionPolicy | None,
) -> PermissionPolicy | None:
    if policy is None:
        return None
    return PermissionPolicy(
        rules=tuple(rule.model_copy(deep=True) for rule in policy.rules),
        global_default=policy.global_default,
    )


def _freeze_rulesets(
    rulesets: Mapping[str, tuple[Rule, ...]],
) -> Mapping[str, tuple[Rule, ...]]:
    return MappingProxyType(
        {
            name: tuple(rule.model_copy(deep=True) for rule in rules)
            for name, rules in rulesets.items()
        }
    )


def _freeze_registry(registry: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
    """复制并冻结模型可见 schema，阻止构建期对象污染已发布组合。"""
    return tuple(
        replace(
            tool,
            schema=(_freeze_mapping(tool.schema) if tool.schema is not None else None),
        )
        for tool in registry
    )


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze_json(item) for key, item in value.items()}
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value
