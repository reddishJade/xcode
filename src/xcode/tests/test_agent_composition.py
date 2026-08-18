"""Agent composition 发布边界测试。"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from xcode.agent.context import (
    ContextBlock,
    ContextBlockSource,
    ContextCollectionInput,
    ContextCollectorRegistry,
    ContextPriority,
)
from xcode.agent.request import DefaultRequestAssembler
from xcode.agent.types import ToolOutput, ToolSpec
from xcode.harness.agent_runtime.composition import AgentComposition
from xcode.harness.agent_runtime.config import GateConfig
from xcode.harness.config import AgentConfig


def _provider(name: str) -> Any:
    return cast(Any, object()) if name else None


def _tool(schema: dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        name="read",
        description="read",
        input_hint="path",
        handler=lambda _args, _update: ToolOutput("ok"),
        schema=schema,
    )


def _composition(
    provider: Any,
    *,
    registry: tuple[ToolSpec, ...] = (),
    gate: GateConfig | None = None,
) -> AgentComposition:
    return AgentComposition.create(
        primary_provider=provider,
        fallback_provider=None,
        registry=registry,
        config=AgentConfig(max_steps=7),
        gate=gate or GateConfig(),
        request_assembler=DefaultRequestAssembler(),
        runtime_context_provider=None,
    )


def test_composition_snapshots_and_freezes_published_inputs() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    rulesets: dict[str, tuple[Any, ...]] = {"act": ()}
    composition = _composition(
        _provider("first"),
        registry=(_tool(schema),),
        gate=GateConfig(user_rulesets=rulesets),
    )

    schema["properties"]["extra"] = {"type": "boolean"}
    rulesets["plan"] = ()

    published_schema = composition.registry[0].schema
    assert published_schema is not None
    assert "extra" not in published_schema["properties"]
    assert tuple(composition.gate.user_rulesets) == ("act",)
    with pytest.raises(TypeError):
        cast(Any, composition.gate.user_rulesets)["plan"] = ()
    with pytest.raises(TypeError):
        cast(Any, published_schema)["extra"] = {}
    with pytest.raises(ValidationError):
        composition.config.max_steps = 8


def test_replacement_publishes_new_generation_without_mutating_previous() -> None:
    first_provider = _provider("first")
    second_provider = _provider("second")
    first = _composition(first_provider)

    second = first.with_primary_provider(second_provider)

    assert second.generation_id != first.generation_id
    assert first.primary_provider is first_provider
    assert second.primary_provider is second_provider


class _Collector:
    def __init__(self, block_id: str) -> None:
        self.block_id = block_id

    def collect(self, _input: ContextCollectionInput) -> list[ContextBlock]:
        return [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.LOW,
                content=self.block_id,
                block_id=self.block_id,
            )
        ]


def test_frozen_collector_registry_ignores_later_registrations() -> None:
    collectors = ContextCollectorRegistry()
    collectors.register(_Collector("first"))
    frozen = collectors.freeze()

    collectors.register(_Collector("second"))

    assert [block.block_id for block in frozen.collect(ContextCollectionInput())] == [
        "first"
    ]
