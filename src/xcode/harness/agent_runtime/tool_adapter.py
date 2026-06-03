"""ToolSpec → AgentTool 适配器。

将 harness 层的 ToolSpec 适配为 agent 层的 AgentTool protocol，
使 StructuredAgent 可以将 ToolSpec 注册表传给 agent 核心循环。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...agent.types import (
    AgentToolResult,
    CancellationSignal,
    TextContent,
    ToolExecutionMode,
    ToolUpdateCallback,
)
from ..skills import ToolSpec
from ..observability import PermissionPolicy, redact_text
from ..observability import HITLResult


class ToolSpecAdapter:
    """将 harness ToolSpec 适配为 agent AgentTool protocol。

    依赖方向：harness -> agent（正确）。
    此类在 harness/ 层，实现 agent/ 层定义的 protocol。
    """

    def __init__(
        self,
        spec: ToolSpec,
        *,
        approval_callback: Any | None = None,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        self._spec = spec
        self._approval_callback = approval_callback
        self._permission_policy = permission_policy

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def label(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._spec.schema or {"type": "object"}

    @property
    def execution_mode(self) -> ToolExecutionMode | None:
        return self._spec.execution_mode

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult[None]:
        # 权限检查
        if self._permission_policy:
            from ..skills import stringify_tool_input

            action_input = stringify_tool_input(params)
            decision = self._permission_policy.decide(self._spec.name, action_input)
            if decision == "deny":
                return AgentToolResult(
                    content=[TextContent(text=f"permission denied for tool: {self._spec.name}")]
                )
            if decision == "ask" and self._approval_callback:
                hitl: HITLResult = self._approval_callback(self._spec, params)
                if hitl.decision == "deny":
                    return AgentToolResult(
                        content=[TextContent(text=f"用户拒绝了 {self._spec.name}")]
                    )

        # risk_evaluator 检查
        if self._spec.risk_evaluator:
            risk_decision = self._spec.risk_evaluator(params)
            if risk_decision == "deny":
                return AgentToolResult(
                    content=[TextContent(text=f"permission denied for tool: {self._spec.name}")]
                )
            if risk_decision == "ask" and self._approval_callback:
                hitl = self._approval_callback(self._spec, params)
                if hitl.decision == "deny":
                    return AgentToolResult(
                        content=[TextContent(text=f"用户拒绝了 {self._spec.name}")]
                    )

        # 执行 handler（同步 → 异步）
        content = await asyncio.to_thread(self._spec.handler, params)
        return AgentToolResult(content=[TextContent(text=redact_text(content))])


def adapt_tool_specs(
    specs: tuple[ToolSpec, ...],
    *,
    approval_callback: Any | None = None,
    permission_policy: PermissionPolicy | None = None,
) -> list[ToolSpecAdapter]:
    """批量将 ToolSpec 适配为 AgentTool。"""
    return [
        ToolSpecAdapter(
            spec,
            approval_callback=approval_callback,
            permission_policy=permission_policy,
        )
        for spec in specs
    ]
