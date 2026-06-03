from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from xcode.ai.events import ToolCall
from .tool_events import ToolResult
from .execution_modes import ActPolicy, ExecutionPolicy
from ..config import ExecutionMode
from ..observability import (
    AuditRecord,
    HookManager,
    HookRecord,
    PermissionPolicy,
    redact_text,
)
from ..skills import (
    ApprovalCallback,
    ToolExecutionResult,
    ToolSpec,
    stringify_tool_input,
    run_tool_result,
)

"""工具执行器。

主循环只提交 ToolCall 并接收 ToolResult；并发、HITL、hook、审计和结果格式
都在本模块内处理。
"""


class ExecutionCancelled(Exception):
    """工具执行被取消。

    Cancellation contract: `cancel` 触发后，所有未完成 task 会立即取消，已经
    完成但尚未返回给调用方的结果一律丢弃，并抛出 `ExecutionCancelled`。调用方
    不应接收部分结果。
    """


class ToolExecutor:
    def __init__(
        self,
        registry: tuple[ToolSpec, ...],
        *,
        tool_workers: int = 1,
        approval_callback: ApprovalCallback | None = None,
        permission_policy: PermissionPolicy | None = None,
        hook_manager: HookManager | None = None,
        audit_logger: Callable[[AuditRecord], None] | None = None,
        session_id: str = "local",
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.tool_map = {tool.name: tool for tool in registry}
        self.tool_workers = tool_workers
        self.approval_callback = approval_callback
        self.permission_policy = permission_policy
        self.hook_manager = hook_manager
        self.audit_logger = audit_logger
        self.session_id = session_id
        self.policy = policy or ActPolicy()

    async def execute(
        self,
        calls: list[ToolCall],
        *,
        cancel: asyncio.Event,
        mode: ExecutionMode = "act",
        active_tool_map: dict[str, ToolSpec] | None = None,
    ) -> list[ToolResult]:
        if cancel.is_set():
            raise ExecutionCancelled("tool execution cancelled")
        active = active_tool_map or self.tool_map
        if self.tool_workers <= 1 or not isinstance(self.policy, ActPolicy):
            return await self._execute_serial(calls, cancel, active, mode)
        return await self._execute_parallel(calls, cancel, active, mode)

    async def _execute_serial(
        self,
        calls: list[ToolCall],
        cancel: asyncio.Event,
        active_tool_map: dict[str, ToolSpec],
        mode: ExecutionMode,
    ) -> list[ToolResult]:
        results = []
        for call in calls:
            if cancel.is_set():
                raise ExecutionCancelled("tool execution cancelled")
            # 将同步工具执行卸载到线程池，避免阻塞事件循环
            result, elapsed = await asyncio.to_thread(
                self._timed_run_tool, call, active_tool_map, mode
            )
            results.append(_to_tool_result(call, result, elapsed))
        return results

    async def _execute_parallel(
        self,
        calls: list[ToolCall],
        cancel: asyncio.Event,
        active_tool_map: dict[str, ToolSpec],
        mode: ExecutionMode,
    ) -> list[ToolResult]:
        batches = partition_tool_calls(calls, active_tool_map)
        results: list[ToolResult] = []
        for batch in batches:
            if cancel.is_set():
                raise ExecutionCancelled("tool execution cancelled")
            if len(batch) == 1:
                result, elapsed = self._timed_run_tool(batch[0], active_tool_map, mode)
                results.append(_to_tool_result(batch[0], result, elapsed))
                continue
            workers = max(1, min(self.tool_workers, len(batch)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tasks = [
                    asyncio.wrap_future(
                        executor.submit(
                            self._timed_run_tool, call, active_tool_map, mode
                        )
                    )
                    for call in batch
                ]
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                if cancel.is_set():
                    for task in pending:
                        task.cancel()
                    raise ExecutionCancelled("tool execution cancelled")
                for task in pending:
                    done.add(task)
                for call, task in zip(batch, tasks, strict=True):
                    result, elapsed = await task
                    results.append(_to_tool_result(call, result, elapsed))
        return results

    def _timed_run_tool(
        self,
        call: ToolCall,
        active_tool_map: dict[str, ToolSpec],
        mode: ExecutionMode,
    ) -> tuple[ToolExecutionResult, float]:
        started = perf_counter()
        policy_result = self._policy_result(call)
        if policy_result is not None:
            self._audit_tool_use(call, policy_result)
            return policy_result, 0.0
        result = self._run_tool_with_hooks(call, active_tool_map, mode)
        elapsed = round((perf_counter() - started) * 1000, 3)
        self._audit_tool_use(call, result)
        return result, elapsed

    def _policy_result(self, call: ToolCall) -> ToolExecutionResult | None:
        decision = self.policy.check_call(call)
        if decision == "allow":
            return None
        tool = self.tool_map.get(call.name)
        if decision == "require_approval":
            if self.approval_callback is None or tool is None:
                return ToolExecutionResult(
                    "approval_required", f"工具需要授权：{call.name}"
                )
            hitl = self.approval_callback(tool, dict(call.input))
            if hitl.decision == "deny":
                return ToolExecutionResult(
                    "denied",
                    f"用户拒绝了 {call.name}。请改用只读检查或请用户手动执行。",
                    metadata={"user_decision": "deny", "approval_scope": hitl.scope},
                )
            return None
        return ToolExecutionResult("denied", f"permission denied for tool: {call.name}")

    def _run_tool_with_hooks(
        self,
        call: ToolCall,
        active_tool_map: dict[str, ToolSpec],
        mode: ExecutionMode,
    ) -> ToolExecutionResult:
        tool_input = dict(call.input)
        action_input = stringify_tool_input(tool_input)
        if call.name not in active_tool_map and call.name in self.tool_map:
            return ToolExecutionResult(
                "denied",
                f"tool unavailable in {mode} mode: {call.name}",
            )
        self._emit_hook(HookRecord("pre_tool", tool=call.name, input=action_input))
        try:
            result = run_tool_result(
                active_tool_map,
                call.name,
                tool_input,
                self.approval_callback,
                self.permission_policy,
            )
        except Exception as exc:
            error = str(exc)
            self._emit_hook(
                HookRecord("on_error", tool=call.name, input=action_input, error=error)
            )
            return ToolExecutionResult("error", f"tool error: {error}")
        if result.status == "error":
            error = result.content
            if result.metadata and "error" in result.metadata:
                error = str(result.metadata["error"])
            self._emit_hook(
                HookRecord("on_error", tool=call.name, input=action_input, error=error)
            )
        self._emit_hook(
            HookRecord(
                "post_tool",
                tool=call.name,
                input=action_input,
                output=result.content,
                metadata={"status": result.status, **(result.metadata or {})},
            )
        )
        return result

    def _audit_tool_use(self, call: ToolCall, result: ToolExecutionResult) -> None:
        if self.audit_logger is None:
            return
        tool = self.tool_map.get(call.name)
        static_risk = tool.risk if tool else "unknown"
        tool_input = dict(call.input)
        action_input = stringify_tool_input(tool_input)
        policy_decision = (
            self.permission_policy.decide(call.name, action_input)
            if self.permission_policy
            else None
        )
        if tool and tool.risk_evaluator:
            dynamic_decision = tool.risk_evaluator(tool_input)
        else:
            dynamic_decision = None
        meta = result.metadata or {}
        user_decision = meta.get("user_decision")
        approval_scope = meta.get("approval_scope")
        approved = result.status == "ok" or user_decision == "allow"
        self.audit_logger(
            AuditRecord(
                session_id=self.session_id,
                tool=call.name,
                static_risk=static_risk,
                dynamic_decision=dynamic_decision or static_risk,
                policy_decision=policy_decision,
                final_status=result.status,
                approved=approved,
                redacted_input=redact_text(action_input),
                redacted_output=redact_text(result.content),
                approval_scope=approval_scope,
                user_decision=user_decision,
            )
        )

    def _emit_hook(self, record: HookRecord) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(record)


def tool_result_message(result: ToolResult) -> dict[str, str]:
    return {
        "type": "tool_result",
        "tool_use_id": result.tool_call_id,
        "content": result.content,
        "status": result.status,
    }


def _to_tool_result(
    call: ToolCall, result: ToolExecutionResult, elapsed_ms: float | None = None
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        content=result.content,
        status=result.status,
        elapsed_ms=elapsed_ms,
    )


def partition_tool_calls(
    calls: list[ToolCall],
    active_tool_map: dict[str, ToolSpec],
) -> list[list[ToolCall]]:
    """分区并发算法：
    将工具调用列表划分为并发和串行的 batches。
    并发安全的工具（只读、并发安全且风险不为 high）会被分在同一个并发 batch 中，
    写操作/高风险/未知工具独自串行（每个工具一个单独的 batch），
    并且严格保持模型输出的原始顺序。
    """
    batches: list[list[ToolCall]] = []
    current: list[ToolCall] = []
    for call in calls:
        tool = active_tool_map.get(call.name)
        can_parallel = bool(
            tool and tool.read_only and tool.concurrency_safe and tool.risk != "high"
        )
        if can_parallel:
            current.append(call)
            continue
        if current:
            batches.append(current)
            current = []
        batches.append([call])
    if current:
        batches.append(current)
    return batches
