"""以独立 durable session 运行和恢复本地子代理。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4

from xcode.ai.providers.base import ModelProvider
from xcode.agent.types import ToolSpec
from xcode.harness.observability import SignalHookManager
from xcode.harness.session.inbox import SessionInbox
from xcode.harness.session.recorder import SessionRecorder
from xcode.harness.session.subagent_runs import (
    SubagentDescriptor,
    SubagentActivationEvent,
    SubagentMode,
    SubagentRunEvent,
    SubagentRunStatus,
)
from xcode.harness.session.surface import project_session_surface
from xcode.harness.session.tree_store import TreeSessionRepo

from .cancellation import CancellationToken
from .composition import AgentComposition
from .config import AgentRuntimeConfig, GateRuntimeConfig
from .harness import AgentHarness
from .tool_gate import ToolGate


@dataclass(frozen=True)
class SubagentTaskResult:
    """一次 child turn 的结构化结果。"""

    child_session_id: str
    run_id: str
    status: SubagentRunStatus
    answer: str = ""
    error: str = ""


@dataclass
class _ChildActivation:
    activation_id: str
    descriptor: SubagentDescriptor
    harness: AgentHarness
    recorder: SessionRecorder
    cancellation_token: CancellationToken
    turn_lock: Lock
    active_turns: int = 0


class _ChildCancellationToken(CancellationToken):
    """保留 child 局部 interrupt，同时观察父 run 的取消状态。"""

    def __init__(self, parent: CancellationToken | None) -> None:
        super().__init__()
        self._parent = parent

    def is_cancelled(self) -> bool:
        return super().is_cancelled() or bool(
            self._parent is not None and self._parent.is_cancelled()
        )

    @property
    def reason(self) -> str:
        if super().is_cancelled() or self._parent is None:
            return super().reason
        return self._parent.reason


class SubagentSessionManager:
    """统一拥有 child session 创建、continuation 和冷恢复。"""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        coding_tools: tuple[ToolSpec, ...],
        research_tools: tuple[ToolSpec, ...],
        system_prompts: Mapping[str, str],
        parent_store: TreeSessionRepo,
    ) -> None:
        self._provider = provider
        self._coding_tools = coding_tools
        self._research_tools = research_tools
        self._system_prompts = dict(system_prompts)
        self._parent_store = parent_store
        self._composition_provider: Callable[[], AgentComposition] | None = None
        self._parent_gate: ToolGate | None = None
        self._parent_cancellation: CancellationToken | None = None
        self._activations: dict[str, _ChildActivation] = {}
        self._owned_children: dict[str, set[str]] = {}
        self._lock = RLock()
        self._closed = False
        self._closing = False

    def bind_parent(
        self,
        composition_provider: Callable[[], AgentComposition],
        permission_gate: ToolGate,
        cancellation_token: CancellationToken,
    ) -> None:
        """在父 harness 发布后绑定其 composition 与权限门控。"""
        self._composition_provider = composition_provider
        self._parent_gate = permission_gate
        self._parent_cancellation = cancellation_token

    def replace_provider(self, provider: ModelProvider) -> None:
        """让新 child activation 使用新的 provider。"""
        with self._lock:
            self._provider = provider

    async def execute(
        self,
        *,
        description: str,
        prompt: str,
        subagent_type: str,
        mode: SubagentMode,
        run_id: str,
        batch_id: str,
        task_index: int,
        on_update: Callable[[str], None] | None = None,
    ) -> SubagentTaskResult:
        """创建 child session 并执行其首个 turn。"""
        self._require_open()
        parent_recorder = self._current_parent_recorder()
        child_store = parent_recorder.store.spawn_child(
            title=description,
            summary=f"Subagent task: {description}",
        )
        activation = self._materialize(
            child_store,
            parent_session_id=parent_recorder.store.session_id,
            description=description,
            subagent_type=subagent_type,
            mode=mode,
        )
        activation.recorder.record_subagent_descriptor(activation.descriptor)
        self._register_activation(activation)
        try:
            return await self._run_turn(
                activation,
                parent_recorder,
                prompt=prompt,
                run_id=run_id,
                batch_id=batch_id,
                task_index=task_index,
                on_update=on_update,
            )
        finally:
            if mode == "one_shot":
                self._release_activation(activation, "one-shot settled")

    async def send(
        self,
        child_session_id: str,
        prompt: str,
        *,
        on_update: Callable[[str], None] | None = None,
    ) -> SubagentTaskResult:
        """向 direct continuable child 的 durable inbox 提交下一 turn。"""
        self._require_open()
        parent_recorder = self._current_parent_recorder()
        activation = self._activation_for(
            child_session_id,
            parent_recorder.store.session_id,
        )
        return await self._run_turn(
            activation,
            parent_recorder,
            prompt=prompt,
            run_id=uuid4().hex,
            batch_id=uuid4().hex,
            task_index=1,
            on_update=on_update,
        )

    def interrupt(self, child_session_id: str) -> bool:
        """中断 direct child 的当前 turn，但保留 session 与未 claim inbox。"""
        activation = self._authorized_live_activation(child_session_id)
        return activation.harness.interrupt("interrupted by parent session")

    def release(self, child_session_id: str) -> None:
        """释放 idle activation；durable child session 保持可冷恢复。"""
        activation = self._authorized_live_activation(child_session_id)
        if activation.active_turns or activation.harness.active_run() is not None:
            raise RuntimeError("cannot release an active child turn")
        self._release_activation(activation, "released by parent session")

    def close(self, timeout_seconds: float = 5.0) -> None:
        """父 app 关闭时先 cancel children，再有界等待并释放 activation。"""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            activations = tuple(self._activations.values())
        for activation in activations:
            activation.harness.interrupt("parent runtime is closing")
            activation.cancellation_token.cancel("parent runtime is closing")
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            if all(
                item.active_turns == 0 and item.harness.active_run() is None
                for item in activations
            ):
                break
            time.sleep(0.01)
        if any(
            item.active_turns > 0 or item.harness.active_run() is not None
            for item in activations
        ):
            with self._lock:
                self._closing = False
            raise RuntimeError("child activations did not settle before close timeout")
        for activation in reversed(activations):
            self._release_activation(activation, "parent runtime closed")
        with self._lock:
            self._closed = True
            self._closing = False

    def list_children(self) -> tuple[SubagentDescriptor, ...]:
        """不激活 child，直接从持久化 descriptor 枚举直接子会话。"""
        parent_id = self._parent_store.session_id
        descriptors: list[SubagentDescriptor] = []
        for info in self._parent_store.list_infos(limit=None):
            if info.parent_id != parent_id:
                continue
            descriptor = _read_descriptor(self._repo_at(info.path))
            if descriptor is not None:
                descriptors.append(descriptor)
        return tuple(descriptors)

    def _activation_for(
        self,
        child_session_id: str,
        parent_session_id: str,
    ) -> _ChildActivation:
        with self._lock:
            live = self._activations.get(child_session_id)
        if live is not None:
            if live.descriptor.parent_session_id != parent_session_id:
                raise PermissionError("child does not belong to the current session")
            return live
        info = self._parent_store.find_by_id(child_session_id)
        if info is None or info.parent_id != parent_session_id:
            raise PermissionError("child does not belong to the current session")
        child_store = self._repo_at(info.path)
        descriptor = _read_descriptor(child_store)
        if descriptor is None or descriptor.mode != "continuable":
            raise ValueError("child session is not continuable")
        activation = self._materialize(
            child_store,
            parent_session_id=descriptor.parent_session_id,
            description=descriptor.description,
            subagent_type=descriptor.subagent_type,
            mode=descriptor.mode,
            descriptor=descriptor,
        )
        with self._lock:
            current = self._activations.get(child_session_id)
            if current is None:
                self._register_activation(activation)
                current = activation
        return current

    def _materialize(
        self,
        child_store: TreeSessionRepo,
        *,
        parent_session_id: str,
        description: str,
        subagent_type: str,
        mode: SubagentMode,
        descriptor: SubagentDescriptor | None = None,
    ) -> _ChildActivation:
        composition_provider = self._composition_provider
        parent_gate = self._parent_gate
        if composition_provider is None or parent_gate is None:
            raise RuntimeError("subagent manager is not bound to a parent harness")
        parent_composition = composition_provider()
        with self._lock:
            provider = self._provider
        registry = self._registry_for(
            subagent_type,
            allowed_names=(descriptor.tool_names if descriptor is not None else None),
        )
        system_prompt = self._system_prompts.get(
            subagent_type,
            self._system_prompts["default"],
        )
        composition = AgentComposition.create(
            primary_provider=provider,
            fallback_provider=None,
            registry=registry,
            config=parent_composition.config,
            gate=parent_composition.gate,
            request_assembler=parent_composition.request_assembler,
            runtime_context_provider=lambda _question: [system_prompt],
        )
        recorder = SessionRecorder(child_store)
        hooks = SignalHookManager()
        hooks.register("before_provider_request", recorder.record_provider_request)
        child_gate = parent_gate.fork_for_subagent(
            child_store.session_id,
            hook_manager=hooks,
        )
        cancellation_token = _ChildCancellationToken(self._parent_cancellation)
        harness = AgentHarness(
            composition=composition,
            runtime=AgentRuntimeConfig(
                session_inbox=SessionInbox(child_store),
                gate=GateRuntimeConfig(
                    hook_manager=hooks,
                    session_id=child_store.session_id,
                ),
                gate_instance=child_gate,
                cancellation_token=cancellation_token,
                project_root=child_store.project_root,
            ),
        )
        surface = project_session_surface(child_store.build_branch())
        if surface.messages:
            harness.load_history(list(surface.messages))
        active_descriptor = descriptor or SubagentDescriptor(
            child_session_id=child_store.session_id,
            parent_session_id=parent_session_id,
            mode=mode,
            description=description,
            subagent_type=subagent_type,
            provider_model=str(provider.model),
            composition_id=composition.generation_id,
            tool_names=tuple(tool.name for tool in registry),
        )
        return _ChildActivation(
            activation_id=uuid4().hex,
            descriptor=active_descriptor,
            harness=harness,
            recorder=recorder,
            cancellation_token=cancellation_token,
            turn_lock=Lock(),
        )

    async def _run_turn(
        self,
        activation: _ChildActivation,
        parent_recorder: SessionRecorder,
        *,
        prompt: str,
        run_id: str,
        batch_id: str,
        task_index: int,
        on_update: Callable[[str], None] | None,
    ) -> SubagentTaskResult:
        descriptor = activation.descriptor
        with self._lock:
            activation.active_turns += 1
        acquired = False
        try:
            parent_recorder.record_subagent_run(
                _run_event(activation, run_id, batch_id, task_index, "started")
            )
            while not activation.turn_lock.acquire(blocking=False):
                await asyncio.sleep(0.01)
            acquired = True
            answer = ""
            async for event in activation.harness.arun_stream(prompt):
                activation.recorder.record_event(event)
                if event.type == "tool_update" and on_update is not None:
                    on_update(str(event.data.partial_result))
                if event.type == "final":
                    answer = event.data.answer
        except asyncio.CancelledError:
            parent_recorder.record_subagent_run(
                _run_event(activation, run_id, batch_id, task_index, "cancelled")
            )
            return SubagentTaskResult(
                child_session_id=descriptor.child_session_id,
                run_id=run_id,
                status="cancelled",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            parent_recorder.record_subagent_run(
                _run_event(
                    activation,
                    run_id,
                    batch_id,
                    task_index,
                    "failed",
                    error=error,
                )
            )
            return SubagentTaskResult(
                child_session_id=descriptor.child_session_id,
                run_id=run_id,
                status="failed",
                error=error,
            )
        finally:
            with self._lock:
                activation.active_turns -= 1
            if acquired:
                activation.turn_lock.release()
        parent_recorder.record_subagent_run(
            _run_event(
                activation,
                run_id,
                batch_id,
                task_index,
                "completed",
                summary=answer,
            )
        )
        return SubagentTaskResult(
            child_session_id=descriptor.child_session_id,
            run_id=run_id,
            status="completed",
            answer=answer,
        )

    def _registry_for(
        self,
        subagent_type: str,
        allowed_names: tuple[str, ...] | None = None,
    ) -> tuple[ToolSpec, ...]:
        registry = (
            self._coding_tools if subagent_type == "coding" else self._research_tools
        )
        if allowed_names is None:
            return registry
        allowed = frozenset(allowed_names)
        return tuple(tool for tool in registry if tool.name in allowed)

    def _register_activation(self, activation: _ChildActivation) -> None:
        descriptor = activation.descriptor
        with self._lock:
            self._activations[descriptor.child_session_id] = activation
            self._owned_children.setdefault(
                descriptor.parent_session_id,
                set(),
            ).add(descriptor.child_session_id)
        activation.recorder.record_subagent_activation(
            SubagentActivationEvent(
                child_session_id=descriptor.child_session_id,
                parent_session_id=descriptor.parent_session_id,
                activation_id=activation.activation_id,
                status="materialized",
            )
        )

    def _release_activation(
        self,
        activation: _ChildActivation,
        reason: str,
    ) -> None:
        descriptor = activation.descriptor
        with self._lock:
            current = self._activations.get(descriptor.child_session_id)
            if current is not activation:
                return
            self._activations.pop(descriptor.child_session_id, None)
            owned = self._owned_children.get(descriptor.parent_session_id)
            if owned is not None:
                owned.discard(descriptor.child_session_id)
                if not owned:
                    self._owned_children.pop(descriptor.parent_session_id, None)
        activation.recorder.record_subagent_activation(
            SubagentActivationEvent(
                child_session_id=descriptor.child_session_id,
                parent_session_id=descriptor.parent_session_id,
                activation_id=activation.activation_id,
                status="released",
                reason=reason,
            )
        )

    def _authorized_live_activation(
        self,
        child_session_id: str,
    ) -> _ChildActivation:
        parent_session_id = self._parent_store.session_id
        with self._lock:
            activation = self._activations.get(child_session_id)
        if (
            activation is None
            or activation.descriptor.parent_session_id != parent_session_id
        ):
            raise PermissionError("child is not a live direct child of this session")
        return activation

    def _require_open(self) -> None:
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("subagent manager is closed")

    def _current_parent_recorder(self) -> SessionRecorder:
        return SessionRecorder(self._repo_at(self._parent_store.current_path))

    def _repo_at(self, path: Path) -> TreeSessionRepo:
        repo = TreeSessionRepo(
            self._parent_store.sessions_dir,
            project_root=self._parent_store.project_root,
        )
        repo.resume(path)
        return repo


def _run_event(
    activation: _ChildActivation,
    run_id: str,
    batch_id: str,
    task_index: int,
    status: SubagentRunStatus,
    *,
    summary: str = "",
    error: str = "",
) -> SubagentRunEvent:
    descriptor = activation.descriptor
    return SubagentRunEvent(
        run_id=run_id,
        activation_id=activation.activation_id,
        child_session_id=descriptor.child_session_id,
        batch_id=batch_id,
        task_index=task_index,
        description=descriptor.description,
        subagent_type=descriptor.subagent_type,
        mode=descriptor.mode,
        status=status,
        summary=summary.strip()[:4000],
        error=error.strip()[:1000],
    )


def _read_descriptor(store: TreeSessionRepo) -> SubagentDescriptor | None:
    for entry in store.build_branch():
        if entry.type != "event" or not isinstance(entry.content, dict):
            continue
        if entry.content.get("type") != "subagent/descriptor":
            continue
        return SubagentDescriptor.model_validate(entry.content.get("data"))
    return None
