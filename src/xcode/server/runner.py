"""Web 运行控制器。

服务端持有唯一的 XcodeApp 实例（单一会话）。所有连接的 WebSocket 客户端
共享同一事件流：任何一端提交的回合都会被广播给所有浏览器端。

审批桥接：权限引擎在工具线程中同步回调；控制器把请求推送到浏览器端并阻塞
等待回答，浏览器通过 approval 消息回填结果。
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Callable
from typing import Any

from xcode.agent.types import ApprovalRequest
from xcode.coding_agent.app import XcodeApp
from xcode.coding_agent.execution_modes import ExecutionMode
from xcode.harness.security import HITLResult
from xcode.harness.security.approval import HITLScope

from .serialize import event_to_dict

# 浏览器端审批超时后按拒绝处理，避免工具线程永久挂起
_APPROVAL_TIMEOUT_SECONDS = 300.0

Sink = Callable[[dict[str, Any]], None]


class _PendingApproval:
    """阻塞中的审批请求。"""

    def __init__(self, request: ApprovalRequest) -> None:
        self.request_id = uuid.uuid4().hex[:12]
        self.request = request
        self.done = threading.Event()
        self.result: HITLResult | None = None


class WebRunHub:
    """并发安全的运行舞台：单活动回合 + 审批 + 广播。"""

    def __init__(self, app: XcodeApp) -> None:
        self._app = app
        self._sinks: list[Sink] = []
        self._run_task: asyncio.Task[None] | None = None
        self._pending: _PendingApproval | None = None
        self.set_app(app)

    @property
    def app(self) -> XcodeApp:
        return self._app

    def set_app(self, app: XcodeApp) -> None:
        """替换运行时应用（工作区切换）；重建审批桥接。"""
        self._app = app
        agent = getattr(app, "agent", None)
        if agent is not None:
            # 所有权限引擎的用户审批请求都走浏览器端 HITL 桥
            agent.user_approval_callback = self._approval_callback
        self._pending = None

    # ── 连接管理 ──

    def attach(self, sink: Sink) -> None:
        self._sinks.append(sink)
        self.broadcast({"type": "hello", "info": self.server_info()})

    def detach(self, sink: Sink) -> None:
        if sink in self._sinks:
            self._sinks.remove(sink)

    def broadcast(self, payload: dict[str, Any]) -> None:
        for sink in list(self._sinks):
            try:
                sink(payload)
            except Exception:
                # 单一连接失败不影响其他客户端
                continue

    def server_info(self) -> dict[str, Any]:
        app = self._app
        info: dict[str, Any] = {"session_id": app.session_store.session_id}
        try:
            info["model"] = app.get_model_info()
        except Exception:
            info["model"] = {}
        try:
            info["mcp"] = [
                {
                    "server": st.get("server"),
                    "status": st.get("status"),
                    "tools": _count_tools(st.get("tools")),
                }
                for st in app.mcp_status()
            ]
        except Exception:
            info["mcp"] = []
        info["busy"] = self.is_running
        return info

    # ── 运行控制 ──

    @property
    def is_running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    def submit(self, text: str, mode: ExecutionMode | None = None) -> None:
        if self.is_running:
            self.broadcast(
                {
                    "type": "run_error",
                    "message": "当前已有回合在运行，请等待完成或先取消。",
                }
            )
            return
        if not text.strip():
            return
        self.broadcast({"type": "user_message", "text": text, "mode": mode})
        self.broadcast({"type": "run_started"})
        self._run_task = asyncio.create_task(self._run(text, mode))

    def cancel(self) -> None:
        pending = self._pending
        if pending is not None:
            # 审批等待中：直接以拒绝关闭，让工具线程继续收敛
            self._resolve_pending(pending, "deny", "once", "")
            return
        agent = getattr(self._app, "agent", None)
        token = getattr(agent, "cancellation_token", None)
        if token is not None:
            token.cancel("cancelled by web user")
            self.broadcast({"type": "run_cancelled"})

    def resolve_approval(
        self,
        request_id: str,
        decision: str,
        scope: str,
        suggestion: str = "",
    ) -> bool:
        pending = self._pending
        if pending is None or pending.request_id != request_id:
            return False
        self._resolve_pending(pending, decision, scope, suggestion)
        return True

    def _resolve_pending(
        self,
        pending: _PendingApproval,
        decision: str,
        scope: str,
        suggestion: str,
    ) -> None:
        scope_value: HITLScope = (
            "once" if scope not in {"once", "session", "permanent"} else scope  # type: ignore[assignment]
        )
        if decision == "allow":
            pending.result = HITLResult(
                decision="allow",
                scope=scope_value,
                suggestion=suggestion,
            )
        else:
            pending.result = HITLResult(
                decision="deny", scope="once", suggestion=suggestion
            )
        pending.done.set()

    async def _run(self, text: str, mode: ExecutionMode | None) -> None:
        """在独立工作线程中消费同步事件流，避免 agent 预热阻塞事件循环。"""
        loop = asyncio.get_running_loop()

        def _consume() -> None:
            try:
                for event in self._app.ask_stream(text, mode=mode):
                    payload = {"type": "event", "event": event_to_dict(event)}
                    loop.call_soon_threadsafe(self.broadcast, payload)
            except Exception as exc:  # noqa: BLE001 - 广播给前端展示
                loop.call_soon_threadsafe(
                    self.broadcast,
                    {
                        "type": "run_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                )

        try:
            await loop.run_in_executor(None, _consume)
        except asyncio.CancelledError:
            self.broadcast({"type": "run_cancelled"})
        finally:
            self._pending = None
            self.broadcast({"type": "run_idle"})

    def new_session(self) -> str | None:
        """清空当前账本并重建空会话；运行中拒绝。"""
        if self.is_running:
            return None
        store = self._app.session_store
        store.clear()
        self._app.restore_session()
        agent = getattr(self._app, "agent", None)
        if agent is not None:
            agent.session_id = store.session_id
        new_id = store.session_id
        self.broadcast({"type": "session_reset", "session_id": new_id})
        return new_id

    def resume_session(self, session_id: str) -> str | None:
        """切换到历史会话并恢复 agent 状态；运行中拒绝。

        返回 None 表示正在运行；抛出 KeyError 表示会话不存在。
        """
        if self.is_running:
            return None
        store = self._app.session_store
        try:
            view = store.find_by_id(session_id)
        except Exception:  # noqa: BLE001 - 索引损坏时按未知处理
            view = None
        if view is None:
            raise KeyError(f"session not found: {session_id}")
        store.resume(view.path)
        self._app.restore_session()
        agent = getattr(self._app, "agent", None)
        if agent is not None:
            agent.session_id = store.session_id
        resumed_id = store.session_id
        self.broadcast({"type": "session_switched", "session_id": resumed_id})
        return resumed_id

    # ── 审批桥 ──

    def _approval_callback(self, request: ApprovalRequest) -> HITLResult:
        """权限引擎同步回调（运行于工具线程）。"""
        pending = _PendingApproval(request)
        self._pending = pending
        self.broadcast(
            {
                "type": "approval_request",
                "id": pending.request_id,
                "tool": to_tool_preview(request),
                "allowed_scopes": list(request.allowed_scopes),
                "reason": request.reason,
                "transcript": request.transcript,
                "working_directory": request.working_directory,
            }
        )
        pending.done.wait(timeout=_APPROVAL_TIMEOUT_SECONDS)
        self._pending = None
        return pending.result or HITLResult(
            decision="deny",
            scope="once",
            status="failed",
            rationale="Web 审批超时，按拒绝处理",
        )


def _count_tools(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def to_tool_preview(request: ApprovalRequest) -> dict[str, Any]:
    """提取审批请求中的工具摘要。"""

    def _value(obj: object, name: str, default: object = "") -> object:
        return getattr(obj, name, default)

    tool = request.tool
    return {
        "name": str(_value(tool, "name", "")),
        "description": str(_value(tool, "description", ""))[:500],
        "arguments": request.action_input,
    }
