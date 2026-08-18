"""会话级运行控制与 durable inbox 调度。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Event, Lock
from uuid import uuid4

from ...agent.messages import AgentMessage, UserMessage
from ..session.inbox import InboxLane, InboxSource, SessionInbox
from .cancellation import CancellationToken


class ActiveRunState(StrEnum):
    """当前 run 的协作式生命周期状态。"""

    RUNNING = "running"
    CANCELLING = "cancelling"
    FINISHING = "finishing"
    FINISHED = "finished"


class BusyMessageMode(StrEnum):
    """会话忙碌时的新消息处理策略。"""

    STEER = "steer"
    FOLLOW_UP = "followup"
    INTERRUPT = "interrupt"


class SubmitStatus(StrEnum):
    """输入被调度后的可观测结果。"""

    STEER_ACCEPTED = "steer_accepted"
    FOLLOW_UP_QUEUED = "followup_queued"
    INJECT_QUEUED = "inject_queued"
    INTERRUPT_REQUESTED = "interrupt_requested"


@dataclass(frozen=True)
class SubmitOutcome:
    """输入提交结果。"""

    status: SubmitStatus
    run_id: str | None = None
    wake_required: bool = False


class ActiveRunHandle:
    """包装单个 active run 的 identity、输入边界和取消状态。"""

    def __init__(
        self,
        run_id: str,
        session_id: str,
        inbox: SessionInbox,
        cancellation_token: CancellationToken,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self._inbox = inbox
        self._cancellation_token = cancellation_token
        self._state = ActiveRunState.RUNNING
        self._accepting_step_input = True
        self._lock = Lock()
        self._finished = Event()

    def steer(
        self,
        message: AgentMessage,
        *,
        source: InboxSource = "user",
        display_text: str | None = None,
    ) -> SubmitOutcome | None:
        """在 run 仍有消费边界时把消息写入 next-step lane。"""
        with self._lock:
            if (
                self._state is not ActiveRunState.RUNNING
                or not self._accepting_step_input
            ):
                return None
            self._inbox.insert(
                message,
                InboxLane.NEXT_STEP,
                source=source,
                display_text=display_text,
                wake=False,
            )
        return SubmitOutcome(SubmitStatus.STEER_ACCEPTED, self.run_id)

    def claim_step_input(self) -> list[AgentMessage]:
        """在模型调用边界 claim 当前 next-step 输入。"""
        return self._inbox.claim_next_step(self.run_id)

    def finish_step_input(self) -> list[AgentMessage]:
        """关闭入口并原子 claim 生成期间最后到达的输入。"""
        with self._lock:
            self._accepting_step_input = False
            return self._inbox.claim_next_step(self.run_id)

    def reopen_step_input(self) -> None:
        """末轮输入触发新 step 后重新开放入口。"""
        with self._lock:
            if self._state is ActiveRunState.RUNNING:
                self._accepting_step_input = True

    def interrupt(self, reason: str = "interrupted by user") -> SubmitOutcome | None:
        """请求协作式取消；run 真正退出前保持 CANCELLING。"""
        with self._lock:
            if self._state is not ActiveRunState.RUNNING:
                return None
            self._state = ActiveRunState.CANCELLING
            self._accepting_step_input = False
            self._cancellation_token.cancel(reason)
        return SubmitOutcome(SubmitStatus.INTERRUPT_REQUESTED, self.run_id)

    def state(self) -> ActiveRunState:
        """返回当前生命周期状态快照。"""
        with self._lock:
            return self._state

    def begin_finishing(self) -> None:
        """标记模型循环已退出，正在提交 history/hooks/result。"""
        with self._lock:
            if self._state is not ActiveRunState.FINISHED:
                self._state = ActiveRunState.FINISHING
                self._accepting_step_input = False

    def finish(self) -> None:
        """标记 run 已完整提交并唤醒等待者。"""
        with self._lock:
            self._state = ActiveRunState.FINISHED
            self._accepting_step_input = False
            self._finished.set()

    def wait_finished(self, timeout: float | None = None) -> bool:
        """同步等待 run 完整结束。"""
        return self._finished.wait(timeout)


class SessionRunController:
    """串行管理一个 session 的 active run 与 durable inbox。"""

    def __init__(self, inbox: SessionInbox) -> None:
        self._inbox = inbox
        self._active: ActiveRunHandle | None = None
        self._lock = Lock()

    @property
    def session_id(self) -> str:
        return self._inbox.session_id

    def reload(self) -> None:
        """切换 session 后重建 pending input 投影。"""
        with self._lock:
            if self._active is not None:
                raise RuntimeError("cannot reload inbox during an active run")
            self._inbox.reload()

    def begin_run(self, cancellation_token: CancellationToken) -> ActiveRunHandle:
        """为新 run 建立唯一 identity，并拒绝同 session 重叠运行。"""
        with self._lock:
            if self._active is not None:
                raise RuntimeError("session already has an active run")
            handle = ActiveRunHandle(
                uuid4().hex,
                self.session_id,
                self._inbox,
                cancellation_token,
            )
            self._active = handle
            return handle

    def claim_initial(self, handle: ActiveRunHandle) -> list[AgentMessage]:
        """claim 新 run 的 next-step 输入和一条 next-turn 输入。"""
        with self._lock:
            if self._active is not handle:
                raise RuntimeError("cannot claim input for a non-active run")
            return self._inbox.claim_initial(handle.run_id)

    def active_run(self) -> ActiveRunHandle | None:
        """返回当前 active handle。"""
        with self._lock:
            return self._active

    def submit(
        self,
        message: UserMessage,
        mode: BusyMessageMode = BusyMessageMode.STEER,
        *,
        display_text: str | None = None,
    ) -> SubmitOutcome:
        """按照 busy policy 处理用户输入。"""
        with self._lock:
            active = self._active

        if mode is BusyMessageMode.FOLLOW_UP:
            self._inbox.insert(
                message,
                InboxLane.NEXT_TURN,
                display_text=display_text,
                wake=True,
            )
            return SubmitOutcome(
                SubmitStatus.FOLLOW_UP_QUEUED,
                active.run_id if active else None,
                wake_required=active is None,
            )

        if mode is BusyMessageMode.INTERRUPT:
            self._inbox.insert(
                message,
                InboxLane.NEXT_TURN,
                display_text=display_text,
                wake=True,
            )
            if active is not None:
                outcome = active.interrupt()
                if outcome is not None:
                    return outcome
            return SubmitOutcome(SubmitStatus.FOLLOW_UP_QUEUED, wake_required=True)

        if active is not None:
            outcome = active.steer(message, display_text=display_text)
            if outcome is not None:
                return outcome

        self._inbox.insert(
            message,
            InboxLane.NEXT_STEP,
            display_text=display_text,
            wake=True,
        )
        return SubmitOutcome(
            SubmitStatus.INJECT_QUEUED,
            active.run_id if active else None,
            wake_required=active is None,
        )

    def inject_runtime(self, message: AgentMessage) -> SubmitOutcome:
        """把运行时生成的模型可见消息注入当前或下一 run。"""
        with self._lock:
            active = self._active
        if active is not None:
            outcome = active.steer(message, source="runtime")
            if outcome is not None:
                return outcome
        self._inbox.insert(
            message,
            InboxLane.NEXT_STEP,
            source="runtime",
            wake=True,
        )
        return SubmitOutcome(
            SubmitStatus.INJECT_QUEUED,
            active.run_id if active else None,
            wake_required=active is None,
        )

    def complete_run(self, handle: ActiveRunHandle) -> None:
        """在 run 完整提交后清除 active identity。"""
        with self._lock:
            if self._active is not handle:
                raise RuntimeError("cannot complete a non-active run")
            handle.finish()
            self._active = None

    def has_waking_input(self) -> bool:
        """返回是否存在应启动 run 的 pending input。"""
        return self._inbox.has_waking_input()

    def pending_count(self) -> int:
        """返回 durable inbox 中尚未 claim 的输入数。"""
        return self._inbox.pending_count()
