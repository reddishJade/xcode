"""会话级运行控制与忙时消息调度。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from threading import Event, Lock

from ...agent.agent import Agent
from ...agent.messages import AgentMessage, UserMessage
from ...agent.types import TextContent
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
    COLLECT = "collect"
    INTERRUPT = "interrupt"


class SubmitStatus(StrEnum):
    """运行时接收用户消息后的可观测结果。"""

    STEER_ACCEPTED = "steer_accepted"
    FOLLOW_UP_QUEUED = "followup_queued"
    INTERRUPT_REQUESTED = "interrupt_requested"
    NO_ACTIVE_RUN = "no_active_run"


@dataclass(frozen=True)
class SubmitOutcome:
    """忙时消息提交结果。"""

    status: SubmitStatus
    run_id: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status is not SubmitStatus.NO_ACTIVE_RUN


@dataclass
class _PendingTurn:
    """保持到达顺序的一次后续 turn。"""

    messages: list[UserMessage]
    collect: bool = False


class ActiveRunHandle:
    """包装单个 active run 的 identity、控制入口和状态。"""

    def __init__(
        self,
        run_id: str,
        session_id: str,
        agent: Agent,
        cancellation_token: CancellationToken,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self._agent = agent
        self._cancellation_token = cancellation_token
        self._state = ActiveRunState.RUNNING
        self._lock = Lock()
        self._finished = Event()

    def steer(self, message: AgentMessage) -> SubmitOutcome:
        """尝试把消息加入当前 run 的下一安全边界。"""
        with self._lock:
            if self._state is not ActiveRunState.RUNNING:
                return SubmitOutcome(SubmitStatus.NO_ACTIVE_RUN, self.run_id)
            accepted = self._agent.try_steer(message)
        return SubmitOutcome(
            SubmitStatus.STEER_ACCEPTED if accepted else SubmitStatus.NO_ACTIVE_RUN,
            self.run_id,
        )

    def interrupt(self, reason: str = "interrupted by user") -> SubmitOutcome:
        """请求协作式取消；run 真正退出前保持 CANCELLING。"""
        with self._lock:
            if self._state is ActiveRunState.FINISHED:
                return SubmitOutcome(SubmitStatus.NO_ACTIVE_RUN, self.run_id)
            if self._state is ActiveRunState.RUNNING:
                self._state = ActiveRunState.CANCELLING
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

    def finish(self) -> None:
        """标记 run 已完整提交并唤醒等待者。"""
        with self._lock:
            self._state = ActiveRunState.FINISHED
            self._finished.set()

    def wait_finished(self, timeout: float | None = None) -> bool:
        """同步等待 run 完整结束。"""
        return self._finished.wait(timeout)


class SessionRunController:
    """串行管理一个 session 的 active run 与 next-run 消息。"""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._active: ActiveRunHandle | None = None
        self._pending_turns: deque[_PendingTurn] = deque()
        self._run_index = 0
        self._lock = Lock()

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        with self._lock:
            self._session_id = value

    def begin_run(
        self,
        agent: Agent,
        cancellation_token: CancellationToken,
    ) -> ActiveRunHandle:
        """为新 run 建立唯一 identity，并拒绝同 session 重叠运行。"""
        with self._lock:
            if self._active is not None:
                raise RuntimeError("session already has an active run")
            self._run_index += 1
            handle = ActiveRunHandle(
                f"{self._session_id}:run:{self._run_index}",
                self._session_id,
                agent,
                cancellation_token,
            )
            self._active = handle
            return handle

    def active_run(self) -> ActiveRunHandle | None:
        """返回 active handle；FINISHED run 不再视为 active。"""
        with self._lock:
            return self._active

    def submit(
        self,
        message: UserMessage,
        mode: BusyMessageMode = BusyMessageMode.STEER,
    ) -> SubmitOutcome:
        """按照 busy policy 处理当前 session 的新用户消息。"""
        with self._lock:
            active = self._active
            if active is None:
                return SubmitOutcome(SubmitStatus.NO_ACTIVE_RUN)

            if mode is BusyMessageMode.FOLLOW_UP:
                self._pending_turns.append(_PendingTurn([message]))
                return SubmitOutcome(SubmitStatus.FOLLOW_UP_QUEUED, active.run_id)
            if mode is BusyMessageMode.COLLECT:
                if self._pending_turns and self._pending_turns[-1].collect:
                    self._pending_turns[-1].messages.append(message)
                else:
                    self._pending_turns.append(_PendingTurn([message], collect=True))
                return SubmitOutcome(SubmitStatus.FOLLOW_UP_QUEUED, active.run_id)
            if mode is BusyMessageMode.INTERRUPT:
                self._pending_turns.append(_PendingTurn([message]))

        if mode is BusyMessageMode.INTERRUPT:
            outcome = active.interrupt()
            if outcome.accepted:
                return outcome
            return SubmitOutcome(SubmitStatus.FOLLOW_UP_QUEUED, active.run_id)

        outcome = active.steer(message)
        if outcome.accepted:
            return outcome

        # run 已越过最后 steer 边界；消息可靠退化为 next-run follow-up。
        with self._lock:
            self._pending_turns.append(_PendingTurn([message]))
        return SubmitOutcome(SubmitStatus.FOLLOW_UP_QUEUED, active.run_id)

    def complete_run(
        self,
        handle: ActiveRunHandle,
        unconsumed_steers: list[AgentMessage] | None = None,
    ) -> None:
        """在 run 完整提交后清除 active，并保留未消费 steer。"""
        fallback = [m for m in unconsumed_steers or [] if isinstance(m, UserMessage)]
        with self._lock:
            if self._active is not handle:
                raise RuntimeError("cannot complete a non-active run")
            for message in reversed(fallback):
                self._pending_turns.appendleft(_PendingTurn([message]))
            handle.finish()
            self._active = None

    def take_follow_up(self) -> UserMessage | None:
        """取出下一条 next-run 消息；collect 消息会合并为一次 turn。"""
        with self._lock:
            if self._active is not None:
                return None
            if not self._pending_turns:
                return None
            pending = self._pending_turns.popleft()
        if len(pending.messages) == 1:
            return pending.messages[0]
        return UserMessage(
            content="\n\n".join(_message_text(msg) for msg in pending.messages)
        )

    def pending_count(self) -> int:
        """返回尚未启动的 next-run 消息数量。"""
        with self._lock:
            return len(self._pending_turns)


def _message_text(message: UserMessage) -> str:
    """把 collect 中的复合用户内容转换为可合并文本。"""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )
