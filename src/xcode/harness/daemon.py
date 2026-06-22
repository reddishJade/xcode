"""后台服务生命周期与心跳任务。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import threading
import time
from pathlib import Path
from typing import Any

from xcode.harness.mailbox import AgentMailbox
from xcode.harness.task_store import TaskStore

logger = logging.getLogger("xcode.harness.daemon")


@dataclass(frozen=True)
class DaemonHealth:
    """守护进程健康状态快照。"""

    running: bool
    restart_count: int
    last_heartbeat_at: float
    last_error: str = ""
    task_failures: dict[str, int] = field(default_factory=dict)


class HeartbeatDaemon:
    """会话级后台心跳守护进程。

    支持定期轮询（例如每 30 秒），运行注册的定时任务（Cron Tasks），
    并将结果写入 AgentMailbox 邮箱，提供主动的后台助手姿态。
    """

    def __init__(
        self,
        project_root: Path,
        mailbox: AgentMailbox,
        task_store: TaskStore,
        *,
        interval_seconds: int = 30,
        agent_id: str = "xcode_agent",
    ) -> None:
        self.project_root = project_root
        self.interval_seconds = interval_seconds
        self.agent_id = agent_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tasks: dict[str, Callable[[], list[dict[str, Any]] | None]] = {}
        self._callbacks: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._restart_count = 0
        self._last_heartbeat_at = 0.0
        self._last_error = ""
        self._task_failures: dict[str, int] = {}
        self.mailbox = mailbox
        self.task_store = task_store

        # 注册默认定时任务
        self.register_task("check_mailbox", self.check_mailbox)
        self.register_task("check_git_status", self.check_git_status)
        self.register_task("check_background_tasks", self.check_background_tasks)

    def register_task(
        self, name: str, func: Callable[[], list[dict[str, Any]] | None]
    ) -> None:
        """注册定时轮询任务。"""
        self._tasks[name] = func

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """注册守护事件回调。"""
        clean_type = event_type.strip() or "*"
        self._callbacks.setdefault(clean_type, []).append(callback)

    def start(self) -> None:
        """启动后台心跳线程。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("HeartbeatDaemon started with interval %ds", self.interval_seconds)

    def stop(self) -> None:
        """停止后台心跳线程。"""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        logger.info("HeartbeatDaemon stopped")

    def health_check(self) -> DaemonHealth:
        """返回当前守护进程健康状态。"""
        running = self._thread is not None and self._thread.is_alive()
        return DaemonHealth(
            running=running,
            restart_count=self._restart_count,
            last_heartbeat_at=self._last_heartbeat_at,
            last_error=self._last_error,
            task_failures=dict(self._task_failures),
        )

    def ensure_healthy(self) -> DaemonHealth:
        """如果后台线程异常退出，则自动重启。"""
        health = self.health_check()
        if health.running or self._stop_event.is_set():
            return health
        self._restart_count += 1
        self.start()
        return self.health_check()

    def _run_loop(self) -> None:
        """心跳轮询主循环。"""
        while not self._stop_event.is_set():
            self._run_loop_once()

            # 以小刻度休眠，以便快速响应 stop 信号
            sleep_step = 0.5
            elapsed = 0.0
            while elapsed < self.interval_seconds and not self._stop_event.is_set():
                time.sleep(sleep_step)
                elapsed += sleep_step

    def _run_loop_once(self) -> None:
        """执行一轮心跳任务。"""
        self._last_heartbeat_at = time.time()
        for name, func in list(self._tasks.items()):
            if self._stop_event.is_set():
                break
            try:
                events = func()
                if events:
                    for evt in events:
                        self._publish_event(evt)
            except Exception as e:
                self._last_error = f"{name}: {e}"
                self._task_failures[name] = self._task_failures.get(name, 0) + 1
                logger.error("Error running task %s: %s", name, e)
                self._publish_event(
                    {
                        "type": "daemon_task_error",
                        "payload": {
                            "task": name,
                            "error": str(e),
                            "failure_count": self._task_failures[name],
                        },
                    }
                )

    def _publish_event(self, event: dict[str, Any]) -> None:
        """发送守护事件到 mailbox 并通知回调。"""
        event_type = str(event.get("type", "system_alert"))
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"value": payload}
        normalized = {"type": event_type, "payload": payload}
        self.mailbox.send_message(
            sender_id="heartbeat_daemon",
            recipient_id=self.agent_id,
            type_name=event_type,
            payload=payload,
        )
        for callback in self._callbacks.get(event_type, []) + self._callbacks.get(
            "*", []
        ):
            try:
                callback(normalized)
            except Exception:
                logger.debug("daemon callback failed", exc_info=True)

    def check_mailbox(self) -> list[dict[str, Any]] | None:
        """检查未读邮件，如果有未读消息则产生通知事件。"""
        try:
            unread = self.mailbox.read_unread_messages(
                self.agent_id,
                exclude_senders={"heartbeat_daemon"},
            )
            # 顺带清理过期消息，避免 mailbox 文件无限膨胀
            try:
                self.mailbox.cleanup_expired_messages(self.agent_id)
            except Exception:
                logger.debug("mailbox cleanup failed", exc_info=True)
            if unread:
                return [
                    {
                        "type": "mailbox_summary",
                        "payload": {
                            "unread_count": len(unread),
                            "latest_sender": unread[-1].get("sender"),
                        },
                    }
                ]
        except Exception:
            logger.warning("mailbox check failed", exc_info=True)
        return None

    def check_git_status(self) -> list[dict[str, Any]] | None:
        """运行轻量级工作区脏检查。"""
        import subprocess

        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().splitlines()
                return [
                    {
                        "type": "git_dirty_alert",
                        "payload": {
                            "dirty_files_count": len(lines),
                            "summary": lines[0][:80],
                        },
                    }
                ]
        except Exception:
            logger.warning("git status check failed", exc_info=True)
        return None

    def check_background_tasks(self) -> list[dict[str, Any]] | None:
        """检查任务存储中的待办或已被领取的任务。"""
        try:
            tasks = self.task_store.list()
            pending = [t for t in tasks if t.status == "pending"]
            claimed = [t for t in tasks if t.status == "claimed"]
            if pending or claimed:
                return [
                    {
                        "type": "tasks_summary",
                        "payload": {
                            "pending_count": len(pending),
                            "claimed_count": len(claimed),
                        },
                    }
                ]
        except Exception:
            logger.warning("background task check failed", exc_info=True)
        return None
