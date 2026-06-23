"""后台服务生命周期与心跳任务。

回声防护：daemon 自身产生的事件类型集合在 ``DAEMON_EVENT_TYPES`` 中声明，
``check_mailbox`` 通过 ``exclude_senders`` + ``exclude_types`` 组合过滤，
避免 daemon 事件被转发回 daemon 触发循环处理。所有 daemon 事件的 payload
携带 ``source="heartbeat_daemon"`` 元数据。

自愈恢复：``register_task(persistent=True)`` 将任务名持久化到
``.local/daemon_tasks.json``，进程重启后 ``__init__`` 恢复启用状态；
builtin 任务总是自动注册。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import filelock

from xcode.experimental.mailbox import AgentMailbox
from xcode.experimental.task_store import CLAIMED, PENDING, TaskStore
from xcode.experimental.worktree import WorktreeTaskRunner

logger = logging.getLogger("xcode.harness.daemon")

DAEMON_SOURCE = "heartbeat_daemon"
DAEMON_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "daemon_task_error",
        "mailbox_summary",
        "git_dirty_alert",
        "tasks_summary",
        "worktree_prune_report",
    }
)
_BUILTIN_TASKS: tuple[str, ...] = (
    "check_mailbox",
    "check_git_status",
    "check_background_tasks",
    "check_worktree_prune",
)


@dataclass(frozen=True)
class DaemonHealth:
    """守护进程健康状态快照。"""

    running: bool
    restart_count: int
    last_heartbeat_at: float
    last_error: str = ""
    task_failures: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DaemonTaskInfo:
    """单个已注册任务的信息。"""

    name: str
    registered: bool
    persistent: bool
    builtin: bool


class HeartbeatDaemon:
    """会话级后台心跳守护进程。"""

    def __init__(
        self,
        project_root: Path,
        mailbox: AgentMailbox | None = None,
        task_store: TaskStore | None = None,
        worktree_runner: WorktreeTaskRunner | None = None,
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
        self._persistent_names: set[str] = set()
        self._callbacks: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._restart_count = 0
        self._last_heartbeat_at = 0.0
        self._last_error = ""
        self._task_failures: dict[str, int] = {}
        self.mailbox = mailbox
        self.task_store = task_store
        self.worktree_runner = worktree_runner
        self._tasks_file = project_root / ".local" / "daemon_tasks.json"
        self._tasks_lock = filelock.FileLock(
            project_root / ".local" / ".daemon_tasks.lock", timeout=10.0
        )

        # 注册 builtin 定时任务
        if self.mailbox is not None:
            self.register_task("check_mailbox", self.check_mailbox)
        self.register_task("check_git_status", self.check_git_status)
        if self.task_store is not None:
            self.register_task("check_background_tasks", self.check_background_tasks)
        if self.worktree_runner is not None:
            self.register_task("check_worktree_prune", self.check_worktree_prune)
        # 恢复持久化的自定义任务名（callable 需外部重新注册）
        self._restore_persistent_tasks()

    def register_task(
        self,
        name: str,
        func: Callable[[], list[dict[str, Any]] | None],
        *,
        persistent: bool = False,
    ) -> None:
        """注册定时轮询任务。

        persistent=True 时将任务名写入 .local/daemon_tasks.json，
        进程重启后 __init__ 会恢复其启用状态（callable 需重新注册）。
        """
        self._tasks[name] = func
        if persistent:
            self._persistent_names.add(name)
            self._persist_task_names()
        elif name in self._persistent_names:
            # 重新注册同一名字仍保持持久化
            self._persistent_names.add(name)

    def unregister_task(self, name: str) -> bool:
        """移除已注册任务。返回是否移除成功。"""
        if name not in self._tasks:
            return False
        del self._tasks[name]
        if name in self._persistent_names:
            self._persistent_names.discard(name)
            self._persist_task_names()
        return True

    def list_daemon_tasks(self) -> list[DaemonTaskInfo]:
        """返回当前注册的任务清单。"""
        return [
            DaemonTaskInfo(
                name=name,
                registered=True,
                persistent=name in self._persistent_names,
                builtin=name in _BUILTIN_TASKS,
            )
            for name in self._tasks
        ]

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
        """如果后台线程异常退出，则自动重启并补齐 builtin 任务。"""
        health = self.health_check()
        if health.running or self._stop_event.is_set():
            return health
        # 补齐 builtin 任务（防止被 unregister 后重启丢失）
        builtins = [("check_git_status", self.check_git_status)]
        if self.mailbox is not None:
            builtins.append(("check_mailbox", self.check_mailbox))
        if self.task_store is not None:
            builtins.append(("check_background_tasks", self.check_background_tasks))
        if self.worktree_runner is not None:
            builtins.append(("check_worktree_prune", self.check_worktree_prune))
        for name, func in builtins:
            if name not in self._tasks:
                self._tasks[name] = func
        self._restart_count += 1
        self.start()
        return self.health_check()

    def _run_loop(self) -> None:
        """心跳轮询主循环；意外退出单轮时原地恢复。"""
        while not self._stop_event.is_set():
            try:
                self._run_loop_once()
            except Exception as exc:
                self._restart_count += 1
                self._last_error = f"daemon loop: {exc}"
                logger.exception("heartbeat loop failed; restarting")

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
        """发送守护事件到 mailbox 并通知回调。

        所有 daemon 事件 payload 自动添加 source 元数据（不覆盖已有值）。
        """
        event_type = str(event.get("type", "system_alert"))
        raw_payload = event.get("payload", {})
        payload: dict[str, Any] = (
            dict(raw_payload)
            if isinstance(raw_payload, dict)
            else {"value": raw_payload}
        )
        payload.setdefault("source", DAEMON_SOURCE)
        payload.setdefault("origin", self.agent_id)
        normalized = {"type": event_type, "payload": payload}
        if self.mailbox is not None:
            self.mailbox.send_message(
                sender_id=DAEMON_SOURCE,
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
        """检查未读邮件，过滤 daemon 自身事件，避免回声循环。"""
        if self.mailbox is None:
            return None
        try:
            unread = self.mailbox.read_unread_messages(
                self.agent_id,
                exclude_senders={DAEMON_SOURCE},
                exclude_types=set(DAEMON_EVENT_TYPES),
            )
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
        if self.task_store is None:
            return None
        try:
            tasks = self.task_store.list()
            pending = [t for t in tasks if t.status == PENDING]
            claimed = [t for t in tasks if t.status == CLAIMED]
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

    def check_worktree_prune(self) -> list[dict[str, Any]] | None:
        """清理孤儿 worktree，并在发生清理时发布报告。"""
        if self.worktree_runner is None:
            return None
        cleaned = self.worktree_runner.prune_stale()
        if not cleaned:
            return None
        return [
            {
                "type": "worktree_prune_report",
                "payload": {
                    "pruned_count": len(cleaned),
                    "paths": [str(path) for path in cleaned],
                },
            }
        ]

    def _persist_task_names(self) -> None:
        """持久化自定义任务名清单到 .local/daemon_tasks.json。"""
        custom_names = sorted(
            n for n in self._persistent_names if n not in _BUILTIN_TASKS
        )
        payload = {"tasks": [{"name": n, "type": "custom"} for n in custom_names]}
        self._tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._tasks_file.with_name(
            f".{self._tasks_file.name}.{uuid.uuid4().hex}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with self._tasks_lock:
            os.replace(tmp_path, self._tasks_file)

    def _restore_persistent_tasks(self) -> None:
        """从 .local/daemon_tasks.json 恢复持久化任务名。

        builtin 任务已在 __init__ 显式注册；此处仅恢复自定义任务名，
        但 callable 需外部重新 register_task 传入函数后才会真正执行。
        """
        if not self._tasks_file.exists():
            return
        try:
            data = json.loads(self._tasks_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("failed to load daemon_tasks.json")
            return
        for entry in data.get("tasks", []):
            name = str(entry.get("name", ""))
            if name and name not in _BUILTIN_TASKS:
                self._persistent_names.add(name)
                logger.info(
                    "restored persistent daemon task name %s (callable pending)", name
                )
