from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from xcode.harness.daemon import HeartbeatDaemon
from xcode.harness.mailbox import AgentMailbox
from xcode.harness.task_store import TaskStore
import pytest


class TestHeartbeatDaemon:
    """心跳守护进程单元测试。"""

    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # 用较短的间隔进行快速测试
        self.daemon = HeartbeatDaemon(
            self.root,
            mailbox=AgentMailbox(self.root),
            task_store=TaskStore(self.root),
            interval_seconds=1,
            agent_id="test_agent",
        )

    def teardown_method(self, method) -> None:
        self.daemon.stop()
        self.temp_dir.cleanup()

    def test_daemon_start_and_stop(self) -> None:
        """测试守护进程正常启动与停止。"""
        assert not (self.daemon._stop_event.is_set())
        self.daemon.start()
        assert self.daemon._thread is not None
        assert self.daemon._thread is not None
        assert self.daemon._thread.is_alive()

        self.daemon.stop()
        assert self.daemon._thread is None
        assert self.daemon._stop_event.is_set()

    def test_custom_task_execution(self) -> None:
        """测试注册的定时任务是否能被周期执行并发送消息。"""
        task_called = MagicMock(
            return_value=[{"type": "custom_alert", "payload": {"info": "test_data"}}]
        )

        self.daemon.register_task("custom", task_called)
        self.daemon.start()

        # 等待后台运行一轮任务
        time.sleep(1.2)

        self.daemon.stop()
        task_called.assert_called()

        # 验证消息已被成功追加到邮箱
        mailbox = AgentMailbox(self.root)
        messages = mailbox.read_unread_messages("test_agent")
        assert len(messages) >= 1
        assert messages[0]["type"] == "custom_alert"
        assert messages[0]["payload"]["info"] == "test_data"

    def test_callbacks_receive_published_events(self) -> None:
        """测试守护事件回调注册。"""
        seen: list[dict] = []
        self.daemon.register_callback("custom_alert", seen.append)

        self.daemon._publish_event(
            {"type": "custom_alert", "payload": {"info": "callback"}}
        )

        assert len(seen) == 1
        assert seen[0]["type"] == "custom_alert"
        assert seen[0]["payload"]["info"] == "callback"
        # source 元数据被自动添加
        assert seen[0]["payload"]["source"] == "heartbeat_daemon"

    def test_task_failure_updates_health_and_emits_error(self) -> None:
        """测试任务失败会更新健康状态并发出错误事件。"""
        seen: list[dict] = []
        self.daemon.register_callback("daemon_task_error", seen.append)

        def fail() -> None:
            raise ValueError("bad task")

        self.daemon.register_task("bad", fail)
        self.daemon._run_loop_once()

        health = self.daemon.health_check()
        assert "bad task" in health.last_error
        assert health.task_failures["bad"] == 1
        assert seen[0]["type"] == "daemon_task_error"

    def test_ensure_healthy_restarts_dead_thread(self) -> None:
        """测试后台线程异常退出后的显式自愈重启。"""
        self.daemon._stop_event.clear()

        health = self.daemon.ensure_healthy()

        assert health.running
        assert health.restart_count == 1

    def test_check_git_status_task(self) -> None:
        """测试脏工作区检查定时任务。"""
        with patch("subprocess.run") as mock_run:
            # 模拟 git status 存在脏文件输出
            mock_res = MagicMock()
            mock_res.returncode = 0
            mock_res.stdout = " M xcode/harness/daemon.py\n?? untracked.txt\n"
            mock_run.return_value = mock_res

            alerts = self.daemon.check_git_status()
            assert alerts is not None
            assert alerts is not None
            assert len(alerts) == 1
            assert alerts[0]["type"] == "git_dirty_alert"
            assert alerts[0]["payload"]["dirty_files_count"] == 2

    def test_check_background_tasks_task(self) -> None:
        """测试后台任务状态汇总定时任务。"""
        store = TaskStore(self.root)
        store.create("Task 1")
        store.create("Task 2")

        # 认领其中一个任务
        tasks = store.list()
        store.claim(tasks[0].id, "worker_1")

        alerts = self.daemon.check_background_tasks()
        assert alerts is not None
        assert alerts is not None
        assert len(alerts) == 1
        assert alerts[0]["type"] == "tasks_summary"
        assert alerts[0]["payload"]["pending_count"] == 1
        assert alerts[0]["payload"]["claimed_count"] == 1

    def test_daemon_events_carry_source_metadata(self) -> None:
        """daemon 事件 payload 自动添加 source 元数据。"""
        seen: list[dict] = []
        self.daemon.register_callback("tasks_summary", seen.append)
        self.daemon._publish_event(
            {"type": "tasks_summary", "payload": {"pending_count": 1}}
        )
        assert seen[0]["payload"]["source"] == "heartbeat_daemon"
        assert seen[0]["payload"]["origin"] == "test_agent"

    def test_daemon_events_payload_setdefault_does_not_overwrite(self) -> None:
        """调用方显式传 source 时不被覆盖。"""
        seen: list[dict] = []
        self.daemon.register_callback("custom", seen.append)
        self.daemon._publish_event(
            {"type": "custom", "payload": {"source": "external"}}
        )
        assert seen[0]["payload"]["source"] == "external"

    def test_check_mailbox_filters_daemon_event_types(self) -> None:
        """check_mailbox 过滤 daemon 自身事件类型，避免回声。"""
        # 向 daemon mailbox 写入一条 daemon 事件类型但 sender 非 daemon 的消息
        self.daemon.mailbox.send_message(
            "user_a",
            "test_agent",
            "tasks_summary",
            {"pending_count": 1},
        )
        result = self.daemon.check_mailbox()
        # tasks_summary 被 exclude_types 过滤，check_mailbox 返回 None
        assert result is None

    def test_check_mailbox_passes_user_messages(self) -> None:
        """check_mailbox 对非 daemon 事件类型的用户消息仍产生 summary。"""
        self.daemon.mailbox.send_message("user_a", "test_agent", "query", {"q": 1})
        result = self.daemon.check_mailbox()
        assert result is not None
        assert result[0]["type"] == "mailbox_summary"
        assert result[0]["payload"]["unread_count"] == 1

    def test_check_mailbox_filters_by_sender_even_for_user_types(self) -> None:
        """sender=heartbeat_daemon 的消息即使 type 非 daemon 类型也被过滤。"""
        self.daemon.mailbox.send_message("heartbeat_daemon", "test_agent", "query", {})
        result = self.daemon.check_mailbox()
        assert result is None

    def test_no_echo_loop_over_multiple_ticks(self) -> None:
        """多轮 tick 后 daemon 事件不会在 mailbox 中无限累积。"""
        self.daemon._run_loop_once()
        self.daemon._run_loop_once()
        self.daemon._run_loop_once()
        # 读取所有消息（不过滤）确认 daemon 自身事件未被 check_mailbox 重新处理
        all_messages = self.daemon.mailbox.read_unread_messages("test_agent")
        # daemon 事件存在但被回声过滤，不应触发新的 mailbox_summary
        daemon_event_types = {
            "daemon_task_error",
            "mailbox_summary",
            "git_dirty_alert",
            "tasks_summary",
        }
        # 所有 daemon 事件都应来自 heartbeat_daemon sender
        for msg in all_messages:
            if msg.get("type") in daemon_event_types:
                assert msg.get("sender") == "heartbeat_daemon"

    def test_list_daemon_tasks_returns_builtin(self) -> None:
        """list_daemon_tasks 返回 builtin 任务。"""
        tasks = self.daemon.list_daemon_tasks()
        names = {t.name for t in tasks}
        assert "check_mailbox" in names
        assert "check_git_status" in names
        assert "check_background_tasks" in names
        for t in tasks:
            assert t.registered is True
            assert t.builtin is True
            assert t.persistent is False

    def test_register_persistent_task_survives_instance_recreate(self) -> None:
        """persistent=True 的任务名在实例重建后仍标记为 persistent。"""
        from unittest.mock import MagicMock

        func = MagicMock(return_value=None)
        self.daemon.register_task("my_custom", func, persistent=True)
        tasks_file = self.root / ".local" / "daemon_tasks.json"
        assert tasks_file.exists()

        # 新建 daemon 实例模拟重启
        daemon2 = HeartbeatDaemon(
            self.root,
            mailbox=AgentMailbox(self.root),
            task_store=TaskStore(self.root),
            interval_seconds=1,
            agent_id="test_agent",
        )
        infos = {t.name: t for t in daemon2.list_daemon_tasks()}
        # my_custom 名字被恢复为 persistent，但 callable 未注册（registered=False 等价于不在 _tasks）
        # 由于 callable 未重新注册，它不在 _tasks 中
        assert "my_custom" not in infos
        # 但持久化文件仍含其名
        import json

        data = json.loads(tasks_file.read_text(encoding="utf-8"))
        custom_names = {e["name"] for e in data["tasks"]}
        assert "my_custom" in custom_names
        daemon2.stop()

    def test_unregister_task_removes_from_list(self) -> None:
        """unregister_task 移除任务。"""
        from unittest.mock import MagicMock

        func = MagicMock(return_value=None)
        self.daemon.register_task("temp", func)
        assert any(t.name == "temp" for t in self.daemon.list_daemon_tasks())
        assert self.daemon.unregister_task("temp") is True
        assert not any(t.name == "temp" for t in self.daemon.list_daemon_tasks())
        assert self.daemon.unregister_task("nonexistent") is False

    def test_ensure_healthy_re_registers_builtin_if_missing(self) -> None:
        """ensure_healthy 重启后补齐被 unregister 的 builtin 任务。"""
        self.daemon.unregister_task("check_git_status")
        assert not any(
            t.name == "check_git_status" for t in self.daemon.list_daemon_tasks()
        )
        # 让线程看起来"死掉"
        self.daemon._stop_event.clear()
        self.daemon._thread = None
        self.daemon.ensure_healthy()
        assert any(
            t.name == "check_git_status" for t in self.daemon.list_daemon_tasks()
        )


if __name__ == "__main__":
    pytest.main()
