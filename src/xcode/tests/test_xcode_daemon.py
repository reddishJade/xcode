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

        assert seen == [{"type": "custom_alert", "payload": {"info": "callback"}}]

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


if __name__ == "__main__":
    pytest.main()
