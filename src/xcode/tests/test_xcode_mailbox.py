from __future__ import annotations

import json
import tempfile
from pathlib import Path
import logging

from xcode.harness.mailbox import (
    AgentMailbox,
    LocalFileMailboxTransport,
    build_mailbox_tools,
)
from xcode.harness.task_store import TaskStore
import pytest


class TestTaskStoreAndMailbox:
    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Suppress logging warnings to avoid cluttering test outputs
        logging.getLogger("xcode.harness.mailbox").setLevel(logging.ERROR)

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_task_store_with_filelock(self) -> None:
        store = TaskStore(self.root)
        task1 = store.create("Test Task 1", {"foo": "bar"})
        assert task1.id == 1
        assert task1.title == "Test Task 1"
        assert task1.status == "pending"

        # Test claim
        claimed = store.claim(1, "worker1")
        assert claimed is not None
        assert claimed is not None
        assert claimed.status == "claimed"
        assert claimed.claimed_by == "worker1"

        # List all
        tasks = store.list()
        assert len(tasks) == 1
        assert tasks[0].id == 1

    def test_agent_mailbox_basic_flow(self) -> None:
        mailbox = AgentMailbox(self.root)

        # Send messages
        msg_id1 = mailbox.send_message(
            "agent_a", "agent_b", "query", {"question": "What is 1+1?"}
        )
        msg_id2 = mailbox.send_message("agent_c", "agent_b", "ping", {})

        # Read unread
        unread = mailbox.read_unread_messages("agent_b")
        assert len(unread) == 2
        assert unread[0]["message_id"] == msg_id1
        assert unread[1]["message_id"] == msg_id2

        # Acknowledge first message
        mailbox.acknowledge_message(msg_id1, "agent_b")

        # Read unread again
        unread_after = mailbox.read_unread_messages("agent_b")
        assert len(unread_after) == 1
        assert unread_after[0]["message_id"] == msg_id2

    def test_agent_mailbox_idempotence(self) -> None:
        mailbox = AgentMailbox(self.root)
        msg_id = mailbox.send_message("agent_a", "agent_b", "query", {})

        # Acknowledge multiple times
        mailbox.acknowledge_message(msg_id, "agent_b")
        mailbox.acknowledge_message(msg_id, "agent_b")
        mailbox.acknowledge_message(msg_id, "agent_b")

        # Verify it is unread-free
        unread = mailbox.read_unread_messages("agent_b")
        assert len(unread) == 0

    def test_agent_mailbox_corrupt_lines_tolerance(self) -> None:
        mailbox = AgentMailbox(self.root)
        recipient_id = "agent_corrupt"

        # Send good message
        msg_id1 = mailbox.send_message("agent_a", recipient_id, "good1", {})

        # Manually append some corrupt lines
        mailbox_file = mailbox.inbox_dir / f"{recipient_id}.jsonl"
        with open(mailbox_file, "a", encoding="utf-8") as f:
            f.write("{\n")  # invalid JSON
            f.write('{"event": "corrupt", "message_id": }\n')  # invalid JSON
            f.write("\n")  # empty line

        # Send another good message
        msg_id2 = mailbox.send_message("agent_a", recipient_id, "good2", {})

        # Read unread - should gracefully skip the corrupt lines
        unread = mailbox.read_unread_messages(recipient_id)
        assert len(unread) == 2
        assert unread[0]["message_id"] == msg_id1
        assert unread[1]["message_id"] == msg_id2

    def test_mailbox_tools_basic_flow(self) -> None:
        tools = {
            tool.name: tool for tool in build_mailbox_tools(AgentMailbox(self.root))
        }

        sent = tools["send_mailbox_message"].handler(
            {
                "sender_id": "agent_a",
                "recipient_id": "agent_b",
                "type": "query",
                "payload": {"question": "ping"},
            }
        )
        assert "sent message" in sent
        message_id = sent.split()[2]

        unread = tools["read_mailbox_messages"].handler({"recipient_id": "agent_b"})
        assert message_id in unread
        assert '"question": "ping"' in unread

        acked = tools["acknowledge_mailbox_message"].handler(
            {"recipient_id": "agent_b", "message_id": message_id}
        )
        assert "acknowledged message" in acked

        assert (
            tools["read_mailbox_messages"].handler({"recipient_id": "agent_b"}) == "[]"
        )

    def test_send_message_with_metadata(self) -> None:
        """send_message 带 thread_id/priority/expires_at 时写入 event 顶层。"""
        import json

        mailbox = AgentMailbox(self.root)
        mailbox.send_message(
            "agent_a",
            "agent_b",
            "query",
            {"q": 1},
            thread_id="t1",
            priority="high",
            expires_at="2026-12-31T23:59:59Z",
        )
        path = mailbox.inbox_dir / "agent_b.jsonl"
        line = path.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["thread_id"] == "t1"
        assert data["priority"] == "high"
        assert data["expires_at"] == "2026-12-31T23:59:59Z"

    def test_read_sort_by_priority(self) -> None:
        """sort_by=priority 时 high 排在 low 前面。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("a", "b", "t", {}, priority="low")
        mailbox.send_message("a", "b", "t", {}, priority="high")
        mailbox.send_message("a", "b", "t", {}, priority="normal")
        unread = mailbox.read_unread_messages("b", sort_by="priority")
        priorities = [m.get("priority") for m in unread]
        assert priorities == ["high", "normal", "low"]

    def test_read_sort_by_created_at_default(self) -> None:
        """默认按 created_at 排序。"""
        mailbox = AgentMailbox(self.root)
        id1 = mailbox.send_message("a", "b", "t", {})
        id2 = mailbox.send_message("a", "b", "t", {})
        unread = mailbox.read_unread_messages("b")
        assert unread[0]["message_id"] == id1
        assert unread[1]["message_id"] == id2

    def test_read_filter_type(self) -> None:
        """filter_type 仅返回匹配类型的消息。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("a", "b", "query", {})
        mailbox.send_message("a", "b", "alert", {})
        mailbox.send_message("a", "b", "query", {})
        unread = mailbox.read_unread_messages("b", filter_type="query")
        assert len(unread) == 2
        assert all(m["type"] == "query" for m in unread)

    def test_read_exclude_senders(self) -> None:
        """exclude_senders 过滤指定 sender 的消息。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("daemon", "b", "t", {})
        mailbox.send_message("user", "b", "t", {})
        unread = mailbox.read_unread_messages("b", exclude_senders={"daemon"})
        assert len(unread) == 1
        assert unread[0]["sender"] == "user"

    def test_read_exclude_types(self) -> None:
        """exclude_types 过滤指定 type 的消息。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("a", "b", "daemon_task_error", {})
        mailbox.send_message("a", "b", "query", {})
        mailbox.send_message("a", "b", "tasks_summary", {})
        unread = mailbox.read_unread_messages(
            "b", exclude_types={"daemon_task_error", "tasks_summary"}
        )
        assert len(unread) == 1
        assert unread[0]["type"] == "query"

    def test_read_combined_filters(self) -> None:
        """组合 filter_type + exclude_senders。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("daemon", "b", "query", {})
        mailbox.send_message("user", "b", "query", {})
        mailbox.send_message("user", "b", "alert", {})
        unread = mailbox.read_unread_messages(
            "b", filter_type="query", exclude_senders={"daemon"}
        )
        assert len(unread) == 1
        assert unread[0]["sender"] == "user"
        assert unread[0]["type"] == "query"

    def test_old_messages_without_metadata_backward_compatible(self) -> None:
        """旧消息无 priority/thread_id 字段，read 不报错，sort 按 created_at 兜底。"""
        import json

        mailbox = AgentMailbox(self.root)
        # 手动写入旧格式消息（无 priority 字段）
        path = mailbox.inbox_dir / "b.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "event": "message",
                    "message_id": "old-1",
                    "created_at": "2026-01-01T00:00:00Z",
                    "sender": "a",
                    "recipient": "b",
                    "type": "t",
                    "payload": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unread = mailbox.read_unread_messages("b", sort_by="priority")
        assert len(unread) == 1
        assert unread[0]["message_id"] == "old-1"

    def test_send_mailbox_message_tool_with_metadata(self) -> None:
        """send_mailbox_message 工具透传 priority/thread_id。"""
        import json

        tools = {
            tool.name: tool for tool in build_mailbox_tools(AgentMailbox(self.root))
        }
        tools["send_mailbox_message"].handler(
            {
                "sender_id": "a",
                "recipient_id": "b",
                "type": "query",
                "payload": {},
                "priority": "high",
                "thread_id": "t1",
            }
        )
        path = mailbox_inbox(self.root) / "b.jsonl"
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["priority"] == "high"
        assert data["thread_id"] == "t1"

    def test_read_mailbox_message_tool_with_sort_and_filter(self) -> None:
        """read_mailbox_messages 工具支持 sort_by 和 filter_type。"""
        tools = {
            tool.name: tool for tool in build_mailbox_tools(AgentMailbox(self.root))
        }
        tools["send_mailbox_message"].handler(
            {"sender_id": "a", "recipient_id": "b", "type": "alert", "payload": {}}
        )
        tools["send_mailbox_message"].handler(
            {
                "sender_id": "a",
                "recipient_id": "b",
                "type": "query",
                "payload": {},
                "priority": "high",
            }
        )
        result = tools["read_mailbox_messages"].handler(
            {"recipient_id": "b", "sort_by": "priority", "filter_type": "query"}
        )
        import json

        messages = json.loads(result)
        assert len(messages) == 1
        assert messages[0]["type"] == "query"


def mailbox_inbox(root: Path) -> Path:
    return root / ".local" / "team" / "inbox"


class TestMailboxExpiryAndAckSeparation:
    """3.1 过期清理 + 3.2 ACK 分离存储。"""

    def setup_method(self, method) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        logging.getLogger("xcode.harness.mailbox").setLevel(logging.ERROR)

    def teardown_method(self, method) -> None:
        self.temp_dir.cleanup()

    def test_send_message_includes_default_expires_at(self) -> None:
        """send_message 自动写入 expires_at（默认 retention_days 后）。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("a", "b", "t", {})
        path = mailbox_inbox(self.root) / "b.jsonl"
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert "expires_at" in data
        # 格式应为 ISO 8601
        assert data["expires_at"].endswith("Z")

    def test_send_message_custom_expires_at_overrides_default(self) -> None:
        """调用方显式传 expires_at 时覆盖默认。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("a", "b", "t", {}, expires_at="2020-01-01T00:00:00Z")
        path = mailbox_inbox(self.root) / "b.jsonl"
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["expires_at"] == "2020-01-01T00:00:00Z"

    def test_read_skips_expired_messages(self) -> None:
        """read_unread_messages 跳过过期消息。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("a", "b", "t", {}, expires_at="2020-01-01T00:00:00Z")
        mailbox.send_message("a", "b", "t", {}, expires_at="2099-12-31T23:59:59Z")
        unread = mailbox.read_unread_messages("b")
        assert len(unread) == 1
        # 保留的是未过期的
        assert unread[0]["expires_at"] == "2099-12-31T23:59:59Z"

    def test_read_treats_missing_expires_at_as_never_expire(self) -> None:
        """旧消息无 expires_at 字段视为永不过期。"""
        mailbox = AgentMailbox(self.root)
        path = mailbox_inbox(self.root) / "b.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "event": "message",
                    "message_id": "old-1",
                    "created_at": "2020-01-01T00:00:00Z",
                    "sender": "a",
                    "recipient": "b",
                    "type": "t",
                    "payload": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unread = mailbox.read_unread_messages("b")
        assert len(unread) == 1
        assert unread[0]["message_id"] == "old-1"

    def test_ack_written_to_separate_file(self) -> None:
        """ACK 事件写入 .ack 文件，主 .jsonl 不含 ack 行。"""
        mailbox = AgentMailbox(self.root)
        msg_id = mailbox.send_message("a", "b", "t", {})
        mailbox.acknowledge_message(msg_id, "b")
        main_path = mailbox_inbox(self.root) / "b.jsonl"
        ack_path = mailbox_inbox(self.root) / "b.ack"
        assert ack_path.exists()
        # 主文件仅含 message 行
        main_lines = [
            line for line in main_path.read_text(encoding="utf-8").splitlines() if line
        ]
        assert all(json.loads(line).get("event") == "message" for line in main_lines)
        # ack 文件含 ack 行
        ack_lines = [
            line for line in ack_path.read_text(encoding="utf-8").splitlines() if line
        ]
        assert any(json.loads(line).get("event") == "ack" for line in ack_lines)

    def test_read_merges_main_and_ack_files(self) -> None:
        """read 合并主文件 message 与 .ack 文件 ack 计算 unread。"""
        mailbox = AgentMailbox(self.root)
        msg_id = mailbox.send_message("a", "b", "t", {})
        mailbox.acknowledge_message(msg_id, "b")
        unread = mailbox.read_unread_messages("b")
        assert unread == []

    def test_cleanup_removes_expired_messages(self) -> None:
        """cleanup_expired_messages 剔除过期 message。"""
        transport = LocalFileMailboxTransport(self.root)
        mailbox = AgentMailbox(self.root, transport=transport)
        mailbox.send_message("a", "b", "t", {}, expires_at="2020-01-01T00:00:00Z")
        mailbox.send_message("a", "b", "t", {}, expires_at="2099-12-31T23:59:59Z")
        removed = mailbox.cleanup_expired_messages("b")
        assert removed == 1
        # 主文件仅剩 1 条
        main_path = mailbox_inbox(self.root) / "b.jsonl"
        lines = [
            line for line in main_path.read_text(encoding="utf-8").splitlines() if line
        ]
        assert len(lines) == 1
        assert json.loads(lines[0])["expires_at"] == "2099-12-31T23:59:59Z"

    def test_cleanup_preserves_ack_for_surviving_message(self) -> None:
        """cleanup 保留存活 message 对应的 ack，避免被重新算成未读。"""
        mailbox = AgentMailbox(self.root)
        surviving_id = mailbox.send_message(
            "a", "b", "t", {}, expires_at="2099-12-31T23:59:59Z"
        )
        mailbox.send_message("a", "b", "t", {}, expires_at="2020-01-01T00:00:00Z")
        mailbox.acknowledge_message(surviving_id, "b")
        mailbox.cleanup_expired_messages("b")
        # cleanup 后存活 message 仍被视为已读
        unread = mailbox.read_unread_messages("b")
        assert surviving_id not in [m["message_id"] for m in unread]

    def test_cleanup_removes_acks_for_expired_messages(self) -> None:
        """cleanup 丢弃过期 message 对应的 ack。"""
        mailbox = AgentMailbox(self.root)
        expired_id = mailbox.send_message(
            "a", "b", "t", {}, expires_at="2020-01-01T00:00:00Z"
        )
        mailbox.acknowledge_message(expired_id, "b")
        mailbox.cleanup_expired_messages("b")
        ack_path = mailbox_inbox(self.root) / "b.ack"
        if ack_path.exists():
            ack_lines = [
                line
                for line in ack_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            assert all(
                json.loads(line).get("message_id") != expired_id for line in ack_lines
            )

    def test_cleanup_noop_when_nothing_expired(self) -> None:
        """无过期消息时 cleanup 是 no-op，不重写文件。"""
        mailbox = AgentMailbox(self.root)
        mailbox.send_message("a", "b", "t", {}, expires_at="2099-12-31T23:59:59Z")
        main_path = mailbox_inbox(self.root) / "b.jsonl"
        mtime_before = main_path.stat().st_mtime_ns
        removed = mailbox.cleanup_expired_messages("b")
        assert removed == 0
        mtime_after = main_path.stat().st_mtime_ns
        assert mtime_before == mtime_after

    def test_daemon_check_mailbox_triggers_cleanup(self) -> None:
        """daemon.check_mailbox 顺带清理过期消息。"""
        from xcode.harness.daemon import HeartbeatDaemon
        from xcode.harness.mailbox import AgentMailbox as _AgentMailbox
        from xcode.harness.task_store import TaskStore as _TaskStore

        daemon = HeartbeatDaemon(
            self.root,
            mailbox=_AgentMailbox(self.root),
            task_store=_TaskStore(self.root),
            agent_id="test_agent",
        )
        mailbox = daemon.mailbox
        # 写入一条过期消息
        mailbox.send_message(
            "a", "test_agent", "t", {}, expires_at="2020-01-01T00:00:00Z"
        )
        main_path = mailbox_inbox(self.root) / "test_agent.jsonl"
        assert main_path.exists()
        daemon.check_mailbox()
        # cleanup 后主文件应被重写为空或仅含非过期行
        lines = [
            line for line in main_path.read_text(encoding="utf-8").splitlines() if line
        ]
        assert all(
            json.loads(line).get("expires_at") != "2020-01-01T00:00:00Z"
            for line in lines
        )


if __name__ == "__main__":
    pytest.main()
