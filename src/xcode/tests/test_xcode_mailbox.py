from __future__ import annotations

import tempfile
from pathlib import Path
import logging

from xcode.harness.mailbox import AgentMailbox, build_mailbox_tools
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

        assert tools["read_mailbox_messages"].handler({"recipient_id": "agent_b"}) == "[]"

if __name__ == "__main__":
    pytest.main()
