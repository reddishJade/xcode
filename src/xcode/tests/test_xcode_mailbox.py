from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import logging

from xcode.experimental.mailbox import AgentMailbox, build_mailbox_tools
from xcode.experimental.tasks import TaskStore


class TestTaskStoreAndMailbox(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Suppress logging warnings to avoid cluttering test outputs
        logging.getLogger("xcode.experimental.mailbox").setLevel(logging.ERROR)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_task_store_with_filelock(self) -> None:
        store = TaskStore(self.root)
        task1 = store.create("Test Task 1", {"foo": "bar"})
        self.assertEqual(task1.id, 1)
        self.assertEqual(task1.title, "Test Task 1")
        self.assertEqual(task1.status, "pending")

        # Test claim
        claimed = store.claim(1, "worker1")
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.status, "claimed")
        self.assertEqual(claimed.claimed_by, "worker1")

        # List all
        tasks = store.list()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, 1)

    def test_agent_mailbox_basic_flow(self) -> None:
        mailbox = AgentMailbox(self.root)

        # Send messages
        msg_id1 = mailbox.send_message(
            "agent_a", "agent_b", "query", {"question": "What is 1+1?"}
        )
        msg_id2 = mailbox.send_message("agent_c", "agent_b", "ping", {})

        # Read unread
        unread = mailbox.read_unread_messages("agent_b")
        self.assertEqual(len(unread), 2)
        self.assertEqual(unread[0]["message_id"], msg_id1)
        self.assertEqual(unread[1]["message_id"], msg_id2)

        # Acknowledge first message
        mailbox.acknowledge_message(msg_id1, "agent_b")

        # Read unread again
        unread_after = mailbox.read_unread_messages("agent_b")
        self.assertEqual(len(unread_after), 1)
        self.assertEqual(unread_after[0]["message_id"], msg_id2)

    def test_agent_mailbox_idempotence(self) -> None:
        mailbox = AgentMailbox(self.root)
        msg_id = mailbox.send_message("agent_a", "agent_b", "query", {})

        # Acknowledge multiple times
        mailbox.acknowledge_message(msg_id, "agent_b")
        mailbox.acknowledge_message(msg_id, "agent_b")
        mailbox.acknowledge_message(msg_id, "agent_b")

        # Verify it is unread-free
        unread = mailbox.read_unread_messages("agent_b")
        self.assertEqual(len(unread), 0)

    def test_agent_mailbox_corrupt_lines_tolerance(self) -> None:
        mailbox = AgentMailbox(self.root)
        recipient_id = "agent_corrupt"

        # Send good message
        msg_id1 = mailbox.send_message("agent_a", recipient_id, "good1", {})

        # Manually append some corrupt lines
        mailbox_file = mailbox._mailbox_path(recipient_id)
        with open(mailbox_file, "a", encoding="utf-8") as f:
            f.write("{\n")  # invalid JSON
            f.write('{"event": "corrupt", "message_id": }\n')  # invalid JSON
            f.write("\n")  # empty line

        # Send another good message
        msg_id2 = mailbox.send_message("agent_a", recipient_id, "good2", {})

        # Read unread - should gracefully skip the corrupt lines
        unread = mailbox.read_unread_messages(recipient_id)
        self.assertEqual(len(unread), 2)
        self.assertEqual(unread[0]["message_id"], msg_id1)
        self.assertEqual(unread[1]["message_id"], msg_id2)

    def test_mailbox_tools_basic_flow(self) -> None:
        tools = {
            tool.name: tool for tool in build_mailbox_tools(AgentMailbox(self.root))
        }

        sent = tools["send_mailbox_message"].handler(
            '{"sender_id":"agent_a","recipient_id":"agent_b","type":"query","payload":{"question":"ping"}}'
        )
        self.assertIn("sent message", sent)
        message_id = sent.split()[2]

        unread = tools["read_mailbox_messages"].handler('{"recipient_id":"agent_b"}')
        self.assertIn(message_id, unread)
        self.assertIn('"question": "ping"', unread)

        acked = tools["acknowledge_mailbox_message"].handler(
            f'{{"recipient_id":"agent_b","message_id":"{message_id}"}}'
        )
        self.assertIn("acknowledged message", acked)

        self.assertEqual(
            tools["read_mailbox_messages"].handler('{"recipient_id":"agent_b"}'),
            "[]",
        )


if __name__ == "__main__":
    unittest.main()
