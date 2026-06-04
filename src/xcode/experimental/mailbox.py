from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from typing import Protocol

from ..harness.skills import ToolInput, ToolSpec


class MailboxTransport(Protocol):
    """跨进程/跨机器 Agent 通信的传输层协议。"""

    def send_message(
        self, sender_id: str, recipient_id: str, type_name: str, payload: dict[str, Any]
    ) -> str: ...

    def read_unread_messages(self, recipient_id: str) -> list[dict[str, Any]]: ...

    def acknowledge_message(self, message_id: str, recipient_id: str) -> None: ...

logger = logging.getLogger("xcode.experimental.mailbox")


class LocalFileMailboxTransport:
    """基于 filelock + JSONL append-only 日志的本地文件系统传输层。"""

    def __init__(self, root: Path, lock_timeout_seconds: float = 5.0) -> None:
        self.root = root
        self.inbox_dir = root / ".team" / "inbox"
        self.lock_timeout_seconds = lock_timeout_seconds

    def _mailbox_path(self, agent_id: str) -> Path:
        return self.inbox_dir / f"{agent_id}.jsonl"

    def _lock_path(self, agent_id: str) -> Path:
        return self.inbox_dir / f"{agent_id}.lock"

    def send_message(
        self, sender_id: str, recipient_id: str, type_name: str, payload: dict[str, Any]
    ) -> str:
        import filelock

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        message_id = str(uuid.uuid4())
        event = {
            "event": "message",
            "message_id": message_id,
            "created_at": self._timestamp(),
            "sender": sender_id,
            "recipient": recipient_id,
            "type": type_name,
            "payload": payload,
        }
        lock = filelock.FileLock(
            self._lock_path(recipient_id), timeout=self.lock_timeout_seconds
        )
        with lock:
            path = self._mailbox_path(recipient_id)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        return message_id

    def read_unread_messages(self, recipient_id: str) -> list[dict[str, Any]]:
        import filelock

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        path = self._mailbox_path(recipient_id)
        if not path.exists():
            return []
        lock = filelock.FileLock(
            self._lock_path(recipient_id), timeout=self.lock_timeout_seconds
        )
        messages = {}
        acked_ids = set()
        with lock:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        event_type = data.get("event")
                        msg_id = data.get("message_id")
                        if not msg_id:
                            continue
                        if event_type == "message":
                            messages[msg_id] = data
                        elif event_type == "ack":
                            acked_ids.add(msg_id)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping bad line in mailbox: %s", exc)
                        continue
        unread = []
        for msg_id, msg in messages.items():
            if msg_id not in acked_ids:
                unread.append(msg)
        return sorted(unread, key=lambda m: m.get("created_at", ""))

    def acknowledge_message(self, message_id: str, recipient_id: str) -> None:
        import filelock

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "ack",
            "message_id": message_id,
            "recipient": recipient_id,
            "ack_at": self._timestamp(),
        }
        lock = filelock.FileLock(
            self._lock_path(recipient_id), timeout=self.lock_timeout_seconds
        )
        with lock:
            path = self._mailbox_path(recipient_id)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class AgentMailbox:
    """基于传输层抽象的并发代理邮箱，默认使用本地文件系统。"""

    def __init__(
        self,
        root: Path,
        transport: MailboxTransport | None = None,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self.root = root
        self._transport = transport or LocalFileMailboxTransport(
            root, lock_timeout_seconds
        )

    @property
    def inbox_dir(self) -> Path:
        if isinstance(self._transport, LocalFileMailboxTransport):
            return self._transport.inbox_dir
        return self.root / ".team" / "inbox"

    def send_message(
        self, sender_id: str, recipient_id: str, type_name: str, payload: dict[str, Any]
    ) -> str:
        return self._transport.send_message(sender_id, recipient_id, type_name, payload)

    def read_unread_messages(self, recipient_id: str) -> list[dict[str, Any]]:
        return self._transport.read_unread_messages(recipient_id)

    def acknowledge_message(self, message_id: str, recipient_id: str) -> None:
        self._transport.acknowledge_message(message_id, recipient_id)


def build_mailbox_tools(mailbox: AgentMailbox) -> tuple[ToolSpec, ...]:
    def send_mailbox_message(args: ToolInput) -> str:
        sender_id = str(args.get("sender_id", "")).strip()
        recipient_id = str(args.get("recipient_id", "")).strip()
        type_name = str(args.get("type", args.get("type_name", ""))).strip()
        payload = args.get("payload", {})
        if not sender_id:
            raise ValueError("sender_id is required")
        if not recipient_id:
            raise ValueError("recipient_id is required")
        if not type_name:
            raise ValueError("type is required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        message_id = mailbox.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            type_name=type_name,
            payload=payload,
        )
        return f"sent message {message_id} to {recipient_id}"

    def read_mailbox_messages(args: ToolInput) -> str:
        recipient_id = str(args.get("recipient_id", "")).strip()
        if not recipient_id:
            raise ValueError("recipient_id is required")
        messages = mailbox.read_unread_messages(recipient_id)
        return json.dumps(messages, ensure_ascii=False, indent=2)

    def acknowledge_mailbox_message(args: ToolInput) -> str:
        message_id = str(args.get("message_id", "")).strip()
        recipient_id = str(args.get("recipient_id", "")).strip()
        if not message_id:
            raise ValueError("message_id is required")
        if not recipient_id:
            raise ValueError("recipient_id is required")
        mailbox.acknowledge_message(message_id, recipient_id)
        return f"acknowledged message {message_id} for {recipient_id}"

    return (
        ToolSpec(
            name="send_mailbox_message",
            description="Send an append-only message event to an agent mailbox.",
            input_hint='{"sender_id":"agent_a","recipient_id":"agent_b","type":"query","payload":{}}',
            handler=send_mailbox_message,
            risk="low",
            schema={
                "type": "object",
                "properties": {
                    "sender_id": {"type": "string"},
                    "recipient_id": {"type": "string"},
                    "type": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["sender_id", "recipient_id", "type"],
                "additionalProperties": False,
            },
            group="mailbox",
        ),
        ToolSpec(
            name="read_mailbox_messages",
            description="Read unread message events for a recipient mailbox.",
            input_hint='{"recipient_id":"agent_b"}',
            handler=read_mailbox_messages,
            risk="low",
            schema={
                "type": "object",
                "properties": {"recipient_id": {"type": "string"}},
                "required": ["recipient_id"],
                "additionalProperties": False,
            },
            read_only=True,
            group="mailbox",
        ),
        ToolSpec(
            name="acknowledge_mailbox_message",
            description="Append an ACK event for a mailbox message.",
            input_hint='{"recipient_id":"agent_b","message_id":"..."}',
            handler=acknowledge_mailbox_message,
            risk="low",
            schema={
                "type": "object",
                "properties": {
                    "recipient_id": {"type": "string"},
                    "message_id": {"type": "string"},
                },
                "required": ["recipient_id", "message_id"],
                "additionalProperties": False,
            },
            group="mailbox",
        ),
    )
