"""本地 append-only Agent mailbox。"""

from __future__ import annotations

import calendar
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from xcode.harness.skills import ToolInput, ToolSpec
import filelock


_PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}
_DEFAULT_RETENTION_DAYS = 30


class MailboxTransport(Protocol):
    """跨进程/跨机器 Agent 通信的传输层协议。"""

    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        type_name: str,
        payload: dict[str, Any],
        *,
        thread_id: str | None = ...,
        priority: str | None = ...,
        expires_at: str | None = ...,
    ) -> str: ...

    def read_unread_messages(
        self,
        recipient_id: str,
        *,
        sort_by: str = ...,
        filter_type: str | None = ...,
        exclude_senders: set[str] | None = ...,
        exclude_types: set[str] | None = ...,
    ) -> list[dict[str, Any]]: ...

    def acknowledge_message(self, message_id: str, recipient_id: str) -> None: ...


logger = logging.getLogger("xcode.harness.mailbox")


def _parse_iso_timestamp(value: str) -> float:
    """解析 ISO 8601 UTC 时间戳为 epoch 秒；空或非法返回 0.0（视为永不过期）。"""
    if not value:
        return 0.0
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return 0.0


class LocalFileMailboxTransport:
    """基于 filelock + JSONL append-only 日志的本地文件系统传输层。

    消息存储在 ``{recipient}.jsonl``，ACK 事件分离到 ``{recipient}.ack``。
    支持基于 ``expires_at`` 的消息过期与 ``cleanup_expired_messages`` 重写压缩。
    """

    def __init__(
        self,
        root: Path,
        lock_timeout_seconds: float = 5.0,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.root = root
        self.inbox_dir = root / ".local" / "team" / "inbox"
        self.lock_timeout_seconds = lock_timeout_seconds
        self.retention_days = retention_days

    def _mailbox_path(self, agent_id: str) -> Path:
        return self.inbox_dir / f"{agent_id}.jsonl"

    def _ack_path(self, agent_id: str) -> Path:
        return self.inbox_dir / f"{agent_id}.ack"

    def _lock_path(self, agent_id: str) -> Path:
        return self.inbox_dir / f"{agent_id}.lock"

    def _default_expires_at(self) -> str:
        future = time.time() + self.retention_days * 86400
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(future))

    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        type_name: str,
        payload: dict[str, Any],
        *,
        thread_id: str | None = None,
        priority: str | None = None,
        expires_at: str | None = None,
    ) -> str:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        message_id = str(uuid.uuid4())
        event: dict[str, Any] = {
            "event": "message",
            "message_id": message_id,
            "created_at": self._timestamp(),
            "sender": sender_id,
            "recipient": recipient_id,
            "type": type_name,
            "payload": payload,
        }
        if thread_id is not None:
            event["thread_id"] = thread_id
        if priority is not None:
            event["priority"] = priority
        event["expires_at"] = expires_at or self._default_expires_at()
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

    def read_unread_messages(
        self,
        recipient_id: str,
        *,
        sort_by: str = "created_at",
        filter_type: str | None = None,
        exclude_senders: set[str] | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        path = self._mailbox_path(recipient_id)
        if not path.exists():
            return []
        lock = filelock.FileLock(
            self._lock_path(recipient_id), timeout=self.lock_timeout_seconds
        )
        messages = {}
        acked_ids = self._read_ack_ids(recipient_id, lock)
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
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping bad line in mailbox: %s", exc)
                        continue
        now = time.time()
        exclude_senders = exclude_senders or set()
        exclude_types = exclude_types or set()
        unread = []
        for msg_id, msg in messages.items():
            if msg_id in acked_ids:
                continue
            if msg.get("sender") in exclude_senders:
                continue
            if msg.get("type") in exclude_types:
                continue
            if filter_type is not None and msg.get("type") != filter_type:
                continue
            expires_at = _parse_iso_timestamp(str(msg.get("expires_at", "")))
            if expires_at and expires_at < now:
                continue
            unread.append(msg)
        if sort_by == "priority":
            unread.sort(
                key=lambda m: (
                    _PRIORITY_ORDER.get(m.get("priority", "normal"), 1),
                    m.get("created_at", ""),
                )
            )
        else:
            unread.sort(key=lambda m: m.get("created_at", ""))
        return unread

    def acknowledge_message(self, message_id: str, recipient_id: str) -> None:
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
            path = self._ack_path(recipient_id)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

    def cleanup_expired_messages(self, recipient_id: str) -> int:
        """重写主 JSONL 与 ACK 文件，剔除过期 message 及其对应的 ack。

        返回被清理的 message 条数。仅在存在过期或已 ACK 条目时触发重写。
        """
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        lock = filelock.FileLock(
            self._lock_path(recipient_id), timeout=self.lock_timeout_seconds
        )
        now = time.time()
        with lock:
            main_path = self._mailbox_path(recipient_id)
            ack_path = self._ack_path(recipient_id)
            if not main_path.exists():
                return 0
            surviving_messages: list[dict[str, Any]] = []
            surviving_ids: set[str] = set()
            expired_count = 0
            with open(main_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("event") != "message":
                        continue
                    msg_id = data.get("message_id")
                    if not msg_id:
                        continue
                    expires_at = _parse_iso_timestamp(str(data.get("expires_at", "")))
                    if expires_at and expires_at < now:
                        expired_count += 1
                        continue
                    surviving_messages.append(data)
                    surviving_ids.add(str(msg_id))
            if expired_count == 0 and not ack_path.exists():
                return 0
            # 重写主文件：仅存活 message
            self._atomic_write_lines(main_path, surviving_messages)
            # 重写 ack 文件：仅保留存活 message 对应的 ack
            surviving_acks: list[dict[str, Any]] = []
            if ack_path.exists():
                with open(ack_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            data.get("event") == "ack"
                            and str(data.get("message_id")) in surviving_ids
                        ):
                            surviving_acks.append(data)
                self._atomic_write_lines(ack_path, surviving_acks)
            return expired_count

    def _read_ack_ids(self, recipient_id: str, lock: filelock.FileLock) -> set[str]:
        """读取 ACK 文件中的 message_id 集合（调用方需持锁）。"""
        ack_path = self._ack_path(recipient_id)
        if not ack_path.exists():
            return set()
        acked = set()
        with open(ack_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("event") == "ack":
                        msg_id = data.get("message_id")
                        if msg_id:
                            acked.add(str(msg_id))
                except json.JSONDecodeError:
                    continue
        return acked

    def _atomic_write_lines(self, path: Path, items: list[dict[str, Any]]) -> None:
        """原子性地重写 JSONL 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)

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
        return self.root / ".local" / "team" / "inbox"

    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        type_name: str,
        payload: dict[str, Any],
        *,
        thread_id: str | None = None,
        priority: str | None = None,
        expires_at: str | None = None,
    ) -> str:
        return self._transport.send_message(
            sender_id,
            recipient_id,
            type_name,
            payload,
            thread_id=thread_id,
            priority=priority,
            expires_at=expires_at,
        )

    def read_unread_messages(
        self,
        recipient_id: str,
        *,
        sort_by: str = "created_at",
        filter_type: str | None = None,
        exclude_senders: set[str] | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._transport.read_unread_messages(
            recipient_id,
            sort_by=sort_by,
            filter_type=filter_type,
            exclude_senders=exclude_senders,
            exclude_types=exclude_types,
        )

    def acknowledge_message(self, message_id: str, recipient_id: str) -> None:
        self._transport.acknowledge_message(message_id, recipient_id)

    def cleanup_expired_messages(self, recipient_id: str) -> int:
        """清理过期与已 ACK 的消息，返回清理条数。

        仅当 transport 为 LocalFileMailboxTransport 时支持。
        """
        if isinstance(self._transport, LocalFileMailboxTransport):
            return self._transport.cleanup_expired_messages(recipient_id)
        return 0


def build_mailbox_tools(mailbox: AgentMailbox) -> tuple[ToolSpec, ...]:
    def send_mailbox_message(args: ToolInput) -> str:
        sender_id = str(args.get("sender_id", "")).strip()
        recipient_id = str(args.get("recipient_id", "")).strip()
        type_name = str(args.get("type", "")).strip()
        payload = args.get("payload", {})
        if not sender_id:
            raise ValueError("sender_id is required")
        if not recipient_id:
            raise ValueError("recipient_id is required")
        if not type_name:
            raise ValueError("type is required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        thread_id = args.get("thread_id")
        priority = args.get("priority")
        expires_at = args.get("expires_at")
        message_id = mailbox.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            type_name=type_name,
            payload=payload,
            thread_id=str(thread_id) if thread_id else None,
            priority=str(priority) if priority else None,
            expires_at=str(expires_at) if expires_at else None,
        )
        return f"sent message {message_id} to {recipient_id}"

    def read_mailbox_messages(args: ToolInput) -> str:
        recipient_id = str(args.get("recipient_id", "")).strip()
        if not recipient_id:
            raise ValueError("recipient_id is required")
        sort_by = str(args.get("sort_by", "created_at")).strip() or "created_at"
        filter_type = args.get("filter_type")
        exclude_senders_raw = args.get("exclude_senders")
        exclude_types_raw = args.get("exclude_types")
        exclude_senders = (
            set(exclude_senders_raw) if isinstance(exclude_senders_raw, list) else None
        )
        exclude_types = (
            set(exclude_types_raw) if isinstance(exclude_types_raw, list) else None
        )
        messages = mailbox.read_unread_messages(
            recipient_id,
            sort_by=sort_by,
            filter_type=str(filter_type) if filter_type else None,
            exclude_senders=exclude_senders,
            exclude_types=exclude_types,
        )
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
            description="Send an append-only message event to an agent mailbox. Optional thread_id, priority, expires_at metadata.",
            input_hint='{"sender_id":"agent_a","recipient_id":"agent_b","type":"query","payload":{},"priority":"high"}',
            handler=send_mailbox_message,
            schema={
                "type": "object",
                "properties": {
                    "sender_id": {"type": "string"},
                    "recipient_id": {"type": "string"},
                    "type": {"type": "string"},
                    "payload": {"type": "object"},
                    "thread_id": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "normal", "low"],
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "ISO 8601 timestamp; message skipped after this time",
                    },
                },
                "required": ["sender_id", "recipient_id", "type"],
                "additionalProperties": False,
            },
            group="mailbox",
        ),
        ToolSpec(
            name="read_mailbox_messages",
            description="Read unread message events for a recipient mailbox. Supports sort_by, filter_type, exclude_senders, exclude_types.",
            input_hint='{"recipient_id":"agent_b","sort_by":"priority","filter_type":"query"}',
            handler=read_mailbox_messages,
            schema={
                "type": "object",
                "properties": {
                    "recipient_id": {"type": "string"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["created_at", "priority"],
                    },
                    "filter_type": {"type": "string"},
                    "exclude_senders": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "exclude_types": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
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
