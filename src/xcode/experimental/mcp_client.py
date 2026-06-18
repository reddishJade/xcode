"""MCP 协议客户端。

基于 stdio 的 JSON-RPC 2.0 客户端，实现 Model Context Protocol 的
initialize 握手、tools/list 和 tools/call。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, BinaryIO, cast

logger = logging.getLogger("xcode.experimental.mcp_client")

_REDACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"(Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"(sk-)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:api_key|token|secret)=)[^\s]+", re.IGNORECASE),
]


def redact_mcp_text(text: str) -> str:
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(r"\1****", text)
    return text


def truncate_redact(text: str, max_len: int = 200) -> str:
    redacted = redact_mcp_text(text)
    if len(redacted) > max_len:
        redacted = redacted[:max_len] + "..."
    return redacted


class McpClient:
    """Standard Synchronous Model Context Protocol (MCP) Client over Stdio."""

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.command = command
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
        self.process: subprocess.Popen | None = None
        self.next_id = 1
        self._lock = threading.Lock()
        self._pending_responses: dict[int | str, dict[str, Any]] = {}
        self._read_thread: threading.Thread | None = None
        self._running = False
        self._timeout = timeout
        self._status: str = "pending"

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> None:
        if self.process is not None:
            return
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
            )
        except OSError as e:
            self._status = "failed"
            raise RuntimeError(f"Failed to start MCP server {self.command}: {e}")

        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        _active_clients.append(self)

        try:
            self.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "xcode-client", "version": "0.1.0"},
                },
            )
            self.send_notification("notifications/initialized")
        except Exception as e:
            self.stop()
            self._status = "failed"
            raise RuntimeError(f"MCP handshake failed: {e}")

        self._status = "connected"

    def _read_loop(self) -> None:
        while self._running and self.process and self.process.stdout:
            msg = _read_jsonrpc_message(cast(BinaryIO, self.process.stdout))
            if msg is None:
                break
            if "id" in msg:
                with self._lock:
                    self._pending_responses[msg["id"]] = msg

    def send_request(
        self, method: str, params: dict, timeout: float | None = None
    ) -> dict[str, Any]:
        if not self._running or not self.process:
            raise RuntimeError("MCP client is not running.")

        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout is None:
            effective_timeout = 10.0

        with self._lock:
            req_id = self.next_id
            self.next_id += 1

        req = {"jsonrpc": "2.0", "method": method, "id": req_id, "params": params}
        payload = _encode_jsonrpc_message(req)
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"Failed to write request to MCP server: {e}")

        start_time = time.time()
        while time.time() - start_time < effective_timeout:
            if self.process.poll() is not None:
                err_content = ""
                if self.process.stderr:
                    try:
                        raw_err = self.process.stderr.read()
                        raw_text = raw_err.decode("utf-8", errors="replace")
                        err_content = truncate_redact(raw_text, max_len=200)
                    except Exception:
                        logger.debug("failed to read MCP server stderr", exc_info=True)
                raise RuntimeError(
                    f"MCP server process exited unexpectedly. Stderr: {err_content}"
                )
            with self._lock:
                if req_id in self._pending_responses:
                    resp = self._pending_responses.pop(req_id)
                    if "error" in resp:
                        raise RuntimeError(f"MCP error response: {resp['error']}")
                    return resp.get("result", {})
            time.sleep(0.01)

        raise TimeoutError(f"Timeout waiting for response to {method} (id={req_id})")

    def send_notification(self, method: str, params: dict | None = None) -> None:
        if not self._running or not self.process:
            raise RuntimeError("MCP client is not running.")

        req: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params

        payload = _encode_jsonrpc_message(req)
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"Failed to send notification to MCP server: {e}")

    def list_tools(self, timeout: float | None = None) -> list[dict[str, Any]]:
        result = self.send_request("tools/list", {}, timeout=timeout)
        return result.get("tools", [])

    def call_tool(
        self, name: str, arguments: dict, timeout: float | None = None
    ) -> dict[str, Any]:
        return self.send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )

    def stop(self) -> None:
        self._running = False
        if self in _active_clients:
            try:
                _active_clients.remove(self)
            except ValueError:
                pass
        if self.process:
            for stream_attr in ("stdin", "stdout", "stderr"):
                stream = getattr(self.process, stream_attr, None)
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        logger.debug(
                            "failed to close MCP server stream %s",
                            stream_attr,
                            exc_info=True,
                        )
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    logger.debug("failed to kill MCP server process", exc_info=True)
            self.process = None
        self._status = "disabled"


# ── JSON-RPC 编解码 ──

_active_clients: list[McpClient] = []


def _encode_jsonrpc_message(message: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message in newline-delimited JSON format.

    MCP SDK >= 1.0 uses newline-delimited JSON (one JSON object per line)
    rather than the Content-Length header format used by earlier revisions.
    """
    body = json.dumps(message, ensure_ascii=False)
    return (body + "\n").encode("utf-8")


def _read_jsonrpc_message(stream: BinaryIO) -> dict[str, Any] | None:
    """Read a JSON-RPC message from a stdio stream.

    Supports both newline-delimited JSON (MCP SDK >= 1.0) and the older
    Content-Length header format for backward compatibility.
    """
    line = stream.readline()
    if not line:
        return None

    text = line.decode("utf-8", errors="replace")

    # NDJSON: line starts with '{'
    stripped = text.strip()
    if stripped and stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    # Content-Length header format
    headers: dict[str, str] = {}
    # The first line we read may already be a header
    header_line = text.strip()
    if ":" in header_line:
        key, value = header_line.split(":", 1)
        headers[key.lower()] = value.strip()

    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        header_text = line.decode("ascii", errors="replace").strip()
        if ":" not in header_text:
            continue
        key, value = header_text.split(":", 1)
        headers[key.lower()] = value.strip()

    length_text = headers.get("content-length")
    if not length_text:
        return None
    try:
        length = int(length_text)
    except ValueError:
        return None
    body = stream.read(length)
    if len(body) != length:
        return None
    try:
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _cleanup_clients() -> None:
    while _active_clients:
        client = _active_clients[0]
        try:
            client.stop()
        except Exception:
            logger.debug(
                "failed to stop MCP client during atexit cleanup", exc_info=True
            )


atexit.register(_cleanup_clients)


class LazyClientRef:
    """Manages lazy instantiation of MCP clients to avoid startup overhead."""

    def __init__(self, server_name: str, config: dict[str, Any]) -> None:
        self.server_name = server_name
        self.config = config
        self.client: McpClient | None = None

    def get_or_create(self) -> McpClient:
        if self.client is None or self.client.status == "failed":
            if self.client is not None:
                self.client.stop()
            command = [self.config["command"]] + self.config.get("args", [])
            env = self.config.get("env")
            timeout = self.config.get("timeout")
            self.client = McpClient(command, env, timeout=timeout)
            self.client.start()
        return self.client

    def stop(self) -> None:
        if self.client is not None:
            self.client.stop()
            self.client = None
