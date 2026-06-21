"""受信任外部命令 hook 的隔离执行与诊断状态。"""

from __future__ import annotations

import fnmatch
import json
import logging
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ..config import ExternalHookRuntimeConfig
from ..session import JsonValue
from .audit import redact_text
from .hooks import HookRecord

logger = logging.getLogger("xcode.harness.observability.external_hooks")

MAX_HOOK_STDOUT_CHARS = 64 * 1024
type ExternalHookStatus = Literal["succeeded", "failed"]


class ExternalHookFailure(RuntimeError):
    """failure_policy=fail 的外部 hook 执行失败。"""


@dataclass(frozen=True)
class ExternalHookExecution:
    """单次外部 hook 执行结果。"""

    index: int
    event: str
    source: str
    command: tuple[str, ...]
    status: ExternalHookStatus
    response: dict[str, JsonValue]
    error: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class ExternalHookDiagnostic:
    """用于 `/hooks` 展示的 hook 状态快照。"""

    index: int
    event: str
    source: str
    command: tuple[str, ...]
    matcher: str | None
    enabled: bool
    failure_policy: str
    inherit_to_subagents: bool
    run_count: int
    last_status: str
    last_error: str
    last_run_at: str


class ExternalHookRunner:
    """通过 JSON stdin/stdout 执行配置声明的外部命令。"""

    def __init__(
        self,
        entries: tuple[ExternalHookRuntimeConfig, ...],
        project_root: Path,
    ) -> None:
        """保存 hook 声明和项目工作目录。"""
        self._entries = entries
        self._project_root = project_root.resolve()
        self._lock = threading.Lock()
        self._diagnostics = {
            index: ExternalHookDiagnostic(
                index=index,
                event=entry.event,
                source=entry.source,
                command=entry.command,
                matcher=entry.matcher,
                enabled=entry.enabled,
                failure_policy=entry.failure_policy,
                inherit_to_subagents=entry.inherit_to_subagents,
                run_count=0,
                last_status="never",
                last_error="",
                last_run_at="",
            )
            for index, entry in enumerate(entries)
        }

    def execute(
        self,
        record: HookRecord,
        *,
        subagent: bool = False,
        cwd: Path | None = None,
    ) -> tuple[ExternalHookExecution, ...]:
        """按声明顺序执行匹配 hook，并应用失败策略。"""
        working_directory = (cwd or self._project_root).resolve()
        executions: list[ExternalHookExecution] = []
        for index, entry in enumerate(self._entries):
            if not _matches(entry, record, subagent):
                continue
            execution = self._execute_one(
                index,
                entry,
                record,
                working_directory,
            )
            executions.append(execution)
            if execution.status == "failed":
                self._handle_failure(entry, execution)
        return tuple(executions)

    def diagnostics(self) -> tuple[ExternalHookDiagnostic, ...]:
        """返回线程安全的 hook 诊断快照。"""
        with self._lock:
            return tuple(
                self._diagnostics[index] for index in sorted(self._diagnostics)
            )

    def _execute_one(
        self,
        index: int,
        entry: ExternalHookRuntimeConfig,
        record: HookRecord,
        working_directory: Path,
    ) -> ExternalHookExecution:
        """执行单个 hook 并记录结果。"""
        timestamp = datetime.now(UTC).isoformat()
        try:
            completed = subprocess.run(
                list(entry.command),
                cwd=working_directory,
                input=_hook_payload(record),
                text=True,
                capture_output=True,
                timeout=entry.timeout,
                shell=False,
                check=False,
            )
            if completed.returncode != 0:
                error = _process_error(completed.returncode, completed.stderr)
                return self._failed(index, entry, error, timestamp)
            response = _parse_hook_response(completed.stdout)
            _validate_hook_response(record.event, response)
        except subprocess.TimeoutExpired:
            return self._failed(
                index,
                entry,
                f"hook timed out after {entry.timeout:g} seconds",
                timestamp,
            )
        except OSError as exc:
            return self._failed(
                index,
                entry,
                f"failed to start hook command: {exc}",
                timestamp,
            )
        except ValueError as exc:
            return self._failed(index, entry, str(exc), timestamp)

        execution = ExternalHookExecution(
            index=index,
            event=entry.event,
            source=entry.source,
            command=entry.command,
            status="succeeded",
            response=response,
            timestamp=timestamp,
        )
        self._record_diagnostic(execution)
        return execution

    def _failed(
        self,
        index: int,
        entry: ExternalHookRuntimeConfig,
        error: str,
        timestamp: str,
    ) -> ExternalHookExecution:
        """构建并记录脱敏失败结果。"""
        execution = ExternalHookExecution(
            index=index,
            event=entry.event,
            source=entry.source,
            command=entry.command,
            status="failed",
            response={},
            error=redact_text(error),
            timestamp=timestamp,
        )
        self._record_diagnostic(execution)
        return execution

    def _record_diagnostic(self, execution: ExternalHookExecution) -> None:
        """更新最近一次执行诊断。"""
        with self._lock:
            previous = self._diagnostics[execution.index]
            self._diagnostics[execution.index] = ExternalHookDiagnostic(
                index=previous.index,
                event=previous.event,
                source=previous.source,
                command=previous.command,
                matcher=previous.matcher,
                enabled=previous.enabled,
                failure_policy=previous.failure_policy,
                inherit_to_subagents=previous.inherit_to_subagents,
                run_count=previous.run_count + 1,
                last_status=execution.status,
                last_error=execution.error,
                last_run_at=execution.timestamp,
            )

    @staticmethod
    def _handle_failure(
        entry: ExternalHookRuntimeConfig,
        execution: ExternalHookExecution,
    ) -> None:
        """按配置处理 hook 失败。"""
        message = (
            f"external hook {execution.index} ({entry.event}) failed: {execution.error}"
        )
        if entry.failure_policy == "ignore":
            return
        if entry.failure_policy == "warn":
            logger.warning(message)
            return
        raise ExternalHookFailure(message)


def _matches(
    entry: ExternalHookRuntimeConfig,
    record: HookRecord,
    subagent: bool,
) -> bool:
    """判断声明是否应处理当前事件。"""
    if not entry.enabled or entry.event != record.event:
        return False
    if subagent and not entry.inherit_to_subagents:
        return False
    if entry.matcher is None:
        return True
    return any(
        fnmatch.fnmatchcase(candidate, entry.matcher)
        for candidate in _match_candidates(record)
    )


def _match_candidates(record: HookRecord) -> tuple[str, ...]:
    """返回 matcher 可见的稳定候选值。"""
    metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
    candidates = [
        record.event,
        record.tool,
        str(metadata.get("mode", "")),
        str(metadata.get("profile", "")),
    ]
    return tuple(candidate for candidate in candidates if candidate)


def _hook_payload(record: HookRecord) -> str:
    """构建发送给外部进程的脱敏 JSON。"""
    payload = {
        "event": record.event,
        "tool": record.tool,
        "input": record.input,
        "output": record.output,
        "error": record.error,
        "metadata": record.metadata or {},
        "timestamp": record.timestamp,
        "session_id": record.session_id,
        "turn_id": record.turn_id,
        "request_id": record.request_id,
        "tool_call_id": record.tool_call_id,
    }
    return json.dumps(_redacted_json(payload), ensure_ascii=False)


def _redacted_json(value: object) -> JsonValue:
    """递归脱敏 JSON 边界值。"""
    if value is None or isinstance(value, int | float | bool):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list | tuple):
        return [_redacted_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redacted_json(item) for key, item in value.items()}
    return redact_text(value)


def _parse_hook_response(stdout: str) -> dict[str, JsonValue]:
    """解析单个 JSON object 响应。"""
    if len(stdout) > MAX_HOOK_STDOUT_CHARS:
        raise ValueError(f"hook stdout exceeds {MAX_HOOK_STDOUT_CHARS} characters")
    text = stdout.strip()
    if not text:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"hook stdout is invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("hook stdout must be a JSON object")
    return {str(key): _json_value(item) for key, item in decoded.items()}


def _validate_hook_response(
    event: str,
    response: dict[str, JsonValue],
) -> None:
    """校验事件允许的响应形状。"""
    if event != "pre_tool" or not response:
        return
    allowed_keys = {"decision", "arguments"}
    unknown_keys = set(response) - allowed_keys
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"pre_tool hook returned unsupported fields: {names}")
    decision = response.get("decision")
    if decision is not None and decision not in {"allow", "deny", "ask"}:
        raise ValueError("pre_tool hook decision must be allow, deny, or ask")
    arguments = response.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        raise ValueError("pre_tool hook arguments must be a JSON object")


def _json_value(value: object) -> JsonValue:
    """规范化外部 hook 返回的 JSON 值。"""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _process_error(returncode: int, stderr: str) -> str:
    """构建非零退出诊断。"""
    detail = stderr.strip()
    if detail:
        return f"hook exited with code {returncode}: {detail}"
    return f"hook exited with code {returncode}"
