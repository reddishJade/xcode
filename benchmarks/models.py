"""长程任务清单的数据模型与校验。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal, cast

StateCheckKind = Literal[
    "command_succeeds",
    "file_changed",
    "file_contains",
    "file_not_contains",
    "file_unchanged",
    "path_absent",
]

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CommandSpec:
    """无需 shell 解析的验证命令。"""

    argv: tuple[str, ...]
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class TurnSpec:
    """一个可重复的用户交互轮次。"""

    prompt: str
    compact_before: bool = False
    restart_after: bool = False


@dataclass(frozen=True)
class StateCheckSpec:
    """压缩或恢复后的确定性状态事实。"""

    id: str
    kind: StateCheckKind
    path: str | None = None
    value: str | None = None
    command: CommandSpec | None = None


@dataclass(frozen=True)
class CompactionSpec:
    """任务使用的压缩尾部预算。"""

    max_recent_messages: int = 6
    keep_recent_tokens: int = 4_000


@dataclass(frozen=True)
class LongHorizonTask:
    """一个长程代码任务及其机器判定规则。"""

    schema_version: int
    id: str
    description: str
    manifest_path: Path
    workspace: Path
    turns: tuple[TurnSpec, ...]
    compaction: CompactionSpec
    success_command: CommandSpec
    state_checks: tuple[StateCheckSpec, ...]


def load_task(path: Path) -> LongHorizonTask:
    """读取并严格校验一个任务清单。"""
    manifest_path = path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = _mapping(payload, "task manifest")
    _reject_unknown(
        data,
        {
            "schema_version",
            "id",
            "description",
            "workspace",
            "compaction",
            "turns",
            "success_command",
            "state_checks",
        },
        "task manifest",
    )
    schema_version = _integer(data.get("schema_version", 1), "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    task_id = _text(data.get("id"), "id")
    if not _SAFE_TASK_ID.fullmatch(task_id):
        raise ValueError(f"invalid task id: {task_id!r}")

    workspace_value = _text(data.get("workspace"), "workspace")
    workspace = _resolve_relative(manifest_path.parent, workspace_value, "workspace")
    if not workspace.is_dir():
        raise ValueError(f"task workspace does not exist: {workspace}")

    raw_turns = _list(data.get("turns"), "turns")
    if not raw_turns:
        raise ValueError("turns must not be empty")
    turns = tuple(_load_turn(item, index) for index, item in enumerate(raw_turns, 1))

    success_command = _load_command(data.get("success_command"), "success_command")
    raw_checks = _list(data.get("state_checks", []), "state_checks")
    state_checks = tuple(
        _load_state_check(item, index) for index, item in enumerate(raw_checks, 1)
    )
    return LongHorizonTask(
        schema_version=schema_version,
        id=task_id,
        description=_text(data.get("description", task_id), "description"),
        manifest_path=manifest_path,
        workspace=workspace,
        turns=turns,
        compaction=_load_compaction(data.get("compaction", {})),
        success_command=success_command,
        state_checks=state_checks,
    )


def discover_task_files(paths: list[Path]) -> tuple[Path, ...]:
    """展开清单文件或目录，并返回稳定排序后的 task.json。"""
    discovered: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file():
            discovered.add(path)
            continue
        if path.is_dir():
            discovered.update(item.resolve() for item in path.rglob("task.json"))
            continue
        raise ValueError(f"task path does not exist: {path}")
    if not discovered:
        raise ValueError("no task manifests found")
    return tuple(sorted(discovered))


def _load_turn(value: object, index: int) -> TurnSpec:
    data = _mapping(value, f"turns[{index}]")
    _reject_unknown(
        data,
        {"prompt", "compact_before", "restart_after"},
        f"turns[{index}]",
    )
    return TurnSpec(
        prompt=_text(data.get("prompt"), f"turns[{index}].prompt"),
        compact_before=_boolean(
            data.get("compact_before", False), f"turns[{index}].compact_before"
        ),
        restart_after=_boolean(
            data.get("restart_after", False), f"turns[{index}].restart_after"
        ),
    )


def _load_state_check(value: object, index: int) -> StateCheckSpec:
    data = _mapping(value, f"state_checks[{index}]")
    _reject_unknown(
        data,
        {"id", "kind", "path", "value", "command"},
        f"state_checks[{index}]",
    )
    check_id = _text(data.get("id"), f"state_checks[{index}].id")
    kind = _text(data.get("kind"), f"state_checks[{index}].kind")
    allowed: set[str] = {
        "command_succeeds",
        "file_changed",
        "file_contains",
        "file_not_contains",
        "file_unchanged",
        "path_absent",
    }
    if kind not in allowed:
        raise ValueError(f"unsupported state check kind: {kind}")
    typed_kind = cast(StateCheckKind, kind)
    path = data.get("path")
    value_text = data.get("value")
    command = data.get("command")
    if typed_kind == "command_succeeds":
        if command is None:
            raise ValueError(f"state check {check_id!r} requires command")
        return StateCheckSpec(
            id=check_id,
            kind=typed_kind,
            command=_load_command(command, f"state_checks[{index}].command"),
        )
    if path is None:
        raise ValueError(f"state check {check_id!r} requires path")
    normalized_path = _safe_relative_path(_text(path, f"state_checks[{index}].path"))
    if typed_kind in {"file_contains", "file_not_contains"} and value_text is None:
        raise ValueError(f"state check {check_id!r} requires value")
    return StateCheckSpec(
        id=check_id,
        kind=typed_kind,
        path=normalized_path,
        value=(
            _text(value_text, f"state_checks[{index}].value")
            if value_text is not None
            else None
        ),
    )


def _load_compaction(value: object) -> CompactionSpec:
    data = _mapping(value, "compaction")
    _reject_unknown(
        data,
        {"max_recent_messages", "keep_recent_tokens"},
        "compaction",
    )
    max_recent_messages = _positive_integer(
        data.get("max_recent_messages", 6), "compaction.max_recent_messages"
    )
    keep_recent_tokens = _positive_integer(
        data.get("keep_recent_tokens", 4_000), "compaction.keep_recent_tokens"
    )
    return CompactionSpec(
        max_recent_messages=max_recent_messages,
        keep_recent_tokens=keep_recent_tokens,
    )


def _load_command(value: object, field: str) -> CommandSpec:
    data = _mapping(value, field)
    _reject_unknown(data, {"argv", "timeout_seconds"}, field)
    raw_argv = _list(data.get("argv"), f"{field}.argv")
    argv = tuple(_text(item, f"{field}.argv") for item in raw_argv)
    if not argv:
        raise ValueError(f"{field}.argv must not be empty")
    timeout = data.get("timeout_seconds", 120)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError(f"{field}.timeout_seconds must be a number")
    if timeout <= 0:
        raise ValueError(f"{field}.timeout_seconds must be positive")
    return CommandSpec(argv=argv, timeout_seconds=float(timeout))


def _resolve_relative(root: Path, value: str, field: str) -> Path:
    relative = Path(_safe_relative_path(value))
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes task directory: {value!r}") from exc
    return resolved


def _safe_relative_path(value: str) -> str:
    normalized_value = value.replace("\\", "/")
    path = Path(normalized_value)
    if len(normalized_value) >= 2 and normalized_value[1] == ":":
        raise ValueError(f"path must stay relative to the task: {value!r}")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must stay relative to the task: {value!r}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("path must not be empty")
    return normalized


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_integer(value: object, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _reject_unknown(data: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}")
