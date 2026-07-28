"""压缩与恢复后的状态事实评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from benchmarks.evaluators.test_result import run_command
from benchmarks.models import StateCheckSpec

_MISSING = "<missing>"


@dataclass(frozen=True)
class StateCheckOutcome:
    """单个状态事实的判定结果。"""

    id: str
    kind: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def capture_initial_state(
    workspace: Path,
    checks: tuple[StateCheckSpec, ...],
) -> dict[str, str]:
    """为需要前后对比的路径保存初始哈希。"""
    snapshots: dict[str, str] = {}
    for check in checks:
        if check.kind not in {"file_changed", "file_unchanged"} or check.path is None:
            continue
        snapshots[check.path] = _file_hash(workspace / check.path)
    return snapshots


def evaluate_state_retention(
    workspace: Path,
    checks: tuple[StateCheckSpec, ...],
    initial_state: dict[str, str],
) -> tuple[StateCheckOutcome, ...]:
    """使用文件、约束和命令事实计算状态保持结果。"""
    outcomes: list[StateCheckOutcome] = []
    for check in checks:
        if check.kind == "command_succeeds":
            assert check.command is not None
            result = run_command(check.command, workspace)
            detail = f"returncode={result.returncode}, timed_out={result.timed_out}"
            outcomes.append(
                StateCheckOutcome(check.id, check.kind, result.passed, detail)
            )
            continue

        assert check.path is not None
        target = workspace / check.path
        if check.kind == "path_absent":
            outcomes.append(
                StateCheckOutcome(
                    check.id,
                    check.kind,
                    not target.exists(),
                    "path is absent" if not target.exists() else "path exists",
                )
            )
            continue
        if check.kind in {"file_changed", "file_unchanged"}:
            before = initial_state.get(check.path, _MISSING)
            after = _file_hash(target)
            passed = (
                after != before if check.kind == "file_changed" else after == before
            )
            outcomes.append(
                StateCheckOutcome(
                    check.id,
                    check.kind,
                    passed,
                    f"before={before}, after={after}",
                )
            )
            continue

        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            outcomes.append(
                StateCheckOutcome(check.id, check.kind, False, f"read failed: {exc}")
            )
            continue
        assert check.value is not None
        contains = check.value in content
        passed = contains if check.kind == "file_contains" else not contains
        outcomes.append(
            StateCheckOutcome(
                check.id,
                check.kind,
                passed,
                "expected text condition satisfied"
                if passed
                else "expected text condition failed",
            )
        )
    return tuple(outcomes)


def retention_rate(outcomes: tuple[StateCheckOutcome, ...]) -> float | None:
    """返回通过事实比例；没有事实时返回空值。"""
    if not outcomes:
        return None
    return sum(outcome.passed for outcome in outcomes) / len(outcomes)


def _file_hash(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError:
        return _MISSING
    return hashlib.sha256(content).hexdigest()
