from __future__ import annotations

from pathlib import Path
import shutil

from .schema import EvalTask


class UnsafeEvalTaskError(ValueError):
    """真实 eval 任务缺少隔离工作目录时抛出的错误。"""


def trial_project_root(
    task: EvalTask,
    trial_index: int,
    *,
    base_root: Path,
    output_dir: Path,
    allow_project_mutation: bool = False,
) -> Path:
    """返回单个 trial 的项目根目录。

    有 fixture_dir 时复制 fixture 到 sandbox。没有 fixture_dir 时默认拒绝，
    避免真实 provider eval 意外修改当前仓库。
    """
    fixture_dir = task.metadata.get("fixture_dir")
    if fixture_dir:
        return copy_fixture_to_sandbox(
            fixture_dir=Path(str(fixture_dir)),
            task=task,
            trial_index=trial_index,
            base_root=base_root,
            output_dir=output_dir,
        )
    if allow_project_mutation:
        return base_root.resolve()
    raise UnsafeEvalTaskError(
        f"real eval task {task.id!r} has no fixture_dir; "
        "use a sandboxed fixture task or pass --allow-project-mutation"
    )


def copy_fixture_to_sandbox(
    *,
    fixture_dir: Path,
    task: EvalTask,
    trial_index: int,
    base_root: Path,
    output_dir: Path,
) -> Path:
    """复制 fixture 目录，返回该 trial 的隔离项目根目录。"""
    fixture_path = fixture_dir
    if not fixture_path.is_absolute():
        fixture_path = base_root / fixture_path
    if not fixture_path.is_dir():
        raise ValueError(f"fixture_dir is not a directory: {fixture_path}")
    sandbox = output_dir / "sandboxes" / f"{task.id}-{trial_index + 1}"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(fixture_path, sandbox)
    return sandbox.resolve()
