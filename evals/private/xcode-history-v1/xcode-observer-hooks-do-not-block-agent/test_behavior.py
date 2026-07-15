"""Agent 不可见的外部 observer 后台调度行为 oracle。"""

from pathlib import Path
import threading

from xcode.harness.assembly import _build_hook_manager
from xcode.harness.observability.hooks import HookRecord


class _BlockingRunner:
    def __init__(self, *, fail: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    def execute(
        self,
        _record: HookRecord,
        *,
        subagent: bool,
        cwd: Path,
    ) -> tuple[()]:
        del subagent, cwd
        self.started.set()
        self.release.wait(timeout=2)
        if self.fail:
            raise RuntimeError("observer failed")
        return ()


def _manager(runner: _BlockingRunner):
    manager = _build_hook_manager(None, runner, Path.cwd(), subagent=False)
    assert manager is not None
    return manager


def test_external_observer_does_not_block_event_emitter() -> None:
    runner = _BlockingRunner()
    manager = _manager(runner)
    emit_finished = threading.Event()

    caller = threading.Thread(
        target=lambda: (manager.emit(HookRecord("post_tool")), emit_finished.set())
    )
    caller.start()
    assert runner.started.wait(timeout=1)
    returned_before_observer_release = emit_finished.wait(timeout=0.25)
    runner.release.set()
    caller.join(timeout=1)
    manager.drain_background()

    assert returned_before_observer_release
    assert emit_finished.is_set()


def test_external_observer_failure_is_isolated_from_emitter() -> None:
    runner = _BlockingRunner(fail=True)
    runner.release.set()
    manager = _manager(runner)

    manager.emit(HookRecord("on_error"))
    manager.drain_background()

    assert runner.started.is_set()
