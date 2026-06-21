"""外部命令 hook 隔离执行测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from xcode.harness.config import (
    ExternalHookRuntimeConfig,
    HookEventName,
    HookFailurePolicy,
)
from xcode.harness.observability.external_hooks import (
    ExternalHookFailure,
    ExternalHookRunner,
)
from xcode.harness.observability.hooks import HookRecord


def _write_hook(root: Path, source: str) -> Path:
    """写入单个临时 hook 脚本。"""
    path = root / "hook.py"
    path.write_text(source, encoding="utf-8")
    return path


def _entry(
    script: Path,
    *,
    event: HookEventName = "pre_tool",
    matcher: str | None = None,
    timeout: float = 1.0,
    failure_policy: HookFailurePolicy = "warn",
    inherit_to_subagents: bool = False,
) -> ExternalHookRuntimeConfig:
    """构建测试用 hook 声明。"""
    return ExternalHookRuntimeConfig(
        event=event,
        command=(sys.executable, str(script)),
        matcher=matcher,
        timeout=timeout,
        failure_policy=failure_policy,
        inherit_to_subagents=inherit_to_subagents,
        source="test-config.json",
    )


class ExternalHookRunnerTests(unittest.TestCase):
    """验证 JSON 进程边界、失败策略和诊断状态。"""

    def test_executes_json_hook_and_redacts_input(self) -> None:
        """stdin 敏感字段先脱敏，stdout JSON object 被保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(
                root,
                (
                    "import json, sys\n"
                    "payload = json.load(sys.stdin)\n"
                    "json.dump({'seen': payload['input']}, sys.stdout)\n"
                ),
            )
            runner = ExternalHookRunner((_entry(script),), root)

            (execution,) = runner.execute(
                HookRecord(
                    "pre_tool",
                    tool="bash",
                    input="token=supersecret",
                )
            )

        self.assertEqual(execution.status, "succeeded")
        self.assertEqual(execution.response["seen"], "token=[REDACTED]")
        diagnostic = runner.diagnostics()[0]
        self.assertEqual(diagnostic.run_count, 1)
        self.assertEqual(diagnostic.last_status, "succeeded")
        self.assertEqual(diagnostic.source, "test-config.json")

    def test_timeout_is_recorded_and_warned(self) -> None:
        """超时按 warn 策略记录但不抛出。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(
                root,
                "import time\ntime.sleep(1)\n",
            )
            runner = ExternalHookRunner(
                (_entry(script, timeout=0.01),),
                root,
            )

            with self.assertLogs(
                "xcode.harness.observability.external_hooks",
                level="WARNING",
            ):
                (execution,) = runner.execute(HookRecord("pre_tool", tool="bash"))

        self.assertEqual(execution.status, "failed")
        self.assertIn("timed out", execution.error)
        self.assertIn("timed out", runner.diagnostics()[0].last_error)

    def test_nonzero_exit_redacts_diagnostics(self) -> None:
        """非零退出保留状态并脱敏 stderr。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(
                root,
                (
                    "import sys\n"
                    "sys.stderr.write('api_key=supersecret')\n"
                    "raise SystemExit(7)\n"
                ),
            )
            runner = ExternalHookRunner(
                (_entry(script, failure_policy="ignore"),),
                root,
            )

            (execution,) = runner.execute(HookRecord("pre_tool", tool="bash"))

        self.assertEqual(execution.status, "failed")
        self.assertIn("code 7", execution.error)
        self.assertIn("api_key=[REDACTED]", execution.error)
        self.assertNotIn("supersecret", execution.error)

    def test_invalid_json_obeys_fail_policy(self) -> None:
        """fail 策略把无效 JSON 转换为 ExternalHookFailure。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(root, "print('not-json')\n")
            runner = ExternalHookRunner(
                (_entry(script, failure_policy="fail"),),
                root,
            )

            with self.assertRaisesRegex(ExternalHookFailure, "invalid JSON"):
                runner.execute(HookRecord("pre_tool", tool="bash"))

        diagnostic = runner.diagnostics()[0]
        self.assertEqual(diagnostic.last_status, "failed")
        self.assertIn("invalid JSON", diagnostic.last_error)

    def test_matcher_and_subagent_inheritance_filter_execution(self) -> None:
        """matcher 与显式 subagent 继承必须同时满足。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(root, "print('{}')\n")
            entries = (
                _entry(script, matcher="read_*"),
                _entry(
                    script,
                    matcher="read_*",
                    inherit_to_subagents=True,
                ),
            )
            runner = ExternalHookRunner(entries, root)

            main_results = runner.execute(HookRecord("pre_tool", tool="read_file"))
            child_results = runner.execute(
                HookRecord("pre_tool", tool="read_file"),
                subagent=True,
            )
            unmatched = runner.execute(HookRecord("pre_tool", tool="bash"))

        self.assertEqual(len(main_results), 2)
        self.assertEqual(len(child_results), 1)
        self.assertEqual(unmatched, ())
        diagnostics = runner.diagnostics()
        self.assertEqual(diagnostics[0].run_count, 1)
        self.assertEqual(diagnostics[1].run_count, 2)


if __name__ == "__main__":
    unittest.main()
