"""外部命令 hook 隔离执行测试。"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from xcode.ai.events import FinalMessage, TextDelta, ToolCall, ToolCallEvent
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import GateConfig
from xcode.harness.assembly import _build_hook_manager
from xcode.harness.config import (
    ExternalHookRuntimeConfig,
    HookEventName,
    HookFailurePolicy,
)
from xcode.harness.observability import HITLResult, PermissionPolicy, StaticPermission
from xcode.harness.observability.external_hooks import (
    ExternalHookFailure,
    ExternalHookRunner,
)
from xcode.harness.observability.hooks import HookRecord
from xcode.harness.skills import ApprovalCallback, ToolSpec
from xcode.tests.fixtures import FakeProvider
import pytest
from xcode.tests._helpers import assert_logs

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}


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


class ExternalHookRunnerTests:
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
            runner = ExternalHookRunner(
                (_entry(script, event="post_tool"),),
                root,
            )

            (execution,) = runner.execute(
                HookRecord(
                    "post_tool",
                    tool="bash",
                    input="token=supersecret",
                )
            )

        assert execution.status == "succeeded"
        assert execution.response["seen"] == "token=[REDACTED]"
        diagnostic = runner.diagnostics()[0]
        assert diagnostic.run_count == 1
        assert diagnostic.last_status == "succeeded"
        assert diagnostic.source == "test-config.json"

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

            with assert_logs(
                "xcode.harness.observability.external_hooks",
                level="WARNING",
            ):
                (execution,) = runner.execute(HookRecord("pre_tool", tool="bash"))

        assert execution.status == "failed"
        assert "timed out" in execution.error
        assert "timed out" in runner.diagnostics()[0].last_error

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

        assert execution.status == "failed"
        assert "code 7" in execution.error
        assert "api_key=[REDACTED]" in execution.error
        assert "supersecret" not in execution.error

    def test_invalid_json_obeys_fail_policy(self) -> None:
        """fail 策略把无效 JSON 转换为 ExternalHookFailure。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(root, "print('not-json')\n")
            runner = ExternalHookRunner(
                (_entry(script, failure_policy="fail"),),
                root,
            )

            with pytest.raises(ExternalHookFailure, match="invalid JSON"):
                runner.execute(HookRecord("pre_tool", tool="bash"))

        diagnostic = runner.diagnostics()[0]
        assert diagnostic.last_status == "failed"
        assert "invalid JSON" in diagnostic.last_error

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

        assert len(main_results) == 2
        assert len(child_results) == 1
        assert unmatched == ()
        diagnostics = runner.diagnostics()
        assert diagnostics[0].run_count == 1
        assert diagnostics[1].run_count == 2

    def test_pre_tool_transforms_arguments_before_permission_and_execution(
        self,
    ) -> None:
        """参数变换会重新校验并沿 canonical 工具执行路径生效。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(
                root,
                (
                    "import json\n"
                    "print(json.dumps({'arguments': {'input': 'changed'}}))\n"
                ),
            )
            runner = ExternalHookRunner((_entry(script),), root)
            seen: list[str] = []

            def handler(value: dict[str, object]) -> str:
                seen.append(str(value["input"]))
                return str(value["input"])

            agent = _agent_with_tool(
                runner,
                handler,
            )

            result = agent.run("go")

        assert seen == ["changed"]
        assert "changed" in str(result.messages)

    def test_pre_tool_deny_blocks_handler(self) -> None:
        """外部 deny 只能收紧权限并阻止 handler。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(
                root,
                'print(\'{"decision":"deny"}\')\n',
            )
            runner = ExternalHookRunner((_entry(script),), root)
            called = False

            def handler(_value: dict[str, object]) -> str:
                nonlocal called
                called = True
                return "unexpected"

            result = _agent_with_tool(runner, handler).run("go")

        assert not (called)
        assert "deny" in str(result.messages).lower()

    def test_pre_tool_allow_cannot_override_static_deny(self) -> None:
        """外部 allow 不覆盖 PermissionEngine 的 deny。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(
                root,
                'print(\'{"decision":"allow"}\')\n',
            )
            runner = ExternalHookRunner((_entry(script),), root)
            called = False

            def handler(_value: dict[str, object]) -> str:
                nonlocal called
                called = True
                return "unexpected"

            agent = _agent_with_tool(
                runner,
                handler,
                permission_policy=PermissionPolicy(
                    (StaticPermission(tool="echo", decision="deny"),)
                ),
            )
            result = agent.run("go")

        assert not (called)
        assert "deny" in str(result.messages).lower()

    def test_pre_tool_ask_requires_normal_approval(self) -> None:
        """外部 ask 仍通过 PermissionEngine 的正常审批回调。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(
                root,
                'print(\'{"decision":"ask"}\')\n',
            )
            runner = ExternalHookRunner((_entry(script),), root)
            approvals: list[str] = []
            agent = _agent_with_tool(
                runner,
                lambda value: str(value["input"]),
                approval_callback=lambda tool, _input: (
                    approvals.append(tool.name) or HITLResult("allow", "once")
                ),
            )

            agent.run("go")

        assert approvals == ["echo"]

    def test_hook_manager_wires_all_non_pre_events(self) -> None:
        """装配层把其余五个事件转交给外部 runner。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "events.jsonl"
            script = _write_hook(
                root,
                (
                    "import json, pathlib, sys\n"
                    "payload = json.load(sys.stdin)\n"
                    f"path = pathlib.Path({str(log_path)!r})\n"
                    "with path.open('a', encoding='utf-8') as handle:\n"
                    "    handle.write(json.dumps(payload) + '\\n')\n"
                    "print('{}')\n"
                ),
            )
            events: tuple[HookEventName, ...] = (
                "post_tool",
                "on_error",
                "on_compact",
                "before_agent_start",
                "before_provider_request",
            )
            entries = tuple(_entry(script, event=event) for event in events)
            runner = ExternalHookRunner(entries, root)
            manager = _build_hook_manager(None, runner, root, subagent=False)
            assert manager is not None

            for event in events:
                manager.emit(HookRecord(event))

            recorded = [
                json.loads(line)["event"]
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        assert recorded == list(events)

    def test_subagent_hook_manager_runs_only_inherited_entries(self) -> None:
        """subagent manager 默认跳过未显式继承的 command hook。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = _write_hook(root, "print('{}')\n")
            runner = ExternalHookRunner(
                (
                    _entry(script, event="before_agent_start"),
                    _entry(
                        script,
                        event="before_agent_start",
                        inherit_to_subagents=True,
                    ),
                ),
                root,
            )
            manager = _build_hook_manager(None, runner, root, subagent=True)
            assert manager is not None

            manager.emit(HookRecord("before_agent_start"))

        diagnostics = runner.diagnostics()
        assert diagnostics[0].run_count == 0
        assert diagnostics[1].run_count == 1


def _agent_with_tool(
    runner: ExternalHookRunner,
    handler: Callable[[dict[str, object]], str],
    *,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: ApprovalCallback | None = None,
) -> StructuredAgent:
    """构建触发一次 echo 调用的测试 agent。"""
    responses = [
        [
            ToolCallEvent(
                calls=[
                    ToolCall(
                        id="call-1",
                        name="echo",
                        input={"input": "original"},
                    )
                ]
            ),
            FinalMessage(content="", stop_reason="end_turn"),
        ],
        [
            TextDelta(chunk="done"),
            FinalMessage(content="", stop_reason="end_turn"),
        ],
    ]
    return StructuredAgent(
        provider=FakeProvider(responses),
        registry=(
            ToolSpec(
                "echo",
                "Echo input.",
                "input",
                handler,
                schema=INPUT_SCHEMA,
            ),
        ),
        gate=GateConfig(
            approval_callback=approval_callback,
            permission_policy=permission_policy,
            external_hook_runner=runner,
        ),
    )


if __name__ == "__main__":
    pytest.main()
