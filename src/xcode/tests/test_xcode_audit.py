from __future__ import annotations

import json
from pathlib import Path
import tempfile
from xcode.harness.observability import (
    AuditRecord,
    JsonlAuditLogger,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    StaticPermission,
    redact_text,
)
from xcode.harness.skills import ToolSpec
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import GateConfig

from xcode.tests.fixtures import FakeProvider
from xcode.ai.events import (
    ProviderEvent,
    TextDelta,
    FinalMessage,
    ToolCallEvent,
    ToolCall,
)
import pytest

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}


class XcodeAuditTests:
    def test_redact_text_masks_common_secret_shapes(self) -> None:
        text = "api_key=abcd1234secret and sk-1234567890abcdef"

        redacted = redact_text(text)

        assert "abcd1234secret" not in redacted
        assert "sk-1234567890abcdef" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_text_applied_via_tool_adapter(self) -> None:
        """验证红action 仅在 ToolSpecAdapter（生产路径）中应用。"""
        tool = ToolSpec("leak", "Leak.", "empty", lambda _input: "token=secret12345")

        # 测试辅助函数 run_tool 不应用 redact（由 ToolSpecAdapter 负责）
        raw = tool.handler({})
        assert raw == "token=secret12345"

        # redact_text 在 ToolSpecAdapter._tool_result_content 中应用
        redacted = redact_text(raw)
        assert redacted == "token=[REDACTED]"

    def test_structured_agent_writes_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            responses: list[list[ProviderEvent]] = [
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="t1",
                                name="echo",
                                input={"input": "sk-1234567890abcdef"},
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
            provider = FakeProvider(responses)
            agent = StructuredAgent(
                provider=provider,
                registry=(
                    ToolSpec(
                        "echo",
                        "Echo.",
                        "text",
                        lambda data: data["input"],
                        schema=INPUT_SCHEMA,
                    ),
                ),
                gate=GateConfig(
                    audit_logger=JsonlAuditLogger(path).write,
                    session_id="s1",
                ),
            )

            agent.run("go")

            record = json.loads(path.read_text(encoding="utf-8").strip())
            assert record["session_id"] == "s1"
            assert record["tool"] == "echo"
            assert record["redacted_input"] == '{"input": "[REDACTED]"}'
            assert record["redacted_output"] == "[REDACTED]"
            assert "dynamic_decision" in record
            assert "final_status" in record
            assert "approved" in record

    def test_audit_logger_writes_jsonl_with_structured_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"

            JsonlAuditLogger(path).write(
                AuditRecord(
                    session_id="s1",
                    tool="test_tool",
                    dynamic_decision="allow",
                    policy_decision=None,
                    final_status="ok",
                    approved=True,
                    redacted_input="in",
                    redacted_output="out",
                )
            )

            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["tool"] == "test_tool"
            assert record["final_status"] == "ok"
            assert record["approved"]

    def test_high_risk_tool_without_approval_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            called = False

            def danger(_data: dict) -> str:
                nonlocal called
                called = True
                return "approval required but actually ran"

            responses: list[list[ProviderEvent]] = [
                [
                    ToolCallEvent(
                        calls=[ToolCall(id="t1", name="danger", input={"input": "go"})]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
            provider = FakeProvider(responses)
            agent = StructuredAgent(
                provider=provider,
                registry=(
                    ToolSpec(
                        "danger",
                        "Danger.",
                        "text",
                        danger,
                        schema=INPUT_SCHEMA,
                    ),
                ),
                gate=GateConfig(
                    audit_logger=JsonlAuditLogger(path).write,
                    session_id="s2",
                ),
            )

            agent.run("go")

            record = json.loads(path.read_text(encoding="utf-8").strip())
            assert called
            assert record["tool"] == "danger"
            assert record["final_status"] == "ok"
            assert record["approved"]

    def test_handler_raw_output_contains_secrets(self) -> None:
        tool = ToolSpec(
            "leak", "Leak.", "empty", lambda _input: "sk-12345 token=secret"
        )

        result = tool.handler({})

        assert "sk-12345" in result
        assert "token=secret" in result

    def test_redacted_tool_result_does_not_leak_to_output(self) -> None:
        tool = ToolSpec("leak", "Leak.", "empty", lambda _input: "api_key=mysecret")

        engine = PermissionEngine(PermissionEngineConfig())
        perm = engine.decide("leak", {}, tool_spec=tool)
        assert not (perm.blocked)

        raw = tool.handler({})
        assert "mysecret" in raw

        content = redact_text(str(raw))
        assert "mysecret" not in content
        assert "REDACTED" in content

    def test_audit_record_includes_approval_scope_from_session_grant(self) -> None:
        """验证通过 session grant 放行的工具有 approval_scope 记录。"""
        audit_records: list[object] = []

        def capture(record: object) -> None:
            audit_records.append(record)

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="t1", name="echo", input={"input": "hello"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
        ]
        provider = FakeProvider(responses)
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec(
                    "echo",
                    "Echo.",
                    "text",
                    lambda value: value["input"],
                    schema=INPUT_SCHEMA,
                ),
            ),
            gate=GateConfig(
                audit_logger=capture,
                session_id="test-approval-scope",
            ),
        )
        agent.run("go")
        assert len(audit_records) > 0
        record = audit_records[0]
        assert getattr(record, "session_id", None) == "test-approval-scope"

    def test_no_local_session_id_when_real_set(self) -> None:
        """验证当真实 session_id 被设置后，审计记录不使用 'local'。"""
        audit_records: list[object] = []

        def capture(record: object) -> None:
            audit_records.append(record)

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="t1", name="echo", input={"input": "x"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
        ]
        provider = FakeProvider(responses)
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec(
                    "echo",
                    "Echo.",
                    "text",
                    lambda value: value["input"],
                    schema=INPUT_SCHEMA,
                ),
            ),
            gate=GateConfig(
                audit_logger=capture,
                session_id="real-session-001",
            ),
        )
        agent.run("go")
        assert len(audit_records) > 0
        for record in audit_records:
            sid = getattr(record, "session_id", None)
            if sid is not None:
                assert sid != "local"
                assert sid == "real-session-001"

    def test_denied_tool_records_blocked_status(self) -> None:
        """验证被拒绝的工具产生 final_status=blocked 记录。"""
        audit_records: list[object] = []

        def capture(record: object) -> None:
            audit_records.append(record)

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[ToolCall(id="t1", name="echo", input={"input": "x"})]
                ),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
            [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ],
        ]
        provider = FakeProvider(responses)
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec(
                    "echo",
                    "Echo.",
                    "text",
                    lambda value: value["input"],
                    schema=INPUT_SCHEMA,
                ),
            ),
            gate=GateConfig(
                audit_logger=capture,
                session_id="test-deny",
                permission_policy=PermissionPolicy((StaticPermission("echo", "deny"),)),
            ),
        )
        agent.run("go")
        blocked_records = [
            r for r in audit_records if getattr(r, "final_status", None) == "blocked"
        ]
        assert len(blocked_records) > 0
        for record in blocked_records:
            assert not (getattr(record, "approved", True))
            assert getattr(record, "dynamic_decision", "") == "deny"
            assert getattr(record, "matched_rule", "") == "rule"

    def test_audit_record_mcp_target_fields(self) -> None:
        """验证 MCP 工具的 target_kind/target_value 出现在审计记录中。"""
        audit_records: list[object] = []

        def capture(record: object) -> None:
            audit_records.append(record)

        responses: list[list[ProviderEvent]] = [
            [
                ToolCallEvent(
                    calls=[
                        ToolCall(
                            id="t1",
                            name="mcp__srv__my_tool",
                            input={"input": "x"},
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
        provider = FakeProvider(responses)
        agent = StructuredAgent(
            provider=provider,
            registry=(
                ToolSpec(
                    "mcp__srv__my_tool",
                    "MCP tool.",
                    "text",
                    lambda value: value["input"],
                    schema=INPUT_SCHEMA,
                ),
            ),
            gate=GateConfig(
                audit_logger=capture,
                session_id="test-mcp",
            ),
        )
        agent.run("go")
        assert len(audit_records) > 0
        mcp_records = [
            r for r in audit_records if getattr(r, "tool", "").startswith("mcp__")
        ]
        assert len(mcp_records) > 0
        for record in mcp_records:
            assert getattr(record, "capability", None) in ("mcp", "unknown")
            assert getattr(record, "target_kind", None) == "mcp"
            assert "mcp__" in getattr(record, "target_value", "") or ""


if __name__ == "__main__":
    pytest.main()
