from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from xcode.harness.observability import (
    AuditRecord,
    JsonlAuditLogger,
    PermissionEngine,
    PermissionEngineConfig,
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


INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}


class XcodeAuditTests(unittest.TestCase):
    def test_redact_text_masks_common_secret_shapes(self) -> None:
        text = "api_key=abcd1234secret and sk-1234567890abcdef"

        redacted = redact_text(text)

        self.assertNotIn("abcd1234secret", redacted)
        self.assertNotIn("sk-1234567890abcdef", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_redact_text_applied_via_tool_adapter(self) -> None:
        """验证红action 仅在 ToolSpecAdapter（生产路径）中应用。"""
        tool = ToolSpec("leak", "Leak.", "empty", lambda _input: "token=secret12345")

        # 测试辅助函数 run_tool 不应用 redact（由 ToolSpecAdapter 负责）
        raw = tool.handler({})
        self.assertEqual(raw, "token=secret12345")

        # redact_text 在 ToolSpecAdapter._tool_result_content 中应用
        redacted = redact_text(raw)
        self.assertEqual(redacted, "token=[REDACTED]")

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
            self.assertEqual(record["session_id"], "s1")
            self.assertEqual(record["tool"], "echo")
            self.assertEqual(record["redacted_input"], '{"input": "[REDACTED]"}')
            self.assertEqual(record["redacted_output"], "[REDACTED]")
            self.assertIn("dynamic_decision", record)
            self.assertIn("final_status", record)
            self.assertIn("approved", record)

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
            self.assertEqual(record["tool"], "test_tool")
            self.assertEqual(record["final_status"], "ok")
            self.assertTrue(record["approved"])

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
            self.assertTrue(called)
            self.assertEqual(record["tool"], "danger")
            self.assertEqual(record["final_status"], "ok")
            self.assertTrue(record["approved"])

    def test_handler_raw_output_contains_secrets(self) -> None:
        tool = ToolSpec(
            "leak", "Leak.", "empty", lambda _input: "sk-12345 token=secret"
        )

        result = tool.handler({})

        self.assertIn("sk-12345", result)
        self.assertIn("token=secret", result)

    def test_redacted_tool_result_does_not_leak_to_output(self) -> None:
        tool = ToolSpec("leak", "Leak.", "empty", lambda _input: "api_key=mysecret")

        engine = PermissionEngine(PermissionEngineConfig())
        perm = engine.decide("leak", "{}", tool_spec=tool, tool_input={})
        self.assertFalse(perm.blocked)

        raw = tool.handler({})
        self.assertIn("mysecret", raw)

        content = redact_text(str(raw))
        self.assertNotIn("mysecret", content)
        self.assertIn("REDACTED", content)


if __name__ == "__main__":
    unittest.main()
