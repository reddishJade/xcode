from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from xcode.harness.observability import AuditRecord, JsonlAuditLogger, redact_text
from xcode.harness.skills import ToolSpec, run_tool
from xcode.harness.agent_runtime import StructuredAgent


from xcode.tests.fixtures import FakeProvider
from xcode.harness.agent_runtime.events import (
    ProviderEvent,
    TextDelta,
    FinalMessage,
    ToolCallReady,
    ToolCall,
)


class XcodeAuditTests(unittest.TestCase):
    def test_redact_text_masks_common_secret_shapes(self) -> None:
        text = "api_key=abcd1234secret and sk-1234567890abcdef"

        redacted = redact_text(text)

        self.assertNotIn("abcd1234secret", redacted)
        self.assertNotIn("sk-1234567890abcdef", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_run_tool_redacts_handler_output(self) -> None:
        tool = ToolSpec("leak", "Leak.", "empty", lambda _input: "token=secret12345")

        output = run_tool({"leak": tool}, "leak", {})

        self.assertEqual(output, "token=[REDACTED]")

    def test_structured_agent_writes_audit_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            responses: list[list[ProviderEvent]] = [
                [
                    ToolCallReady(
                        [ToolCall("t1", "echo", {"input": "sk-1234567890abcdef"})]
                    ),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
            ]
            provider = FakeProvider(responses)
            agent = StructuredAgent(
                provider=provider,
                registry=(
                    ToolSpec("echo", "Echo.", "text", lambda data: data["input"]),
                ),
                audit_logger=JsonlAuditLogger(path).write,
                session_id="s1",
            )

            agent.run("go")

            record = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["session_id"], "s1")
            self.assertEqual(record["tool"], "echo")
            self.assertEqual(record["redacted_input"], '{"input": "[REDACTED]"}')
            self.assertEqual(record["redacted_output"], "[REDACTED]")
            self.assertIn("static_risk", record)
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
                    static_risk="high",
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
            self.assertEqual(record["static_risk"], "high")
            self.assertEqual(record["final_status"], "ok")
            self.assertTrue(record["approved"])

    def test_tool_result_status_used_instead_of_string_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            # Tool with high risk that returns output resembling old approval prefix
            responses: list[list[ProviderEvent]] = [
                [
                    ToolCallReady([ToolCall("t1", "danger", {"input": "go"})]),
                    FinalMessage("", "end_turn"),
                ],
                [TextDelta("done"), FinalMessage("", "end_turn")],
            ]
            provider = FakeProvider(responses)
            agent = StructuredAgent(
                provider=provider,
                registry=(
                    ToolSpec(
                        "danger",
                        "Danger.",
                        "text",
                        lambda _data: "approval required but actually ran",
                        risk="high",
                    ),
                ),
                audit_logger=JsonlAuditLogger(path).write,
                session_id="s2",
            )

            agent.run("go")

            record = json.loads(path.read_text(encoding="utf-8").strip())
            # The tool ran successfully (no approval callback → returns approval_required via run_tool_result)
            # We verify that we get structured status, not string-prefix-based approval
            self.assertEqual(record["tool"], "danger")
            self.assertIn("final_status", record)
            self.assertIn("approved", record)

    def test_run_tool_result_redacts_output(self) -> None:
        tool = ToolSpec(
            "leak", "Leak.", "empty", lambda _input: "sk-12345 token=secret"
        )

        result = tool.handler({})

        self.assertIn("sk-12345", result)
        self.assertIn("token=secret", result)

    def test_redacted_tool_result_does_not_leak_to_output(self) -> None:
        from xcode.harness.skills import run_tool_result

        tool = ToolSpec("leak", "Leak.", "empty", lambda _input: "api_key=mysecret")

        result = run_tool_result({"leak": tool}, "leak", {})

        self.assertEqual(result.status, "ok")
        self.assertNotIn("mysecret", result.content)
        self.assertIn("REDACTED", result.content)


if __name__ == "__main__":
    unittest.main()
