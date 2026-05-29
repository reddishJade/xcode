from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from xcode.evals.eval_harness import EvaluationRunner


class TestEvaluationHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.keep_env_orig = os.environ.get("XCODE_KEEP_EVAL_SANDBOX")

    def tearDown(self) -> None:
        if self.keep_env_orig is not None:
            os.environ["XCODE_KEEP_EVAL_SANDBOX"] = self.keep_env_orig
        elif "XCODE_KEEP_EVAL_SANDBOX" in os.environ:
            del os.environ["XCODE_KEEP_EVAL_SANDBOX"]

        # Clean up any residual sandboxes created during tests
        sandboxes_dir = Path.cwd() / ".local" / "eval_sandboxes"
        if sandboxes_dir.exists():
            shutil.rmtree(sandboxes_dir, ignore_errors=True)

    def test_runner_runs_successful_evaluation_happy_path(self) -> None:
        runner = EvaluationRunner(keep_sandbox=False)
        result = runner.run()

        self.assertTrue(result.success, "Happy path evaluation should succeed")
        self.assertFalse(
            result.sandbox_dir.exists(), "Sandbox should be deleted by default"
        )
        self.assertEqual(
            len(result.assertions), 6, "Should execute exactly 6 assertions"
        )
        for assertion in result.assertions:
            self.assertTrue(
                assertion.passed, f"Assertion for '{assertion.tool_name}' should pass"
            )

    def test_sandbox_preservation_by_env_var(self) -> None:
        os.environ["XCODE_KEEP_EVAL_SANDBOX"] = "1"
        runner = EvaluationRunner()
        result = runner.run()

        try:
            self.assertTrue(result.success)
            self.assertTrue(
                result.sandbox_dir.exists(),
                "Sandbox should be preserved when env var is 1",
            )
        finally:
            shutil.rmtree(result.sandbox_dir, ignore_errors=True)

    def test_runner_reports_failure_on_tool_assertion_mismatch(self) -> None:
        # Mock run_tool_result to return a failure status to force an assertion mismatch
        from xcode.harness.skills import ToolExecutionResult

        with patch("xcode.evals.eval_harness.run_tool_result") as mock_run:
            mock_run.return_value = ToolExecutionResult(
                status="error", content="something went wrong"
            )
            runner = EvaluationRunner(keep_sandbox=False)
            result = runner.run()

            self.assertFalse(
                result.success, "Should fail when a tool returns an error status"
            )
            self.assertEqual(len(result.assertions), 6)
            for assertion in result.assertions:
                self.assertFalse(
                    assertion.passed, "All assertions should fail due to mocked error"
                )
                self.assertIn("Status Mismatch", assertion.details)


if __name__ == "__main__":
    unittest.main()
