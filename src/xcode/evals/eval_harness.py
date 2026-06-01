from __future__ import annotations

import os
import shutil
import uuid
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from xcode.harness.skills import run_tool_result, ToolSpec
from xcode.harness.tools import (
    build_file_tools,
    build_code_tools,
    build_bash_tool,
)
from xcode.harness.observability import PermissionPolicy, PermissionRule


@dataclass
class ToolAssertion:
    tool_name: str
    params: dict[str, Any] | str
    expected_status: str
    expected_content_contains: str
    actual_status: str = ""
    actual_content: str = ""
    passed: bool = False
    details: str = ""


@dataclass
class EvaluationResult:
    success: bool
    sandbox_dir: Path
    assertions: list[ToolAssertion] = field(default_factory=list)


class EvaluationRunner:
    def __init__(self, keep_sandbox: bool = False) -> None:
        self.keep_sandbox = keep_sandbox or (
            os.environ.get("XCODE_KEEP_EVAL_SANDBOX") == "1"
        )
        self.base_dir = Path.cwd() / ".local" / "eval_sandboxes"

    def run(self) -> EvaluationResult:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_uuid = uuid.uuid4().hex[:8]
        sandbox_dir = self.base_dir / f"{timestamp}_{run_uuid}"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Evaluation] Created unique sandbox workspace: {sandbox_dir}")

        assertions: list[ToolAssertion] = []
        success = True

        try:
            # Build tool registry inside sandbox root
            specs: list[ToolSpec] = []
            specs.extend(build_file_tools(sandbox_dir))
            specs.extend(build_code_tools(sandbox_dir))
            specs.append(build_bash_tool(sandbox_dir))

            registry = {spec.name: spec for spec in specs}

            # Scenario Step 1: write_file
            write_content = "hello world\n"
            step_write = ToolAssertion(
                tool_name="write_file",
                params={"path": "test.txt", "content": write_content},
                expected_status="ok",
                expected_content_contains="wrote file: test.txt",
            )
            assertions.append(step_write)

            # Scenario Step 2: read_file
            step_read = ToolAssertion(
                tool_name="read_file",
                params={"path": "test.txt"},
                expected_status="ok",
                expected_content_contains="hello world",
            )
            assertions.append(step_read)

            # Scenario Step 3: edit_file (Needs prior read_file to have fingerprinted test.txt!)
            step_edit = ToolAssertion(
                tool_name="edit_file",
                params={
                    "path": "test.txt",
                    "old_text": "hello world",
                    "new_text": "hello xcode",
                },
                expected_status="ok",
                expected_content_contains="test.txt",
            )
            assertions.append(step_edit)

            # Scenario Step 4: glob_files
            step_glob = ToolAssertion(
                tool_name="glob_files",
                params={"pattern": "*.txt"},
                expected_status="ok",
                expected_content_contains="test.txt",
            )
            assertions.append(step_glob)

            # Scenario Step 5: grep_search
            step_grep = ToolAssertion(
                tool_name="grep_search",
                params={"pattern": "hello"},
                expected_status="ok",
                expected_content_contains="test.txt",
            )
            assertions.append(step_grep)

            # Scenario Step 6: bash
            step_bash = ToolAssertion(
                tool_name="bash",
                params={"command": 'echo "hello from bash"'},
                expected_status="ok",
                expected_content_contains="hello from bash",
            )
            assertions.append(step_bash)

            # Define an allow-all policy to satisfy HITL risk validation on write/edit actions
            allow_all_policy = PermissionPolicy(
                rules=(PermissionRule(tool="*", decision="allow"),)
            )

            # Execute assertions sequentially
            for assertion in assertions:
                input_data = (
                    assertion.params if isinstance(assertion.params, dict) else {}
                )
                print(
                    f"[Harness] Dispatched tool '{assertion.tool_name}' with input: {input_data}"
                )

                # Execute tool using the main tool executor unified dispatch entrypoint
                res = run_tool_result(
                    registry,
                    assertion.tool_name,
                    input_data,
                    permission_policy=allow_all_policy,
                )

                assertion.actual_status = res.status
                assertion.actual_content = res.content

                # Assert status correctness
                status_ok = res.status == assertion.expected_status
                # Assert content inclusion correctness
                content_ok = assertion.expected_content_contains in res.content

                assertion.passed = status_ok and content_ok
                if not assertion.passed:
                    success = False
                    details = []
                    if not status_ok:
                        details.append(
                            f"Status Mismatch: expected '{assertion.expected_status}', got '{res.status}'"
                        )
                    if not content_ok:
                        details.append(
                            f"Content Mismatch: expected to contain '{assertion.expected_content_contains}', actual output: '{res.content}'"
                        )
                    assertion.details = " | ".join(details)
                    print(f"  --> [FAIL] {assertion.details}")
                else:
                    print("  --> [PASS]")

        except Exception as exc:
            success = False
            print(f"[Evaluation] Run encountered critical exception: {exc}")
            raise exc
        finally:
            if not self.keep_sandbox:
                print(f"[Evaluation] Cleaning up sandbox directory: {sandbox_dir}")
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            else:
                print(
                    f"[Evaluation] Keeping sandbox directory for debug: {sandbox_dir}"
                )

        return EvaluationResult(
            success=success, sandbox_dir=sandbox_dir, assertions=assertions
        )


def main() -> int:
    runner = EvaluationRunner()
    result = runner.run()

    print("\n" + "=" * 50)
    print("           EVALUATION HARNESS REPORT           ")
    print("=" * 50)
    print(f"Overall Result: {'SUCCESS' if result.success else 'FAILURE'}")
    print(f"Sandbox Location: {result.sandbox_dir}")
    print("-" * 50)

    for idx, assertion in enumerate(result.assertions, 1):
        try:
            status_icon = "✓" if assertion.passed else "✗"
            print(f"[{status_icon}] Step {idx}: {assertion.tool_name}")
        except UnicodeEncodeError:
            status_icon = "+" if assertion.passed else "-"
            print(f"[{status_icon}] Step {idx}: {assertion.tool_name}")
        print(f"    Params: {assertion.params}")
        print(
            f"    Expected: Status '{assertion.expected_status}', Containing '{assertion.expected_content_contains}'"
        )
        print(
            f"    Actual:   Status '{assertion.actual_status}', Content: {assertion.actual_content.strip()!r}"
        )
        if not assertion.passed:
            print(f"    Error:    {assertion.details}")
        print("-" * 50)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
