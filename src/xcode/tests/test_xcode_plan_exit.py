from __future__ import annotations

import tempfile
from pathlib import Path
import io
from contextlib import redirect_stdout

from xcode.cli.repl import run_repl
from xcode.harness.session import SessionStore
from xcode.tests.test_xcode_repl import FakePrompt
import pytest


class XcodePlanExitTests:
    def test_fork_clean_into_creates_empty_session_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            store.append("user", "test query")
            store.append("assistant", "test plan")

            parent_meta = store.current_metadata()
            assert parent_meta is not None
            assert parent_meta is not None

            fork_meta = store.fork_clean_into("isolate", title="Act continuation")
            assert fork_meta is not None
            assert fork_meta.parent_id == parent_meta.id
            assert fork_meta.fork_type == "isolate"
            assert fork_meta.title == "Act continuation"

            # The records in the new session should be completely empty (since it's a clean fork)
            records = store.load_records()
            assert len(records) == 0

    def test_repl_full_plan_exit_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup a Custom FakeApp that captures the prompts
            class PlanExitApp:
                def __init__(self) -> None:
                    self.prompts: list[str] = []

                def ask_stream(self, text: str, mode: str | None = None):
                    self.prompts.append(text)
                    from xcode.harness.agent_runtime import StructuredAgentResult
                    from xcode.harness.agent_runtime.events import (
                        FinalStructuredEvent,
                    )

                    yield FinalStructuredEvent(
                        "final",
                        1,
                        StructuredAgentResult(
                            answer="This is the generated plan.",
                            messages=[],
                            steps=1,
                            tool_calls=[],
                        ),
                    )

            app = PlanExitApp()
            # Send initial query: "generate plan"
            # Then run "/act --clear" (will trigger clear & fork and load "This is the generated plan.")
            # Then send: "execute plan"
            # Then exit.
            prompt = FakePrompt(
                ["generate plan", "/act --clear", "execute plan", "/exit"]
            )

            with redirect_stdout(io.StringIO()):
                code = run_repl(app, Path(temp_dir), prompt)

            assert code == 0
            # The first prompt sent to app: "generate plan"
            # The second prompt sent to app: should have <approved-plan>... prepended to "execute plan"
            assert len(app.prompts) == 2
            assert app.prompts[0] == "generate plan"
            assert (
                "<approved-plan>\nThis is the generated plan.\n</approved-plan>\nexecute plan"
                in app.prompts[1]
            )

            inspector = SessionStore(Path(temp_dir))
            clean_session = next(
                item
                for item in inspector.list_session_infos()
                if item.title.startswith("Act Continuation")
            )
            inspector.resume(clean_session.id)
            records = inspector.load_records()
            user_records = [record for record in records if record.type == "user"]
            assert user_records[-1].content == "execute plan"
            assert "<approved-plan>" not in str(user_records[-1].content)


if __name__ == "__main__":
    pytest.main()
