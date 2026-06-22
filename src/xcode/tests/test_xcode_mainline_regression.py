"""主回归测试：端到端关键路径校验。

覆盖：
- 项目指令加载（project instruction load）
- 技能发现 + load_skill
- MCP 发现（使用 fixture 服务器，不依赖网络）
- 权限审批（permission approval）
- 文件编辑快照（file edit with snapshot）
- 审计记录（audit record）
- 撤销恢复（undo snapshot）
- 继续会话（continue session）
- 跨项目隔离（cross-project isolation）

每个测试独立且封闭：使用临时目录、fixture 服务器、mock home。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

from xcode.agent.context_collector import (
    ContextCollectionInput,
    InstructionCollector,
)
from xcode.harness.mcp import build_mcp_tools
from xcode.harness.observability import (
    AuditRecord,
    JsonlAuditLogger,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    StaticPermission,
)
from xcode.harness.session import SessionStore
from xcode.harness.skills_registry import (
    SkillRegistry,
    build_load_skill_tool,
    build_skill_search_dirs,
)
from xcode.harness.snapshot import SnapshotService, SnapshotStore
import pytest

FAKE_MCP_SERVER_CODE = r"""
import sys, json
def _read():
    l = sys.stdin.buffer.readline()
    return None if not l else json.loads(l.decode("utf-8").strip())
def _write(m):
    sys.stdout.buffer.write((json.dumps(m) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
def main():
    while True:
        req = _read()
        if req is None:
            break
        rid = req.get("id")
        method = req.get("method")
        if rid is not None:
            if method == "initialize":
                _write({"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"fixture-server","version":"1.0.0"}}})
            elif method == "tools/list":
                _write({"jsonrpc":"2.0","id":rid,"result":{"tools":[{"name":"echo","description":"Echo text","inputSchema":{"type":"object","properties":{"text":{"type":"string","description":"Text"}},"required":["text"]}}]}})
            elif method == "tools/call":
                args = req.get("params",{}).get("arguments",{})
                _write({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":args.get("text","")}],"isError":False}})
if __name__ == "__main__":
    main()
"""


class TestMainlineRegression:
    """主回归测试：覆盖所有关键集成路径。"""

    @classmethod
    def setup_class(cls) -> None:
        cls._home_tmp = tempfile.TemporaryDirectory()
        cls._home_patcher = mock.patch.object(
            Path, "home", return_value=Path(cls._home_tmp.name)
        )
        cls._home_patcher.start()

    @classmethod
    def teardown_class(cls) -> None:
        cls._home_patcher.stop()
        cls._home_tmp.cleanup()

    # ── 1. 项目指令加载 ──

    def test_instruction_load_from_agents_md(self) -> None:
        """AGENTS.md 中的项目指令可被 InstructionCollector 加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text(
                "# Project Instructions\n\nAlways use Python 3.12.\n",
                encoding="utf-8",
            )
            collector = InstructionCollector(project_root=root)
            blocks = collector.collect(ContextCollectionInput(project_root=root))
            assert blocks
            combined = " ".join(b.content for b in blocks)
            assert "Python 3.12" in combined

    # ── 2. 技能发现 + load_skill ──

    def test_skill_discovery_and_load_skill(self) -> None:
        """SkillRegistry 可发现技能，load_skill 可加载正文。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / ".xcode" / "skills" / "tester"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: tester\ndescription: Test skill.\n---\n\n# Tester Skill\n\nRun tests.\n",
                encoding="utf-8",
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            summaries = registry.list_summaries()
            assert len(summaries) == 1
            assert summaries[0].name == "tester"

            tool = build_load_skill_tool(registry)
            output = tool.handler({"name": "tester"})
            assert "Tester Skill" in output
            assert "Run tests." in output

    # ── 3. MCP 发现（fixture 服务器） ──

    def test_mcp_discovery_with_fixture_server(self) -> None:
        """MCP 工具可通过 fixture stdio 服务器发现和调用（不依赖网络）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_file = root / "fixture_mcp_server.py"
            server_file.write_text(FAKE_MCP_SERVER_CODE, encoding="utf-8")
            cmd = [sys.executable, "-u", str(server_file)]
            config_dir = root / ".local"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "mcp_config.json"
            config_path.write_text(
                json.dumps(
                    {"mcpServers": {"fixture": {"command": cmd[0], "args": cmd[1:]}}}
                ),
                encoding="utf-8",
            )
            tools = build_mcp_tools(root)
            tool_names = {t.name for t in tools}
            assert "mcp__fixture__echo" in tool_names

            echo_tool = next(t for t in tools if t.name == "mcp__fixture__echo")
            result = echo_tool.handler({"text": "hello regression"})
            assert result == "hello regression"

    # ── 4. 权限审批 ──

    def test_permission_approval_ask_allow_deny(self) -> None:
        """PermissionEngine 正确处理 ask/allow/deny 三种决策。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # deny
            deny_engine = PermissionEngine(
                PermissionEngineConfig(
                    static_policy=PermissionPolicy(
                        (StaticPermission("write_file", "deny"),)
                    ),
                    project_root=root,
                )
            )
            result = deny_engine.decide("write_file", {"path": "x.txt"})
            assert result.blocked

            # allow
            allow_engine = PermissionEngine(
                PermissionEngineConfig(
                    static_policy=PermissionPolicy(
                        (StaticPermission("read_file", "allow"),)
                    ),
                    project_root=root,
                )
            )
            result = allow_engine.decide("read_file", {"path": "x.txt"})
            assert not (result.blocked)

            # no policy → not blocked
            ask_engine = PermissionEngine(PermissionEngineConfig(project_root=root))
            result = ask_engine.decide("write_file", {"path": "x.txt"})
            assert not (result.blocked)

    # ── 5. 文件编辑快照 ──

    def test_file_edit_snapshot_and_restore(self) -> None:
        """SnapshotService 可跟踪文件变更并恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            test_file = root / "hello.txt"
            test_file.write_text("original content\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "hello.txt"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "initial"],
                cwd=root,
                capture_output=True,
                check=True,
            )

            svc = SnapshotService(root, "test-reg-5")
            pre = svc.track()

            test_file.write_text("modified content\n", encoding="utf-8")
            post = svc.track()

            changes = svc.diff(pre.snapshot_id, post.snapshot_id)
            assert any(c.path == "hello.txt" for c in changes)

            svc.restore_file(pre.snapshot_id, "hello.txt")
            restored = test_file.read_text(encoding="utf-8")
            assert "original content" in restored

    # ── 6. 审计记录 ──

    def test_audit_record_writing(self) -> None:
        """审计记录可写入 JSONL 并正确回读。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / ".local" / "audit.jsonl"
            logger = JsonlAuditLogger(log_path)
            record = AuditRecord(
                session_id="test-reg-6",
                tool="read_file",
                dynamic_decision="allow",
                policy_decision=None,
                final_status="approved",
                approved=True,
                redacted_input='{"path": "hello.txt"}',
                redacted_output="file content",
            )
            logger.write(record)

            assert log_path.exists()
            lines = log_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["tool"] == "read_file"
            assert data["session_id"] == "test-reg-6"
            assert data["approved"]

    # ── 7. 撤销快照恢复 ──

    def test_undo_snapshot_restore_flow(self) -> None:
        """完整撤销流程：track → edit → track → diff → restore → mark undone。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            test_file = root / "data.txt"
            test_file.write_text("v1\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "data.txt"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=root,
                capture_output=True,
                check=True,
            )

            store = SnapshotStore(root)
            svc = store.service("test-reg-7")
            pre = svc.track()

            test_file.write_text("v2\n", encoding="utf-8")
            post = svc.track()
            changes = svc.diff(pre.snapshot_id, post.snapshot_id)

            turn_id = store.next_turn_id("test-reg-7")
            store.record_turn(
                "test-reg-7",
                turn_id,
                pre.snapshot_id,
                post.snapshot_id,
                changes,
            )

            undoable = store.get_undoable_records("test-reg-7", 1)
            assert len(undoable) == 1
            rec = undoable[0]
            for change in rec.changed_files:
                svc.restore_file(rec.pre_snapshot_id, change.path)
            rec.undone = True
            store.update_record("test-reg-7", rec)

            restored = test_file.read_text(encoding="utf-8")
            assert "v1" in restored

            remaining = store.get_undoable_records("test-reg-7", 1)
            assert len(remaining) == 0

    # ── 8. 继续会话 ──

    def test_continue_session_resume(self) -> None:
        """SessionStore 支持写入记录后在新实例中恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions_dir = root / ".local" / "sessions"
            store = SessionStore(sessions_dir, project_root=root)
            store.append("user", "Hello from regression test")
            store.append("assistant", "Hello! How can I help?")

            saved_path = store.current_path

            store2 = SessionStore(sessions_dir, project_root=root)
            store2.resume(saved_path)
            records = store2.load_records()
            types = [r.type for r in records]
            assert "user" in types
            assert "assistant" in types

    # ── 9. 跨项目隔离 ──

    def test_cross_project_skill_isolation(self) -> None:
        """不同项目的技能发现互不泄漏。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj_a = root / "project_a"
            proj_b = root / "project_b"
            proj_a.mkdir()
            proj_b.mkdir()

            (proj_a / ".xcode" / "skills" / "review").mkdir(parents=True)
            (proj_a / ".xcode" / "skills" / "review" / "SKILL.md").write_text(
                "---\nname: review\ndescription: Review code.\n---\n\nReview body.",
                encoding="utf-8",
            )
            (proj_b / ".xcode" / "skills" / "deploy").mkdir(parents=True)
            (proj_b / ".xcode" / "skills" / "deploy" / "SKILL.md").write_text(
                "---\nname: deploy\ndescription: Deploy code.\n---\n\nDeploy body.",
                encoding="utf-8",
            )

            reg_a = SkillRegistry()
            reg_a.discover(build_skill_search_dirs(proj_a))
            names_a = {s.name for s in reg_a.list_summaries()}
            assert "review" in names_a
            assert "deploy" not in names_a

            reg_b = SkillRegistry()
            reg_b.discover(build_skill_search_dirs(proj_b))
            names_b = {s.name for s in reg_b.list_summaries()}
            assert "deploy" in names_b
            assert "review" not in names_b


if __name__ == "__main__":
    pytest.main()
