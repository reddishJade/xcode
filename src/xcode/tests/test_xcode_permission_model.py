from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Literal
import unittest

from xcode.harness.observability import (
    Action,
    ActionExtractor,
    ApprovalCandidate,
    BoundaryContext,
    Constraint,
    FileGrantStore,
    GrantRecord,
    HITLResult,
    InMemoryGrantStore,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionEngineResult,
    PermissionPolicy,
    PermissionResolver,
    StaticPermission,
    StaticPolicyEvaluator,
    evaluate_policy_constraints,
)
from xcode.harness.config import _validate_legacy_security_fields
from xcode.harness.observability._safety_backstop import (
    _is_dd_device_write,
    _is_root_recursive_deletion,
    _is_short_flag_with_r,
    _normalize_backslash_continuation,
    _split_compound_command,
)
from xcode.harness.observability.permission_model import (
    PermissionAccess,
    PolicyEvaluator,
)
from xcode.harness.skills import ToolSpec

# ── test helpers ──


def _grant(
    target_pattern: str,
    *,
    capability: str = "read",
    operation: str = "read_file",
    access: PermissionAccess = "read",
    decision: Literal["allow", "deny"] = "allow",
    grant_id: str | None = None,
) -> GrantRecord:
    return GrantRecord(
        capability=capability,
        operation=operation,
        target_kind="path",
        target_pattern=target_pattern,
        access=access,
        decision=decision,
        scope="session",
        grant_id=grant_id or f"grant:{target_pattern}:{decision}",
    )


class PermissionResolverTests(unittest.TestCase):
    def test_non_bypassable_deny_beats_everything(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("allow", "mode", "mode allows"),
                Constraint("ask", "boundary", "boundary asks"),
                Constraint("deny", "safety", "root delete", non_bypassable=True),
            )
        )

        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.source, "safety")
        self.assertEqual(verdict.reason, "root delete")

    def test_explicit_deny_beats_ask_and_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("allow", "mode", "mode allows"),
                Constraint("ask", "boundary", "boundary asks"),
                Constraint("deny", "rule", "rule denies"),
            )
        )

        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.source, "rule")

    def test_ask_beats_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("allow", "mode", "mode allows"),
                Constraint("ask", "boundary", "boundary asks"),
            )
        )

        self.assertEqual(verdict.decision, "ask")
        self.assertEqual(verdict.source, "boundary")

    def test_explicit_deny_beats_mixed_policy_list(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("ask", "boundary", "boundary asks"),
                Constraint("allow", "mode", "mode allows"),
                Constraint("deny", "rule", "rule denies"),
                Constraint("allow", "grant", "grant allows"),
            )
        )

        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.source, "rule")

    def test_multiple_allows_produce_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("allow", "mode", "mode allows"),
                Constraint("allow", "boundary", "boundary allows"),
            )
        )

        self.assertEqual(verdict.decision, "allow")
        self.assertEqual(verdict.source, "mode")

    def test_no_constraints_produces_default_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(())

        self.assertEqual(verdict.decision, "allow")
        self.assertEqual(verdict.source, "resolver")
        self.assertIsNone(verdict.winning_constraint)

    def test_winning_constraint_metadata_is_preserved(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(
                    "ask",
                    "boundary",
                    "external path",
                    metadata={"target": "../outside.txt"},
                ),
                Constraint("allow", "mode", "mode allows"),
            )
        )

        self.assertEqual(verdict.source, "boundary")
        self.assertEqual(verdict.reason, "external path")
        self.assertEqual(verdict.metadata["target"], "../outside.txt")


class ActionExtractorTests(unittest.TestCase):
    def test_structured_read_file_extracts_path_target(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "./src/main.py"})

        self.assertEqual(action.capability, "read")
        self.assertEqual(action.operation, "read_file")
        self.assertEqual(action.targets[0].kind, "path")
        self.assertEqual(action.targets[0].value, "src/main.py")
        self.assertEqual(action.targets[0].access, "read")

    def test_structured_write_file_extracts_write_target(self) -> None:
        action = ActionExtractor().extract(
            "write_file", {"path": "docs/permission-model.md", "content": "..."}
        )

        self.assertEqual(action.capability, "write")
        self.assertEqual(action.operation, "write_file")
        self.assertEqual(action.targets[0].access, "write")

    def test_structured_edit_file_extracts_write_target(self) -> None:
        action = ActionExtractor().extract("edit_file", {"path": "src/app.py"})

        self.assertEqual(action.capability, "edit")
        self.assertEqual(action.operation, "edit_file")
        self.assertEqual(action.targets[0].value, "src/app.py")
        self.assertEqual(action.targets[0].access, "write")

    def test_apply_patch_extracts_known_paths(self) -> None:
        action = ActionExtractor().extract(
            "apply_patch", {"paths": ["src/a.py", "src/b.py"]}
        )

        self.assertEqual(action.capability, "patch")
        self.assertEqual(
            [target.value for target in action.targets], ["src/a.py", "src/b.py"]
        )

    def test_shell_extracts_command_targets_only(self) -> None:
        action = ActionExtractor().extract(
            "shell", {"commands": ["git status --short", "uv run ruff check src"]}
        )

        self.assertEqual(action.capability, "shell")
        self.assertEqual(action.operation, "run_command")
        self.assertEqual(
            [target.kind for target in action.targets], ["command", "command"]
        )
        self.assertEqual(
            [target.access for target in action.targets], ["execute", "execute"]
        )

    def test_bash_extracts_single_command_target_only(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git status --short"})

        self.assertEqual(action.capability, "shell")
        self.assertEqual(action.targets[0].kind, "command")
        self.assertEqual(action.targets[0].value, "git status --short")
        self.assertEqual(action.targets[0].access, "execute")


class StructuredBoundaryPolicyTests(unittest.TestCase):
    def _boundary_context(self, root: Path) -> BoundaryContext:
        return BoundaryContext(root)

    def test_workspace_internal_read_produces_allow_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "src/app.py"})

        constraints = evaluate_policy_constraints(action)

        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "allow")
        self.assertEqual(constraints[0].source, "boundary")
        self.assertEqual(constraints[0].access, "read")

    def test_workspace_internal_write_produces_allow_constraint(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})

        constraints = evaluate_policy_constraints(action)

        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "allow")
        self.assertEqual(constraints[0].access, "write")

    def test_env_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": ".env.local"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        self.assertEqual(verdict.decision, "deny")
        assert verdict.winning_constraint is not None
        self.assertFalse(verdict.winning_constraint.non_bypassable)
        self.assertIn("sensitive path", verdict.reason)

    def test_credential_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": ".ssh/id_rsa"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        self.assertEqual(verdict.decision, "deny")
        assert verdict.winning_constraint is not None
        self.assertFalse(verdict.winning_constraint.non_bypassable)
        self.assertIn("sensitive path", verdict.reason)

    def test_git_read_produces_bypassable_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": ".git/config"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        self.assertEqual(verdict.decision, "deny")
        assert verdict.winning_constraint is not None
        self.assertFalse(verdict.winning_constraint.non_bypassable)

    def test_git_write_produces_non_bypassable_deny_constraint(self) -> None:
        action = ActionExtractor().extract("edit_file", {"path": ".git/config"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        self.assertEqual(verdict.decision, "deny")
        assert verdict.winning_constraint is not None
        self.assertTrue(verdict.winning_constraint.non_bypassable)

    def test_external_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "../secret.txt"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        self.assertEqual(verdict.decision, "deny")
        self.assertIn("escapes workspace", verdict.reason)

    def test_existing_blocked_workspace_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract(
            "read_file", {"path": ".local/chroma_db/index"}
        )

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        self.assertEqual(verdict.decision, "deny")
        self.assertIn("workspace blocked path", verdict.reason)

    def test_symlink_escape_read_produces_bypassable_deny_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            outside = root.parent / f"{root.name}-outside.txt"
            outside.write_text("secret", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("read_file", {"path": "link.txt"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            self.assertEqual(verdict.decision, "deny")
            assert verdict.winning_constraint is not None
            self.assertFalse(verdict.winning_constraint.non_bypassable)
            self.assertIn("escapes workspace", verdict.reason)
            outside.unlink()

    def test_symlink_to_git_write_produces_non_bypassable_deny(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            git_dir = root / ".git"
            git_dir.mkdir()
            git_config = git_dir / "config"
            git_config.write_text("[core]\n", encoding="utf-8")
            links = root / "links"
            links.mkdir()
            try:
                (links / "gitconfig").symlink_to(git_config)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("edit_file", {"path": "links/gitconfig"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            self.assertEqual(verdict.decision, "deny")
            assert verdict.winning_constraint is not None
            self.assertTrue(verdict.winning_constraint.non_bypassable)
            self.assertIn("git metadata", verdict.reason)
            self.assertEqual(verdict.winning_constraint.target_pattern, ".git/config")

    def test_resolution_error_read_produces_bypassable_deny(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            loop = root / "loop"
            try:
                loop.symlink_to(loop)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("read_file", {"path": "loop"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            self.assertEqual(verdict.decision, "deny")
            assert verdict.winning_constraint is not None
            self.assertFalse(verdict.winning_constraint.non_bypassable)
            self.assertIn("cannot be resolved safely", verdict.reason)

    def test_resolution_error_write_produces_non_bypassable_deny(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            loop = root / "loop"
            try:
                loop.symlink_to(loop)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("edit_file", {"path": "loop"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            self.assertEqual(verdict.decision, "deny")
            assert verdict.winning_constraint is not None
            self.assertTrue(verdict.winning_constraint.non_bypassable)
            self.assertIn("cannot be resolved safely", verdict.reason)

    def test_mode_constraint_participates_in_resolver(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})

        verdict = PermissionResolver().resolve(
            evaluate_policy_constraints(action, execution_decision="deny")
        )

        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.source, "mode")

    def test_static_policy_constraint_uses_legacy_rule_decision(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "docs/a.md"})
        policy = PermissionPolicy(
            (
                StaticPermission("read_file", "allow"),
                StaticPermission("*", "deny"),
            )
        )

        verdict = PermissionResolver().resolve(
            evaluate_policy_constraints(
                action,
                static_policy=policy,
                action_input='{"path": "docs/a.md"}',
            )
        )

        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.source, "rule")
        assert verdict.winning_constraint is not None
        self.assertEqual(verdict.winning_constraint.operation, "read_file")
        self.assertEqual(verdict.winning_constraint.access, "read")
        self.assertEqual(verdict.winning_constraint.target_pattern, "docs/a.md")

    def test_static_policy_ask_beats_boundary_allow(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})
        policy = PermissionPolicy((StaticPermission("write_file", "ask"),))

        verdict = PermissionResolver().resolve(
            evaluate_policy_constraints(
                action,
                static_policy=policy,
                action_input='{"path": "docs/a.md"}',
            )
        )

        self.assertEqual(verdict.decision, "ask")
        self.assertEqual(verdict.source, "rule")

    def test_global_default_ask_when_no_rule_matches(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})
        policy = PermissionPolicy(
            rules=(StaticPermission("read_file", "allow"),),
            global_default="ask",
        )

        constraints = evaluate_policy_constraints(
            action,
            static_policy=policy,
            action_input='{"path": "docs/a.md"}',
        )
        verdict = PermissionResolver().resolve(constraints)

        self.assertEqual(
            [(constraint.source, constraint.decision) for constraint in constraints],
            [("rule", "ask"), ("boundary", "allow")],
        )
        self.assertEqual(verdict.decision, "ask")
        self.assertEqual(verdict.source, "rule")


class GrantStoreTests(unittest.TestCase):
    def _grant(
        self,
        target_pattern: str,
        *,
        decision: Literal["allow", "deny"] = "allow",
        grant_id: str | None = None,
    ) -> GrantRecord:
        return GrantRecord(
            capability="read",
            operation="read_file",
            target_kind="path",
            target_pattern=target_pattern,
            access="read",
            decision=decision,
            scope="session",
            grant_id=grant_id or f"grant:{target_pattern}:{decision}",
        )

    def _lookup_path(self, store: InMemoryGrantStore, path: str) -> GrantRecord | None:
        action = ActionExtractor().extract("read_file", {"path": path})
        return store.lookup(action, action.targets[0])

    def test_path_segment_aware_matching_examples(self) -> None:
        store = InMemoryGrantStore((self._grant("src/foo"),))

        self.assertIsNotNone(self._lookup_path(store, "src/foo"))
        self.assertIsNotNone(self._lookup_path(store, "src/foo/bar.py"))
        self.assertIsNone(self._lookup_path(store, "src/foobar.py"))

        file_store = InMemoryGrantStore((self._grant("src/foo.py"),))
        self.assertIsNone(self._lookup_path(file_store, "src/foo.py.bak"))

    def test_overlapping_deny_beats_allow(self) -> None:
        store = InMemoryGrantStore(
            (
                self._grant("src", decision="allow", grant_id="allow-src"),
                self._grant("src/foo", decision="deny", grant_id="deny-foo"),
            )
        )

        record = self._lookup_path(store, "src/foo/bar.py")

        assert record is not None
        self.assertEqual(record.decision, "deny")
        self.assertEqual(record.grant_id, "deny-foo")

    def test_file_grant_store_round_trips_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileGrantStore(Path(tmp) / ".local" / "approval_grants.json")
            store.add(self._grant("src/foo", grant_id="grant-1"))

            reloaded = FileGrantStore(Path(tmp) / ".local" / "approval_grants.json")
            record = self._lookup_path(
                InMemoryGrantStore(reloaded.records()),
                "src/foo/bar.py",
            )

            assert record is not None
            self.assertEqual(record.grant_id, "grant-1")


class PermissionEngineShadowTests(unittest.TestCase):
    def test_shadow_mode_exposes_action_verdict_and_diff_without_changing_result(
        self,
    ) -> None:
        tool = ToolSpec(
            name="danger",
            description="Dangerous sample tool.",
            input_hint="anything",
            handler=lambda _: "",
        )
        engine = PermissionEngine(
            PermissionEngineConfig(
                shadow_model_enabled=True,
            )
        )

        result = engine.decide(
            "danger",
            '{"input": "go"}',
            tool_spec=tool,
            tool_input={"input": "go"},
        )

        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertIsNotNone(result.shadow_action)
        self.assertIsNotNone(result.shadow_verdict)
        assert result.shadow_verdict is not None
        self.assertEqual(result.shadow_verdict.decision, "allow")
        self.assertIsNone(result.shadow_diff)

    def test_shadow_mode_uses_structured_boundary_constraints(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig(shadow_model_enabled=True))

        result = engine.decide(
            "read_file",
            '{"path": ".env"}',
            tool_input={"path": ".env"},
        )

        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")
        self.assertEqual(result.source, "boundary")
        assert result.shadow_verdict is not None
        self.assertEqual(result.shadow_verdict.decision, "deny")
        self.assertEqual(result.shadow_verdict.source, "boundary")
        self.assertIsNone(result.shadow_diff)

    def test_shadow_boundary_context_denies_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            outside = root.parent / f"{root.name}-outside.txt"
            outside.write_text("secret", encoding="utf-8")
            try:
                (root / "link.txt").symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            engine = PermissionEngine(
                PermissionEngineConfig(
                    shadow_model_enabled=True,
                    project_root=root,
                )
            )

            result = engine.decide(
                "read_file",
                '{"path": "link.txt"}',
                tool_input={"path": "link.txt"},
            )

            self.assertTrue(result.blocked)
            self.assertEqual(result.decision, "deny")
            self.assertEqual(result.source, "boundary")
            assert result.shadow_verdict is not None
            self.assertEqual(result.shadow_verdict.decision, "deny")
            self.assertIn("escapes workspace", result.shadow_verdict.reason)
            outside.unlink()

    def test_shadow_static_policy_constraints_do_not_change_legacy_result(
        self,
    ) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
                shadow_model_enabled=True,
            )
        )

        result = engine.decide(
            "read_file",
            '{"path": "docs/a.md"}',
            tool_input={"path": "docs/a.md"},
        )

        self.assertEqual(result.decision, "ask")
        assert result.shadow_verdict is not None
        self.assertEqual(
            [
                (constraint.source, constraint.decision)
                for constraint in result.shadow_verdict.constraints
            ],
            [("rule", "ask"), ("boundary", "allow")],
        )
        self.assertIsNone(result.shadow_diff)


class ShadowApprovalCandidateTests(unittest.TestCase):
    """PermissionEngine 的 shadow_approval_candidate 行为测试。

    全部在 shadow 模式下断言但不改变 result.decision/blocked/reason。

    注意：legacy 引擎的 _check_hitl_grants 也检查 session_policy/persistent_store，
    因此当这些中有匹配 grant 时 legacy result 也会返回 allow/deny。shadow 路径在
    这些 case 中仍计算 candidate（因为 shadow verdict 为 ask），只是 legacy result
    不是 ask。我们在这些 case 中断言 shadow 路径的预测与 legacy 路径的差异。
    """

    # ── helpers ──

    def _engine(
        self,
        *,
        static_policy: PermissionPolicy | None = None,
        session_grant_store: InMemoryGrantStore | None = None,
        permanent_grant_store: InMemoryGrantStore | None = None,
    ) -> PermissionEngine:
        return PermissionEngine(
            PermissionEngineConfig(
                static_policy=static_policy,
                shadow_model_enabled=True,
                session_grant_store=session_grant_store,
                permanent_grant_store=permanent_grant_store,
            )
        )

    def _decide(
        self,
        engine: PermissionEngine,
        tool: str,
        tool_input: dict[str, object],
    ) -> tuple[PermissionEngineResult, ApprovalCandidate]:
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(tool, action_input, tool_input=tool_input)
        assert result.shadow_approval_candidate is not None, (
            f"expected shadow_approval_candidate for {tool}"
        )
        return result, result.shadow_approval_candidate

    def _assert_legacy_path_untouched(
        self,
        result: PermissionEngineResult,
        expected_decision: str,
        expected_blocked: bool,
    ) -> None:
        self.assertEqual(result.decision, expected_decision)
        self.assertEqual(result.blocked, expected_blocked)

    # ── positive: new-format session grant matching → allow ──

    def test_new_session_grant_allow(self) -> None:
        store = InMemoryGrantStore((_grant("src/foo.py"),))
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            session_grant_store=store,
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        # New resolver path: grant found → allow
        self._assert_legacy_path_untouched(result, "allow", False)
        self.assertEqual(candidate.would_resolve, "allow")
        self.assertEqual(len(candidate.fingerprints), 1)
        self.assertEqual(candidate.fingerprints[0].source, "new_session")
        assert candidate.fingerprints[0].grant is not None
        self.assertEqual(candidate.fingerprints[0].grant.decision, "allow")

    # ── positive: new-format permanent grant matching → deny ──

    def test_new_permanent_grant_deny(self) -> None:
        store = InMemoryGrantStore()
        store.add(_grant("src/foo.py", decision="deny"))
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            permanent_grant_store=store,
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        # New resolver path: deny grant found → deny
        self._assert_legacy_path_untouched(result, "deny", True)
        self.assertEqual(candidate.would_resolve, "deny")
        self.assertEqual(candidate.fingerprints[0].source, "new_permanent")

    # ── positive: no grants at all → would_call_approval ──

    def test_no_grants_produces_would_call_approval(self) -> None:
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        # No grants → callback required
        self._assert_legacy_path_untouched(result, "ask", True)
        self.assertEqual(candidate.would_resolve, "would_call_approval")
        self.assertEqual(candidate.fingerprints[0].source, "none")
        self.assertIsNone(candidate.fingerprints[0].grant)

    # ── positive: no grants at all → would_call_approval ──

    # ── multi-target apply_patch: one allow hit, one none → would_call_approval ──

    def test_multi_target_mixed_hit_produces_would_call_approval(self) -> None:
        store = InMemoryGrantStore(
            (
                _grant(
                    "src/allowed.py",
                    capability="patch",
                    operation="apply_patch",
                    access="write",
                ),
            )
        )
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("apply_patch", "ask"),)),
            session_grant_store=store,
        )
        tool_input: dict[str, object] = {
            "paths": ["src/allowed.py", "src/unknown.py"],
        }
        result, candidate = self._decide(engine, "apply_patch", tool_input)

        # Resolver: mixed → would_call_approval → ask
        self._assert_legacy_path_untouched(result, "ask", True)
        self.assertEqual(candidate.would_resolve, "would_call_approval")
        self.assertEqual(len(candidate.fingerprints), 2)
        # target A hits
        self.assertEqual(candidate.fingerprints[0].source, "new_session")
        self.assertIsNotNone(candidate.fingerprints[0].grant)
        # target B misses
        self.assertEqual(candidate.fingerprints[1].source, "none")
        self.assertIsNone(candidate.fingerprints[1].grant)

    # ── multi-target apply_patch: one deny → deny overrides everything ──

    def test_multi_target_deny_overrides_allow_hits(self) -> None:
        store = InMemoryGrantStore(
            (
                _grant(
                    "src/good.py",
                    capability="patch",
                    operation="apply_patch",
                    access="write",
                ),
                _grant(
                    "src/bad.py",
                    capability="patch",
                    operation="apply_patch",
                    access="write",
                    decision="deny",
                ),
            )
        )
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("apply_patch", "ask"),)),
            session_grant_store=store,
        )
        tool_input: dict[str, object] = {
            "paths": ["src/good.py", "src/bad.py"],
        }
        result, candidate = self._decide(engine, "apply_patch", tool_input)

        # Resolver: deny grant found → deny
        self._assert_legacy_path_untouched(result, "deny", True)
        self.assertEqual(candidate.would_resolve, "deny")
        self.assertEqual(len(candidate.fingerprints), 2)
        deny_grant = candidate.fingerprints[1].grant
        assert deny_grant is not None
        self.assertEqual(deny_grant.decision, "deny")

    # ── multi-target: all targets have allow → allow ──

    def test_multi_target_all_allow_produces_allow(self) -> None:
        store = InMemoryGrantStore(
            (
                _grant(
                    "src/a.py",
                    capability="patch",
                    operation="apply_patch",
                    access="write",
                ),
                _grant(
                    "src/b.py",
                    capability="patch",
                    operation="apply_patch",
                    access="write",
                ),
            )
        )
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("apply_patch", "ask"),)),
            session_grant_store=store,
        )
        tool_input: dict[str, object] = {
            "paths": ["src/a.py", "src/b.py"],
        }
        result, candidate = self._decide(engine, "apply_patch", tool_input)

        # Resolver: all targets have grants → allow
        self._assert_legacy_path_untouched(result, "allow", False)
        self.assertEqual(candidate.would_resolve, "allow")

    # ── edge: shadow_model_enabled=False → no candidate ──

    def test_shadow_disabled_no_candidate(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
                shadow_model_enabled=False,
            )
        )
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True),
            tool_input={"path": "src/foo.py"},
        )

        self.assertIsNone(result.shadow_approval_candidate)

    # ── edge: shadow verdict not ask → no candidate ──

    def test_shadow_verdict_allow_no_candidate(self) -> None:
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "allow"),)),
        )
        action_input = json.dumps(
            {"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True
        )
        result = engine.decide(
            "read_file", action_input, tool_input={"path": "src/foo.py"}
        )

        self.assertIsNone(result.shadow_approval_candidate)

    # ── edge: non-structured tool → no candidate ──

    def test_non_structured_tool_no_candidate(self) -> None:
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("bash", "ask"),)),
        )
        action_input = json.dumps(
            {"command": "echo hello"}, ensure_ascii=False, sort_keys=True
        )
        result = engine.decide(
            "bash", action_input, tool_input={"command": "echo hello"}
        )

        self.assertIsNone(result.shadow_approval_candidate)

    # ── fingerprint shape: matches expected fields ──

    def test_fingerprint_shape(self) -> None:
        store = InMemoryGrantStore((_grant("src/foo.py"),))
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            session_grant_store=store,
        )
        _, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        fp = candidate.fingerprints[0].fingerprint
        self.assertEqual(fp.capability, "read")
        self.assertEqual(fp.operation, "read_file")
        self.assertEqual(fp.target_kind, "path")
        self.assertEqual(fp.target_pattern, "src/foo.py")
        self.assertEqual(fp.access, "read")

    # ── edge: new session deny works ──

    def test_new_session_deny(self) -> None:
        store = InMemoryGrantStore((_grant("src/foo.py", decision="deny"),))
        engine = self._engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            session_grant_store=store,
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        self._assert_legacy_path_untouched(result, "deny", True)
        self.assertEqual(candidate.would_resolve, "deny")
        self.assertEqual(candidate.fingerprints[0].source, "new_session")
        deny_grant = candidate.fingerprints[0].grant
        assert deny_grant is not None
        self.assertEqual(deny_grant.decision, "deny")


class ApprovalCutoverEnabledTests(unittest.TestCase):
    """使用 fake/recording callback 验证授权路径的各分支正确性。"""

    class _RecordingCallback:
        """记录调用并返回预设 HITLResult 的 fake callback。"""

        def __init__(self, result: HITLResult = HITLResult("allow", "once")) -> None:
            self.calls: list[tuple[ToolSpec, dict[str, object]]] = []
            self.result = result

        def __call__(
            self, tool_spec: ToolSpec, tool_input: dict[str, object]
        ) -> HITLResult:
            self.calls.append((tool_spec, tool_input))
            return self.result

    @staticmethod
    def _tool(name: str) -> ToolSpec:
        return ToolSpec(name, name, "text", lambda v: str(v))

    def _cutover_engine(
        self,
        *,
        static_policy: PermissionPolicy | None = None,
        session_grant_store: InMemoryGrantStore | None = None,
        permanent_grant_store: InMemoryGrantStore | None = None,
    ) -> PermissionEngine:
        return PermissionEngine(
            PermissionEngineConfig(
                static_policy=static_policy,
                session_grant_store=session_grant_store,
                permanent_grant_store=permanent_grant_store,
            )
        )

    # ── deny 短接 ──

    def test_deny_verdict_short_circuits(self) -> None:
        """resolver 返回 deny → 短接，不查找授权，不回调。"""
        cb = self._RecordingCallback(HITLResult("allow", "once"))
        engine = self._cutover_engine()
        result = engine.decide(
            "read_file",
            json.dumps({"path": "../outside.txt"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "../outside.txt"},
            approval_callback=cb,
        )
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")
        self.assertEqual(len(cb.calls), 0)
        self.assertIsNone(result.approval_result)

    # ── allow 短接 ──

    def test_allow_verdict_short_circuits(self) -> None:
        """resolver 返回 allow → 短接，不查找授权，不回调。"""
        cb = self._RecordingCallback()
        engine = self._cutover_engine()
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/test.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "src/test.py"},
            approval_callback=cb,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertEqual(len(cb.calls), 0)
        self.assertIsNone(result.approval_result)

    # ── ask + 新格式授权命中 ──

    def test_ask_new_session_grant_hit(self) -> None:
        """新格式 session 授权命中 → 回调不被调用，授权直接决定。"""
        store = InMemoryGrantStore((_grant("src/foo.py"),))
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            session_grant_store=store,
        )
        cb = self._RecordingCallback()
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "src/foo.py"},
            approval_callback=cb,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertEqual(len(cb.calls), 0)
        assert result.approval_result is not None
        self.assertEqual(result.approval_result.decision, "allow")
        self.assertIsNotNone(result.approval_result.grant_id)

    def test_ask_new_permanent_grant_hit(self) -> None:
        """新格式 permanent 授权命中 → 回调不被调用，授权直接决定。"""
        store = InMemoryGrantStore((_grant("src/foo.py", decision="deny"),))
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            permanent_grant_store=store,
        )
        cb = self._RecordingCallback()
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "src/foo.py"},
            approval_callback=cb,
        )
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")
        self.assertEqual(len(cb.calls), 0)
        assert result.approval_result is not None
        self.assertEqual(result.approval_result.decision, "deny")

    # ── ask + 无授权 → 回调 ──

    def test_ask_no_grant_calls_callback(self) -> None:
        """无命中授权 → 以 (tool_spec, tool_input) 调用回调。"""
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
        )
        cb = self._RecordingCallback(HITLResult("allow", "once"))
        tool = self._tool("read_file")
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/unknown.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=tool,
            tool_input={"path": "src/unknown.py"},
            approval_callback=cb,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertEqual(len(cb.calls), 1)
        called_spec, called_input = cb.calls[0]
        self.assertIs(called_spec, tool)
        self.assertEqual(called_input, {"path": "src/unknown.py"})

    # ── 回调返回 allow/once ──

    def test_callback_allow_once_no_store_write(self) -> None:
        """allow/once → 放行，不写入任何 store。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "once"))
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "src/foo.py"},
            approval_callback=cb,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertEqual(len(store.records()), 0)

    # ── 回调返回 allow/session（单目标）──

    def test_callback_allow_session_writes_store(self) -> None:
        """allow/session（单目标）→ 放行，session store 写入 GrantRecord。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "session"))
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "src/foo.py"},
            approval_callback=cb,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertEqual(len(store.records()), 1)
        record = store.records()[0]
        self.assertEqual(record.decision, "allow")
        self.assertEqual(record.scope, "session")
        self.assertEqual(record.target_pattern, "src/foo.py")

    # ── 回调返回 allow/session（apply_patch 多目标）──

    def test_callback_allow_session_apply_patch_no_store_write(self) -> None:
        """allow/session（多目标）→ 放行但不写入 store，元数据记录限制。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("apply_patch", "ask"),)),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "session"))
        tool_input: dict[str, object] = {"paths": ["src/a.py", "src/b.py"]}
        result = engine.decide(
            "apply_patch",
            json.dumps(tool_input, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("apply_patch"),
            tool_input=tool_input,
            approval_callback=cb,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertEqual(len(store.records()), 0)
        assert result.metadata is not None
        self.assertTrue(result.metadata.get("multi_target_restriction"))
        self.assertEqual(result.metadata.get("requested_scope"), "session")
        self.assertEqual(result.metadata.get("effective_scope"), "once")

    # ── 回调返回 deny ──

    def test_callback_deny_no_store_write(self) -> None:
        """deny → 拒绝，不写入任何 store。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("deny", "once"))
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "src/foo.py"},
            approval_callback=cb,
        )
        self.assertTrue(result.blocked)
        self.assertEqual(result.decision, "deny")
        self.assertEqual(len(cb.calls), 1)
        self.assertEqual(len(store.records()), 0)

    # ── 回调返回 allow/permanent ──

    def test_callback_allow_permanent_writes_store(self) -> None:
        """allow/permanent → 放行，permanent store 写入 GrantRecord。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy((StaticPermission("read_file", "ask"),)),
            permanent_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "permanent"))
        result = engine.decide(
            "read_file",
            json.dumps({"path": "src/foo.py"}, ensure_ascii=False, sort_keys=True),
            tool_spec=self._tool("read_file"),
            tool_input={"path": "src/foo.py"},
            approval_callback=cb,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.decision, "allow")
        self.assertEqual(len(store.records()), 1)
        record = store.records()[0]
        self.assertEqual(record.scope, "permanent")
        self.assertEqual(record.target_pattern, "src/foo.py")


class SafetyBackstopHelperTests(unittest.TestCase):
    """SafetyBackstopPolicy 辅助函数的单元测试。"""

    # ── Bug 1: dd device write ──

    def test_dd_of_dev_is_device_write(self) -> None:
        self.assertTrue(_is_dd_device_write("dd if=/dev/zero of=/dev/sda"))

    def test_dd_of_backup_is_not_device_write(self) -> None:
        self.assertFalse(_is_dd_device_write("dd if=/dev/zero of=backup.img"))

    def test_dd_of_dev_nvme_is_device_write(self) -> None:
        self.assertTrue(_is_dd_device_write("dd if=/dev/random of=/dev/nvme0n1"))

    def test_non_dd_not_device_write(self) -> None:
        self.assertFalse(_is_dd_device_write("cat /dev/sda"))

    # ── Bug 2: flag-after-path root deletion ──

    def test_rm_root_with_flag_before(self) -> None:
        self.assertTrue(_is_root_recursive_deletion("rm -rf /"))

    def test_rm_root_with_flag_after(self) -> None:
        self.assertTrue(_is_root_recursive_deletion("rm / -rf"))

    def test_rm_root_glob_with_flag_after(self) -> None:
        self.assertTrue(_is_root_recursive_deletion("rm /* -rf"))

    def test_rm_root_flag_separate(self) -> None:
        self.assertTrue(_is_root_recursive_deletion("rm -r /"))

    def test_rm_non_root_not_recursive(self) -> None:
        self.assertFalse(_is_root_recursive_deletion("rm file.py"))

    def test_rm_non_root_with_flag(self) -> None:
        self.assertFalse(_is_root_recursive_deletion("rm -rf some_dir/"))

    # ── Bug 3: short flag with r ──

    def test_short_flag_rf_matches(self) -> None:
        self.assertTrue(_is_short_flag_with_r("-rf"))

    def test_short_flag_Rf_matches(self) -> None:
        self.assertTrue(_is_short_flag_with_r("-Rf"))

    def test_short_flag_r_matches(self) -> None:
        self.assertTrue(_is_short_flag_with_r("-r"))

    def test_long_flag_double_dash_does_not_match(self) -> None:
        self.assertFalse(_is_short_flag_with_r("--recursive"))

    def test_long_flag_version_does_not_match(self) -> None:
        self.assertFalse(_is_short_flag_with_r("-version"))

    def test_long_flag_format_does_not_match(self) -> None:
        self.assertFalse(_is_short_flag_with_r("-format"))

    def test_dr_still_matches_as_conservative(self) -> None:
        """-dr 是短 flag 组合且含 r，保守匹配可接受。"""
        self.assertTrue(_is_short_flag_with_r("-dr"))

    # ── Bug 5: backslash newline normalization ──

    def test_backslash_newline_normalized(self) -> None:
        raw = "git status \\\n  && rm -rf /"
        normalized = _normalize_backslash_continuation(raw)
        self.assertNotIn("\\\n", normalized)
        self.assertIn("   ", normalized)
        segments = _split_compound_command(normalized)
        self.assertEqual(len(segments), 2)
        self.assertIn("rm -rf /", segments)

    def test_backslash_newline_single_command(self) -> None:
        raw = "echo hello \\\n world"
        normalized = _normalize_backslash_continuation(raw)
        segments = _split_compound_command(normalized)
        self.assertEqual(len(segments), 1)

    def test_no_backslash_unchanged(self) -> None:
        raw = "git status && rm -rf /"
        self.assertEqual(_normalize_backslash_continuation(raw), raw)

    # ── compound split stability ──

    def test_compound_and_splits_correctly(self) -> None:
        segments = _split_compound_command("git status && rm -rf /")
        self.assertEqual(segments, ["git status", "rm -rf /"])

    def test_compound_pipe_splits_correctly(self) -> None:
        segments = _split_compound_command("ls | grep foo")
        self.assertEqual(segments, ["ls", "grep foo"])


class ShellCutoverTests(unittest.TestCase):
    """Shell 命令的统一 resolver 路径测试。"""

    def _engine(self) -> PermissionEngine:
        return PermissionEngine(PermissionEngineConfig())

    def _bash_tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name="bash",
            description="test",
            input_hint="",
            handler=lambda _: "",
        )

    def test_safety_backstop_ask_for_unknown(self) -> None:
        """未知命令 → SafetyBackstop 返回 ask。"""
        engine = self._engine()
        tool_input: dict[str, object] = {"command": "curl example.com"}
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(
            "bash",
            action_input,
            tool_input=tool_input,
            tool_spec=self._bash_tool_spec(),
        )
        self.assertEqual(result.decision, "ask")
        self.assertTrue(result.blocked)

    def test_bucket_a_deny_no_callback(self) -> None:
        """Bucket A 命令 → blocked=True, 不调用 callback。"""
        calls: list[object] = []

        def cb(_tool_spec: object, _tool_input: dict[str, object]) -> HITLResult:
            calls.append("called")
            return HITLResult(decision="allow", scope="once")

        engine = self._engine()
        tool_input: dict[str, object] = {"command": "rm -rf /"}
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(
            "bash",
            action_input,
            tool_input=tool_input,
            tool_spec=self._bash_tool_spec(),
            approval_callback=cb,
        )
        self.assertEqual(result.decision, "deny")
        self.assertTrue(result.blocked)
        self.assertEqual(len(calls), 0)

    def test_bucket_c_allow_no_callback(self) -> None:
        """Bucket C 命令 → allow, 不调用 callback。"""
        calls: list[object] = []

        def cb(_tool_spec: object, _tool_input: dict[str, object]) -> HITLResult:
            calls.append("called")
            return HITLResult(decision="allow", scope="once")

        engine = self._engine()
        tool_input: dict[str, object] = {"command": "git status"}
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(
            "bash",
            action_input,
            tool_input=tool_input,
            tool_spec=self._bash_tool_spec(),
            approval_callback=cb,
        )
        self.assertEqual(result.decision, "allow")
        self.assertFalse(result.blocked)
        self.assertEqual(len(calls), 0)

    def test_bucket_b_ask_calls_callback_once_scope(self) -> None:
        """Bucket B 命令 → callback 被调用, once scope, 无 grant store 写入。"""
        calls: list[HITLResult] = []

        def cb(_tool_spec: object, _tool_input: dict[str, object]) -> HITLResult:
            result = HITLResult(decision="allow", scope="once")
            calls.append(result)
            return result

        engine = self._engine()
        tool_input: dict[str, object] = {"command": "rm some_file.py"}
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(
            "bash",
            action_input,
            tool_input=tool_input,
            tool_spec=self._bash_tool_spec(),
            approval_callback=cb,
        )
        self.assertEqual(result.decision, "allow")
        self.assertFalse(result.blocked)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].scope, "once")

    def test_bucket_b_ask_callback_deny(self) -> None:
        """Bucket B 命令 + callback 返回 deny → blocked=True。"""

        def cb(_tool_spec: object, _tool_input: dict[str, object]) -> HITLResult:
            return HITLResult(decision="deny", scope="once")

        engine = self._engine()
        tool_input: dict[str, object] = {"command": "rm some_file.py"}
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(
            "bash",
            action_input,
            tool_input=tool_input,
            tool_spec=self._bash_tool_spec(),
            approval_callback=cb,
        )
        self.assertEqual(result.decision, "deny")
        self.assertTrue(result.blocked)

    def test_non_shell_tool_unaffected(self) -> None:
        engine = self._engine()
        tool_input: dict[str, object] = {"path": "some_file.py"}
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(
            "read_file",
            action_input,
            tool_input=tool_input,
            tool_spec=ToolSpec(
                name="read_file",
                description="test",
                input_hint="",
                handler=lambda _: "",
            ),
        )
        # 无 static policy → 默认 allow
        self.assertEqual(result.decision, "allow")
        self.assertFalse(result.blocked)

    def test_non_bypassable_deny_flagged_in_metadata(self) -> None:
        """non-bypassable deny 约束在 metadata 中标记。"""
        engine = self._engine()
        tool_input: dict[str, object] = {"command": "rm -rf /"}
        action_input = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        result = engine.decide(
            "bash",
            action_input,
            tool_input=tool_input,
            tool_spec=self._bash_tool_spec(),
        )
        self.assertEqual(result.decision, "deny")
        self.assertTrue(result.blocked)
        assert result.metadata is not None
        self.assertTrue(result.metadata.get("non_bypassable"))


# ── STEP 5 hook invariant test helpers ──


class _AlwaysAllowHook:
    """对所有 action 返回 allow 的测试 hook。"""

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        return (
            Constraint(
                decision="allow",
                source="hook:always_allow",
                reason="test hook allows everything",
            ),
        )


class _AlwaysDenyHook:
    """对所有 action 返回 deny 的测试 hook。"""

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        return (
            Constraint(
                decision="deny",
                source="hook:always_deny",
                reason="test hook denies everything",
            ),
        )


class HookInvariantTests(unittest.TestCase):
    """验证 hook 约束不变量 (STEP 5 第 11.2-11.4 节)。"""

    maxDiff = None

    def test_hook_allow_cannot_override_safety_backstop_deny(self) -> None:
        """Hook 返回 allow 不能覆盖 SafetyBackstop 的 non-bypassable deny。"""
        always_allow_hook: PolicyEvaluator = _AlwaysAllowHook()
        action = ActionExtractor().extract("bash", {"command": "rm -rf /"})

        constraints = evaluate_policy_constraints(
            action,
            execution_decision="allow",
            safety_backstop_enabled=True,
            hook_constraint_providers=(always_allow_hook,),
        )

        verdict = PermissionResolver().resolve(constraints)
        self.assertEqual(verdict.decision, "deny")
        assert verdict.winning_constraint is not None
        self.assertIs(verdict.winning_constraint.non_bypassable, True)
        self.assertEqual(verdict.winning_constraint.source, "safety_backstop")

    def test_hook_allow_cannot_override_static_deny(self) -> None:
        """Hook 返回 allow 不能覆盖 StaticPolicy 的 explicit deny。"""
        always_allow_hook: PolicyEvaluator = _AlwaysAllowHook()
        static_policy = PermissionPolicy(rules=(StaticPermission("bash", "deny"),))
        action = ActionExtractor().extract("bash", {"command": "echo hello"})

        constraints = evaluate_policy_constraints(
            action,
            execution_decision="allow",
            static_policy=static_policy,
            safety_backstop_enabled=True,
            hook_constraint_providers=(always_allow_hook,),
        )

        verdict = PermissionResolver().resolve(constraints)
        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.winning_constraint.source, "rule")

    def test_hook_allow_is_honored_when_no_other_deny(self) -> None:
        """无其他 evaluator 产生 deny 时，hook 的 allow 应被采纳。"""
        always_allow_hook: PolicyEvaluator = _AlwaysAllowHook()
        action = ActionExtractor().extract("bash", {"command": "echo hello"})

        constraints = evaluate_policy_constraints(
            action,
            execution_decision="allow",
            safety_backstop_enabled=True,
            hook_constraint_providers=(always_allow_hook,),
        )

        verdict = PermissionResolver().resolve(constraints)
        self.assertEqual(verdict.decision, "allow")

    def test_hook_deny_can_override_builtin_allow(self) -> None:
        """Hook 返回 deny 可以拒绝内置 evaluator 放行的 action。"""
        always_deny_hook: PolicyEvaluator = _AlwaysDenyHook()
        action = ActionExtractor().extract("bash", {"command": "echo hello"})

        constraints = evaluate_policy_constraints(
            action,
            execution_decision="allow",
            safety_backstop_enabled=True,
            hook_constraint_providers=(always_deny_hook,),
        )

        verdict = PermissionResolver().resolve(constraints)
        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.winning_constraint.source, "hook:always_deny")


class StaticPolicyLastMatchWinsTests(unittest.TestCase):
    """last-match-wins within StaticPolicyEvaluator static rule matching."""

    def test_last_match_wins_basic(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (
            StaticPermission("bash", "deny"),
            StaticPermission("bash", "allow"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "allow")
        self.assertEqual(constraints[0].source, "rule")

    def test_last_match_wins_reversed(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (
            StaticPermission("bash", "allow"),
            StaticPermission("bash", "deny"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "deny")

    def test_global_asterisk_rule(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (StaticPermission("*", "deny"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "deny")

    def test_input_regex_matches(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "curl example.com"})
        rules = (StaticPermission("bash", "ask", input_regex=r"curl|wget"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "ask")

    def test_input_regex_no_match(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "ls -la"})
        rules = (StaticPermission("bash", "ask", input_regex=r"curl|wget"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 0)

    def test_target_matches_action_target(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "src/foo.py"})
        rules = (StaticPermission("read_file", "allow", target="src/foo.py"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "allow")

    def test_target_no_match(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "src/other.py"})
        rules = (StaticPermission("read_file", "allow", target="src/foo.py"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 0)

    def test_target_type_matches_command(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git status"})
        rules = (StaticPermission("bash", "ask", target_type="command"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "ask")

    def test_target_type_no_match(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git status"})
        rules = (StaticPermission("bash", "ask", target_type="path"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 0)

    def test_global_default_applies_when_no_rule_matches(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (StaticPermission("read_file", "allow"),)
        evaluator = StaticPolicyEvaluator(rules, global_default="ask")
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].decision, "ask")
        self.assertIn("global_default", constraints[0].reason)

    def test_no_rules_no_global_default_emits_nothing(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        evaluator = StaticPolicyEvaluator()
        constraints = evaluator.evaluate(action)
        self.assertEqual(len(constraints), 0)

    def test_non_bypassable_deny_still_beats_static_allow(self) -> None:
        """PermissionResolver: non_bypassable deny > static allow."""
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("allow", "rule", "static allows"),
                Constraint(
                    "deny", "safety", "non-bypassable deny", non_bypassable=True
                ),
            )
        )
        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.source, "safety")

    def test_boundary_deny_still_beats_static_allow(self) -> None:
        """PermissionResolver: boundary deny > static allow."""
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("allow", "rule", "static allows"),
                Constraint("deny", "boundary", "boundary denies"),
            )
        )
        self.assertEqual(verdict.decision, "deny")
        self.assertEqual(verdict.source, "boundary")

    def test_static_ask_beats_boundary_allow(self) -> None:
        """PermissionResolver: static ask > boundary allow."""
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint("allow", "boundary", "boundary allows"),
                Constraint("ask", "rule", "static asks"),
            )
        )
        self.assertEqual(verdict.decision, "ask")
        self.assertEqual(verdict.source, "rule")


class LegacyPermissionFieldValidationTests(unittest.TestCase):
    """Config fail-fast on legacy deny_tools/ask_tools/allow_tools."""

    def _assert_raises(self, raw: dict) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_legacy_security_fields(raw)
        self.assertIn("Migrate to", str(ctx.exception))

    def test_legacy_deny_tools_raises(self) -> None:
        self._assert_raises({"security": {"deny_tools": ["bash"]}})

    def test_legacy_ask_tools_raises(self) -> None:
        self._assert_raises({"security": {"ask_tools": ["write_file"]}})

    def test_legacy_allow_tools_raises(self) -> None:
        self._assert_raises({"security": {"allow_tools": ["read_file"]}})

    def test_legacy_mixed_fields_raises(self) -> None:
        self._assert_raises(
            {
                "security": {
                    "deny_tools": ["bash"],
                    "ask_tools": ["write_file"],
                }
            }
        )

    def test_new_rules_field_does_not_raise(self) -> None:
        raw = {"security": {"rules": [{"tool": "bash", "decision": "deny"}]}}
        _validate_legacy_security_fields(raw)  # should not raise

    def test_no_security_does_not_raise(self) -> None:
        _validate_legacy_security_fields({})  # should not raise


if __name__ == "__main__":
    unittest.main()
