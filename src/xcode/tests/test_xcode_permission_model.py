from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Literal
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
    create_grant_record,
    evaluate_policy_constraints,
)
from pydantic import ValidationError
from xcode.harness.config import SecurityExternalDirectory, SecurityRuntimeConfig
from xcode.harness.observability.permission_model import (
    DirAccess,
    ExternalDirectory,
    PermissionAccess,
    PolicyEvaluator,
    Rule,
    SessionGrantStoreManager,
    Target,
)
from xcode.harness.skills import ToolSpec
import pytest
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


class RuleMatcherIntegrationTests:
    def test_action_profile_preserves_shell_unresolved_effects(self) -> None:
        action = ActionExtractor().extract(
            "bash",
            {"command": "cat *.py"},
            action_profile=("shell", "none"),
        )

        assert action.capability == "shell"
        assert action.unresolved_effects

    def test_structured_rule_requires_present_subcommand(self) -> None:
        from xcode.harness.observability.rule_matcher import evaluate as rule_evaluate

        action = Action(
            tool="bash",
            capability="shell",
            operation="run_command",
            targets=(Target(kind="command", value="git", access="execute"),),
            input={"command": "git"},
        )

        decision = rule_evaluate(
            action,
            (Rule(action="bash", command="git", subcommand="status", effect="allow"),),
            fallback="ask",
            shell_command="git",
        )

        assert decision == "ask"

    def test_user_ruleset_overrides_default_rules(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                mode_ruleset=(Rule(action="bash", effect="ask"),),
                user_ruleset=(
                    Rule(
                        action="bash",
                        command="rm",
                        flags_any={"-fr"},
                        effect="allow",
                    ),
                ),
                mode_fallback="ask",
                tool_action_profiles={"bash": ("shell", "none")},
            )
        )

        result = engine.decide("bash", {"command": "rm -rf tmp"})

        assert result.decision == "allow"
        assert not result.blocked


class CommandGrantPatternTests:
    def test_command_grant_generalizes_git_subcommand_argument(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git checkout main"})
        grant = create_grant_record(
            action,
            action.targets[0],
            decision="allow",
            scope="session",
        )

        assert grant.target_pattern == "git checkout *"

        store = InMemoryGrantStore((grant,))
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="bash", decision="ask"),)
                ),
                session_grant_store=store,
            )
        )

        result = engine.decide("bash", {"command": "git checkout feature"})

        assert result.decision == "allow"
        assert result.matched_rule == "session_grant"

    def test_command_grant_tail_wildcard_matches_bare_command(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git add README.md"})
        grant = create_grant_record(
            action,
            action.targets[0],
            decision="allow",
            scope="session",
        )
        store = InMemoryGrantStore((grant,))

        bare_action = ActionExtractor().extract("bash", {"command": "git add"})
        match = store.lookup(bare_action, bare_action.targets[0])

        assert match is grant


class PermissionResolverTests:
    def test_explicit_deny_beats_everything_even_when_late(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="allow", source="mode", reason="mode allows"),
                Constraint(decision="ask", source="boundary", reason="boundary asks"),
                Constraint(
                    decision="deny",
                    source="safety",
                    reason="root delete",
                ),
            )
        )

        assert verdict.decision == "deny"
        assert verdict.source == "safety"
        assert verdict.reason == "root delete"

    def test_explicit_deny_beats_ask_and_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="allow", source="mode", reason="mode allows"),
                Constraint(decision="ask", source="boundary", reason="boundary asks"),
                Constraint(decision="deny", source="rule", reason="rule denies"),
            )
        )

        assert verdict.decision == "deny"
        assert verdict.source == "rule"

    def test_ask_beats_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="allow", source="mode", reason="mode allows"),
                Constraint(decision="ask", source="boundary", reason="boundary asks"),
            )
        )

        assert verdict.decision == "ask"
        assert verdict.source == "boundary"

    def test_explicit_deny_beats_mixed_policy_list(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="ask", source="boundary", reason="boundary asks"),
                Constraint(decision="allow", source="mode", reason="mode allows"),
                Constraint(decision="deny", source="rule", reason="rule denies"),
                Constraint(decision="allow", source="grant", reason="grant allows"),
            )
        )

        assert verdict.decision == "deny"
        assert verdict.source == "rule"

    def test_multiple_allows_produce_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="allow", source="mode", reason="mode allows"),
                Constraint(
                    decision="allow", source="boundary", reason="boundary allows"
                ),
            )
        )

        assert verdict.decision == "allow"
        assert verdict.source == "mode"

    def test_no_constraints_produces_default_allow(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(())

        assert verdict.decision == "allow"
        assert verdict.source == "resolver"
        assert verdict.winning_constraint is None

    def test_winning_constraint_metadata_is_preserved(self) -> None:
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(
                    decision="ask",
                    source="boundary",
                    reason="external path",
                    metadata={"target": "../outside.txt"},
                ),
                Constraint(decision="allow", source="mode", reason="mode allows"),
            )
        )

        assert verdict.source == "boundary"
        assert verdict.reason == "external path"
        assert verdict.metadata["target"] == "../outside.txt"


class ActionExtractorTests:
    def test_structured_read_file_extracts_path_target(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "./src/main.py"})

        assert action.capability == "read"
        assert action.operation == "read_file"
        assert action.targets[0].kind == "path"
        assert action.targets[0].value == "src/main.py"
        assert action.targets[0].access == "read"

    def test_structured_write_file_extracts_write_target(self) -> None:
        action = ActionExtractor().extract(
            "write_file", {"path": "docs/permission-model.md", "content": "..."}
        )

        assert action.capability == "write"
        assert action.operation == "write_file"
        assert action.targets[0].access == "write"

    def test_structured_edit_file_extracts_write_target(self) -> None:
        action = ActionExtractor().extract("edit_file", {"path": "src/app.py"})

        assert action.capability == "edit"
        assert action.operation == "edit_file"
        assert action.targets[0].value == "src/app.py"
        assert action.targets[0].access == "write"

    def test_apply_patch_extracts_known_paths(self) -> None:
        action = ActionExtractor().extract(
            "apply_patch", {"paths": ["src/a.py", "src/b.py"]}
        )

        assert action.capability == "patch"
        assert [target.value for target in action.targets] == ["src/a.py", "src/b.py"]

    def test_apply_patch_extracts_paths_from_patch_text(self) -> None:
        action = ActionExtractor().extract(
            "apply_patch",
            {
                "patch_text": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: src/a.py",
                        "*** Move to: src/b.py",
                        "@@",
                        "-old",
                        "+new",
                        "*** Delete File: src/c.py",
                        "*** End Patch",
                    ]
                )
            },
        )

        assert action.capability == "patch"
        assert [target.value for target in action.targets] == [
            "src/a.py",
            "src/b.py",
            "src/c.py",
        ]

    def test_shell_extracts_command_targets_only(self) -> None:
        action = ActionExtractor().extract(
            "shell", {"commands": ["git status --short", "uv run ruff check src"]}
        )

        assert action.capability == "shell"
        assert action.operation == "run_command"
        assert [target.kind for target in action.targets] == ["command", "command"]
        assert [target.access for target in action.targets] == ["execute", "execute"]

    def test_bash_extracts_single_command_target_only(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git status --short"})

        assert action.capability == "shell"
        assert action.targets[0].kind == "command"
        assert action.targets[0].value == "git status --short"
        assert action.targets[0].access == "execute"

    def test_load_skill_extracts_skill_target(self) -> None:
        action = ActionExtractor().extract("load_skill", {"name": "code-review"})

        assert action.capability == "skill"
        assert action.operation == "load_skill"
        assert action.targets[0].kind == "skill"
        assert action.targets[0].value == "code-review"
        assert action.targets[0].access == "read"


class StructuredBoundaryPolicyTests:
    def _boundary_context(self, root: Path) -> BoundaryContext:
        return BoundaryContext(project_root=root)

    def test_workspace_internal_read_produces_allow_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "src/app.py"})

        constraints = evaluate_policy_constraints(action)

        assert len(constraints) == 1
        assert constraints[0].decision == "allow"
        assert constraints[0].source == "boundary"
        assert constraints[0].access == "read"

    def test_workspace_internal_write_produces_allow_constraint(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})

        constraints = evaluate_policy_constraints(action)

        assert len(constraints) == 1
        assert constraints[0].decision == "allow"
        assert constraints[0].access == "write"

    def test_env_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": ".env.local"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None
        assert "sensitive path" in verdict.reason

    def test_credential_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": ".ssh/id_rsa"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None
        assert "sensitive path" in verdict.reason

    def test_git_read_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": ".git/config"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None

    def test_git_write_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("edit_file", {"path": ".git/config"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None

    def test_external_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "../secret.txt"})

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert "escapes workspace" in verdict.reason

    def test_existing_blocked_workspace_path_produces_deny_constraint(self) -> None:
        action = ActionExtractor().extract(
            "read_file", {"path": ".local/chroma_db/index"}
        )

        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert "workspace blocked path" in verdict.reason

    def test_symlink_escape_read_produces_deny_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            outside = root.parent / f"{root.name}-outside.txt"
            outside.write_text("secret", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                pytest.skip(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("read_file", {"path": "link.txt"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            assert verdict.decision == "deny"
            assert verdict.winning_constraint is not None
            assert "outside all approved roots" in verdict.reason
            outside.unlink()

    def test_symlink_escape_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            outside = root.parent / f"{root.name}-outside.txt"
            outside.write_text("secret", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                pytest.skip(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("read_file", {"path": "link.txt"})
            with caplog.at_level(
                "WARNING",
                logger="xcode.harness.observability.permission_model",
            ):
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )

            assert "path resolved outside workspace boundary: link.txt" in caplog.text
            outside.unlink()

    def test_symlink_to_git_write_produces_deny(self) -> None:
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
                pytest.skip(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("edit_file", {"path": "links/gitconfig"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            assert verdict.decision == "deny"
            assert verdict.winning_constraint is not None
            assert "git metadata" in verdict.reason
            assert verdict.winning_constraint.target_pattern == ".git/config"

    def test_resolution_error_read_produces_deny(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            loop = root / "loop"
            try:
                loop.symlink_to(loop)
            except OSError as exc:
                pytest.skip(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("read_file", {"path": "loop"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            assert verdict.decision == "deny"
            assert verdict.winning_constraint is not None
            assert "cannot be resolved safely" in verdict.reason

    def test_resolution_error_write_produces_deny(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            loop = root / "loop"
            try:
                loop.symlink_to(loop)
            except OSError as exc:
                pytest.skip(f"symlink unavailable: {exc}")

            action = ActionExtractor().extract("edit_file", {"path": "loop"})
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(
                    action,
                    boundary_context=self._boundary_context(root),
                )
            )

            assert verdict.decision == "deny"
            assert verdict.winning_constraint is not None
            assert "cannot be resolved safely" in verdict.reason

    def test_mode_constraint_participates_in_resolver(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})

        verdict = PermissionResolver().resolve(
            evaluate_policy_constraints(action, execution_decision="deny")
        )

        assert verdict.decision == "deny"
        assert verdict.source == "mode"

    def test_static_policy_constraint_uses_legacy_rule_decision(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "docs/a.md"})
        policy = PermissionPolicy(
            (
                StaticPermission(tool="read_file", decision="allow"),
                StaticPermission(tool="*", decision="deny"),
            )
        )

        verdict = PermissionResolver().resolve(
            evaluate_policy_constraints(
                action,
                static_policy=policy,
                action_input='{"path": "docs/a.md"}',
            )
        )

        assert verdict.decision == "deny"
        assert verdict.source == "rule"
        assert verdict.winning_constraint is not None
        assert verdict.winning_constraint.operation == "read_file"
        assert verdict.winning_constraint.access == "read"
        assert verdict.winning_constraint.target_pattern == "docs/a.md"

    def test_static_policy_ask_beats_boundary_allow(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})
        policy = PermissionPolicy(
            (StaticPermission(tool="write_file", decision="ask"),)
        )

        verdict = PermissionResolver().resolve(
            evaluate_policy_constraints(
                action,
                static_policy=policy,
                action_input='{"path": "docs/a.md"}',
            )
        )

        assert verdict.decision == "ask"
        assert verdict.source == "rule"

    def test_global_default_ask_when_no_rule_matches(self) -> None:
        action = ActionExtractor().extract("write_file", {"path": "docs/a.md"})
        policy = PermissionPolicy(
            rules=(StaticPermission(tool="read_file", decision="allow"),),
            global_default="ask",
        )

        constraints = evaluate_policy_constraints(
            action,
            static_policy=policy,
            action_input='{"path": "docs/a.md"}',
        )
        verdict = PermissionResolver().resolve(constraints)

        assert [
            (constraint.source, constraint.decision) for constraint in constraints
        ] == [("rule", "ask"), ("boundary", "allow")]
        assert verdict.decision == "ask"
        assert verdict.source == "rule"

    # ── ExternalDirectory / .env.example / three-way classification tests ──

    def test_env_example_read_in_workspace_allowed(self) -> None:
        """read_file .env.example in workspace → boundary allow."""
        action = ActionExtractor().extract("read_file", {"path": ".env.example"})
        constraints = evaluate_policy_constraints(action)
        verdict = PermissionResolver().resolve(constraints)

        assert verdict.decision == "allow"
        assert verdict.source == "boundary"

    def test_env_example_write_in_workspace_denied(self) -> None:
        """write_file .env.example in workspace → boundary deny."""
        action = ActionExtractor().extract("write_file", {"path": ".env.example"})
        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None
        assert "sensitive path" in verdict.reason

    def test_env_example_edit_in_workspace_denied(self) -> None:
        """edit_file .env.example in workspace → boundary deny."""
        action = ActionExtractor().extract("edit_file", {"path": ".env.example"})
        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None
        assert "sensitive path" in verdict.reason

    def test_env_in_subdir_denied(self) -> None:
        """.env in subdirectory → boundary deny."""
        action = ActionExtractor().extract("read_file", {"path": "config/.env"})
        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None
        assert "sensitive path" in verdict.reason

    def test_env_local_in_subdir_denied(self) -> None:
        """.env.local in subdirectory → boundary deny."""
        action = ActionExtractor().extract("read_file", {"path": "config/.env.local"})
        verdict = PermissionResolver().resolve(evaluate_policy_constraints(action))

        assert verdict.decision == "deny"
        assert verdict.winning_constraint is not None
        assert "sensitive path" in verdict.reason

    # ── external_directory tests ──

    def _boundary_context_with_ext(
        self,
        root: Path,
        *,
        ext_path: str = "/tmp/test-ext",
        ext_access: DirAccess = "read",
    ) -> BoundaryContext:
        """Create BoundaryContext with one external directory."""
        return BoundaryContext(
            project_root=root,
            external_directories=(
                ExternalDirectory(path=Path(ext_path), access=ext_access),
            ),
        )

    def test_external_directory_read_allowed(self) -> None:
        """Absolute path inside approved external_directory with read access → allow."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                (ext_root / "doc.md").write_text("hello", encoding="utf-8")
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read"
                )
                path = (ext_root / "doc.md").as_posix()
                action = ActionExtractor().extract("read_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "allow"
                assert verdict.source == "boundary"

    def test_external_directory_write_denied_when_read_only(self) -> None:
        """Write to ext_dir with access=read → boundary deny (insufficient access)."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read"
                )
                path = (ext_root / "new.txt").as_posix()
                action = ActionExtractor().extract("write_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "deny"
                assert verdict.winning_constraint is not None
                assert "outside all approved roots" in verdict.reason

    def test_external_directory_write_allowed_with_write_access(self) -> None:
        """Write to ext_dir with access=write → allow."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="write"
                )
                path = (ext_root / "new.txt").as_posix()
                action = ActionExtractor().extract("write_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "allow"

    def test_external_directory_read_denied_when_write_only(self) -> None:
        """Read from ext_dir with access=write → boundary deny (insufficient access)."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                (ext_root / "dat.md").write_text("data", encoding="utf-8")
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="write"
                )
                path = (ext_root / "dat.md").as_posix()
                action = ActionExtractor().extract("read_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "deny"
                assert verdict.winning_constraint is not None
                assert "outside all approved roots" in verdict.reason

    def test_external_directory_read_write_access_both(self) -> None:
        """read_write access allows both read and write."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                (ext_root / "dat.md").write_text("data", encoding="utf-8")
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read_write"
                )

                read_action = ActionExtractor().extract(
                    "read_file", {"path": (ext_root / "dat.md").as_posix()}
                )
                read_verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(read_action, boundary_context=ctx)
                )
                assert read_verdict.decision == "allow"

                write_action = ActionExtractor().extract(
                    "write_file", {"path": (ext_root / "new.txt").as_posix()}
                )
                write_verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(write_action, boundary_context=ctx)
                )
                assert write_verdict.decision == "allow"

    def test_path_outside_all_roots_denied(self) -> None:
        """Path outside workspace and all external directories → deny."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            action = ActionExtractor().extract("read_file", {"path": "/etc/passwd"})
            ctx = self._boundary_context_with_ext(
                root, ext_path="/nonexistent-ext", ext_access="read"
            )
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(action, boundary_context=ctx)
            )

            assert verdict.decision == "deny"
            assert verdict.winning_constraint is not None
            assert "outside all approved roots" in verdict.reason

    def test_ext_dir_prefix_not_mistaken(self) -> None:
        """/tmp/foo2 does not match /tmp/foo via is_relative_to."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            ctx = self._boundary_context_with_ext(
                root, ext_path="/tmp/foo", ext_access="read"
            )
            action = ActionExtractor().extract(
                "read_file", {"path": "/tmp/foo2/secret.txt"}
            )
            verdict = PermissionResolver().resolve(
                evaluate_policy_constraints(action, boundary_context=ctx)
            )

            assert verdict.decision == "deny"
            assert verdict.winning_constraint is not None
            assert "outside all approved roots" in verdict.reason

    def test_sensitive_path_inside_external_directory_denied(self) -> None:
        """.env inside external_directory remains denied."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                (ext_root / ".env").write_text("SECRET=1", encoding="utf-8")
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read_write"
                )
                path = (ext_root / ".env").as_posix()
                action = ActionExtractor().extract("read_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "deny"
                assert verdict.winning_constraint is not None
                assert "sensitive path" in verdict.reason

    def test_env_example_write_inside_external_directory_denied(self) -> None:
        """.env.example write inside external_directory remains denied."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read_write"
                )
                path = (ext_root / ".env.example").as_posix()
                action = ActionExtractor().extract("write_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "deny"
                assert verdict.winning_constraint is not None
                assert "sensitive path" in verdict.reason

    def test_git_path_inside_external_directory_denied(self) -> None:
        """.git/config inside external_directory remains denied."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                git_dir = ext_root / ".git"
                git_dir.mkdir()
                (git_dir / "config").write_text("[core]\n", encoding="utf-8")
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read_write"
                )
                path = (git_dir / "config").as_posix()
                action = ActionExtractor().extract("read_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "deny"
                assert verdict.winning_constraint is not None
                assert "git metadata path" in verdict.reason

    def test_credential_path_inside_external_directory_denied(self) -> None:
        """.ssh/id_rsa inside external_directory remains denied."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                ssh_dir = ext_root / ".ssh"
                ssh_dir.mkdir()
                (ssh_dir / "id_rsa").write_text("key", encoding="utf-8")
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read_write"
                )
                path = (ssh_dir / "id_rsa").as_posix()
                action = ActionExtractor().extract("read_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "deny"
                assert verdict.winning_constraint is not None
                assert "sensitive path" in verdict.reason

    def test_boundary_deny_still_beats_static_allow_from_step3(self) -> None:
        """PermissionResolver: boundary deny still beats static allow."""
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="allow", source="rule", reason="static allows"),
                Constraint(
                    decision="deny", source="boundary", reason="boundary denies"
                ),
            )
        )
        assert verdict.decision == "deny"
        assert verdict.source == "boundary"

    def test_env_example_read_inside_external_directory_allowed(self) -> None:
        """.env.example read inside external_directory → allowed (doc)."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            with tempfile.TemporaryDirectory() as ext_text:
                ext_root = Path(ext_text)
                (ext_root / ".env.example").write_text("EXAMPLE=1", encoding="utf-8")
                ctx = self._boundary_context_with_ext(
                    root, ext_path=ext_root.as_posix(), ext_access="read"
                )
                path = (ext_root / ".env.example").as_posix()
                action = ActionExtractor().extract("read_file", {"path": path})
                verdict = PermissionResolver().resolve(
                    evaluate_policy_constraints(action, boundary_context=ctx)
                )

                assert verdict.decision == "allow"


class ExternalDirectoryConfigValidationTests:
    """验证 external_directories 配置的 fail-fast 校验。"""

    def test_valid_external_directory_passes(self) -> None:
        SecurityExternalDirectory.model_validate(
            {"path": "/tmp/valid", "access": "read"}
        )

    def test_valid_external_directory_default_access(self) -> None:
        SecurityExternalDirectory.model_validate({"path": "/tmp/valid"})

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SecurityExternalDirectory.model_validate({"access": "read"})
        assert "path" in str(exc_info.value)

    def test_empty_path_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SecurityExternalDirectory.model_validate({"path": "", "access": "read"})
        assert "path" in str(exc_info.value)

    def test_invalid_access_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SecurityExternalDirectory.model_validate(
                {"path": "/tmp/foo", "access": "delete"}
            )
        assert "access" in str(exc_info.value)

    def test_non_dict_entry_raises(self) -> None:
        from pydantic import ValidationError as Ve

        with pytest.raises(Ve):
            SecurityRuntimeConfig.model_validate(
                {"external_directories": ["not-a-dict"]}
            )

    def test_no_security_does_not_raise(self) -> None:
        SecurityRuntimeConfig.model_validate({})  # should not raise


class GrantStoreTests:
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

        assert self._lookup_path(store, "src/foo") is not None
        assert self._lookup_path(store, "src/foo/bar.py") is not None
        assert self._lookup_path(store, "src/foobar.py") is None

        file_store = InMemoryGrantStore((self._grant("src/foo.py"),))
        assert self._lookup_path(file_store, "src/foo.py.bak") is None

    def test_overlapping_deny_beats_allow(self) -> None:
        store = InMemoryGrantStore(
            (
                self._grant("src", decision="allow", grant_id="allow-src"),
                self._grant("src/foo", decision="deny", grant_id="deny-foo"),
            )
        )

        record = self._lookup_path(store, "src/foo/bar.py")

        assert record is not None
        assert record.decision == "deny"
        assert record.grant_id == "deny-foo"

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
            assert record.grant_id == "grant-1"


class PermissionEngineShadowTests:
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
                mode_fallback="allow",
            )
        )

        result = engine.decide(
            "danger",
            {"input": "go"},
            tool_spec=tool,
        )

        assert not (result.blocked)
        assert result.decision == "allow"
        assert result.shadow_action is not None
        assert result.shadow_verdict is not None
        assert result.shadow_verdict.decision == "allow"
        assert result.shadow_diff is None

    def test_shadow_mode_uses_structured_boundary_constraints(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig(shadow_model_enabled=True))

        result = engine.decide(
            "read_file",
            {"path": ".env"},
        )

        assert result.blocked
        assert result.decision == "deny"
        assert result.source == "boundary"
        assert result.shadow_verdict is not None
        assert result.shadow_verdict.decision == "deny"
        assert result.shadow_verdict.source == "boundary"
        assert result.shadow_diff is None

    def test_shadow_boundary_context_denies_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            outside = root.parent / f"{root.name}-outside.txt"
            outside.write_text("secret", encoding="utf-8")
            try:
                (root / "link.txt").symlink_to(outside)
            except OSError as exc:
                pytest.skip(f"symlink unavailable: {exc}")

            engine = PermissionEngine(
                PermissionEngineConfig(
                    shadow_model_enabled=True,
                    project_root=root,
                )
            )

            result = engine.decide(
                "read_file",
                {"path": "link.txt"},
            )

            assert result.blocked
            assert result.decision == "deny"
            assert result.source == "boundary"
            assert result.shadow_verdict is not None
            assert result.shadow_verdict.decision == "deny"
            assert "outside all approved roots" in result.shadow_verdict.reason
            outside.unlink()

    def test_shadow_static_policy_constraints_do_not_change_legacy_result(
        self,
    ) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="read_file", decision="ask"),)
                ),
                shadow_model_enabled=True,
            )
        )

        result = engine.decide(
            "read_file",
            {"path": "docs/a.md"},
        )

        assert result.decision == "ask"
        assert result.shadow_verdict is not None
        assert [
            (constraint.source, constraint.decision)
            for constraint in result.shadow_verdict.constraints
        ] == [("rule", "ask"), ("boundary", "allow")]
        assert result.shadow_diff is None


class ShadowApprovalCandidateTests:
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
        result = engine.decide(tool, tool_input)
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
        assert result.decision == expected_decision
        assert result.blocked == expected_blocked

    # ── positive: new-format session grant matching → allow ──

    def test_new_session_grant_allow(self) -> None:
        store = InMemoryGrantStore((_grant("src/foo.py"),))
        engine = self._engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            session_grant_store=store,
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        # New resolver path: grant found → allow
        self._assert_legacy_path_untouched(result, "allow", False)
        assert candidate.would_resolve == "allow"
        assert len(candidate.fingerprints) == 1
        assert candidate.fingerprints[0].source == "new_session"
        assert candidate.fingerprints[0].grant is not None
        assert candidate.fingerprints[0].grant.decision == "allow"

    # ── positive: new-format permanent grant matching → deny ──

    def test_new_permanent_grant_deny(self) -> None:
        store = InMemoryGrantStore()
        store.add(_grant("src/foo.py", decision="deny"))
        engine = self._engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            permanent_grant_store=store,
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        # New resolver path: deny grant found → deny
        self._assert_legacy_path_untouched(result, "deny", True)
        assert candidate.would_resolve == "deny"
        assert candidate.fingerprints[0].source == "new_permanent"

    # ── positive: no grants at all → would_call_approval ──

    def test_no_grants_produces_would_call_approval(self) -> None:
        engine = self._engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        # No grants → callback required
        self._assert_legacy_path_untouched(result, "ask", True)
        assert candidate.would_resolve == "would_call_approval"
        assert candidate.fingerprints[0].source == "none"
        assert candidate.fingerprints[0].grant is None

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
            static_policy=PermissionPolicy(
                (StaticPermission(tool="apply_patch", decision="ask"),)
            ),
            session_grant_store=store,
        )
        tool_input: dict[str, object] = {
            "paths": ["src/allowed.py", "src/unknown.py"],
        }
        result, candidate = self._decide(engine, "apply_patch", tool_input)

        # Resolver: mixed → would_call_approval → ask
        self._assert_legacy_path_untouched(result, "ask", True)
        assert candidate.would_resolve == "would_call_approval"
        assert len(candidate.fingerprints) == 2
        # target A hits
        assert candidate.fingerprints[0].source == "new_session"
        assert candidate.fingerprints[0].grant is not None
        # target B misses
        assert candidate.fingerprints[1].source == "none"
        assert candidate.fingerprints[1].grant is None

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
            static_policy=PermissionPolicy(
                (StaticPermission(tool="apply_patch", decision="ask"),)
            ),
            session_grant_store=store,
        )
        tool_input: dict[str, object] = {
            "paths": ["src/good.py", "src/bad.py"],
        }
        result, candidate = self._decide(engine, "apply_patch", tool_input)

        # Resolver: deny grant found → deny
        self._assert_legacy_path_untouched(result, "deny", True)
        assert candidate.would_resolve == "deny"
        assert len(candidate.fingerprints) == 2
        deny_grant = candidate.fingerprints[1].grant
        assert deny_grant is not None
        assert deny_grant.decision == "deny"

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
            static_policy=PermissionPolicy(
                (StaticPermission(tool="apply_patch", decision="ask"),)
            ),
            session_grant_store=store,
        )
        tool_input: dict[str, object] = {
            "paths": ["src/a.py", "src/b.py"],
        }
        result, candidate = self._decide(engine, "apply_patch", tool_input)

        # Resolver: all targets have grants → allow
        self._assert_legacy_path_untouched(result, "allow", False)
        assert candidate.would_resolve == "allow"

    # ── edge: shadow_model_enabled=False → no candidate ──

    def test_shadow_disabled_no_candidate(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="read_file", decision="ask"),)
                ),
                shadow_model_enabled=False,
            )
        )
        result = engine.decide(
            "read_file",
            {"path": "src/foo.py"},
        )

        assert result.shadow_approval_candidate is None

    # ── edge: shadow verdict not ask → no candidate ──

    def test_shadow_verdict_allow_no_candidate(self) -> None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=PermissionPolicy(
                    (StaticPermission(tool="read_file", decision="allow"),)
                ),
                shadow_model_enabled=True,
                mode_fallback="allow",
            )
        )
        result = engine.decide("read_file", {"path": "src/foo.py"})

        assert result.shadow_approval_candidate is None

    # ── edge: bash with command target → now gets shadow approval candidate ──
    # (STRUCTURED_TOOLS gate removed; any action with targets participates)

    def test_bash_with_targets_gets_shadow_candidate(self) -> None:
        engine = self._engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="bash", decision="ask"),)
            ),
        )
        result = engine.decide("bash", {"command": "echo hello"})

        # bash maintenant obtient un shadow_approval_candidate
        # car il a des targets (kind="command")
        assert result.shadow_approval_candidate is not None
        assert result.shadow_approval_candidate.would_resolve == "would_call_approval"

    # ── fingerprint shape: matches expected fields ──

    def test_fingerprint_shape(self) -> None:
        store = InMemoryGrantStore((_grant("src/foo.py"),))
        engine = self._engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            session_grant_store=store,
        )
        _, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        fp = candidate.fingerprints[0].fingerprint
        assert fp.capability == "read"
        assert fp.operation == "read_file"
        assert fp.target_kind == "path"
        assert fp.target_pattern == "src/foo.py"
        assert fp.access == "read"

    # ── edge: new session deny works ──

    def test_new_session_deny(self) -> None:
        store = InMemoryGrantStore((_grant("src/foo.py", decision="deny"),))
        engine = self._engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            session_grant_store=store,
        )
        result, candidate = self._decide(engine, "read_file", {"path": "src/foo.py"})

        self._assert_legacy_path_untouched(result, "deny", True)
        assert candidate.would_resolve == "deny"
        assert candidate.fingerprints[0].source == "new_session"
        deny_grant = candidate.fingerprints[0].grant
        assert deny_grant is not None
        assert deny_grant.decision == "deny"


class ApprovalCutoverEnabledTests:
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
            {"path": "../outside.txt"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert result.blocked
        assert result.decision == "deny"
        assert len(cb.calls) == 0
        assert result.approval_result is None

    # ── allow 短接 ──

    def test_allow_verdict_short_circuits(self) -> None:
        """resolver 返回 allow → 短接，不查找授权，不回调。"""
        cb = self._RecordingCallback()
        engine = PermissionEngine(PermissionEngineConfig(mode_fallback="allow"))
        result = engine.decide(
            "read_file",
            {"path": "src/test.py"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert not (result.blocked)
        assert result.decision == "allow"
        assert len(cb.calls) == 0
        assert result.approval_result is None

    # ── ask + 新格式授权命中 ──

    def test_ask_new_session_grant_hit(self) -> None:
        """新格式 session 授权命中 → 回调不被调用，授权直接决定。"""
        store = InMemoryGrantStore((_grant("src/foo.py"),))
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            session_grant_store=store,
        )
        cb = self._RecordingCallback()
        result = engine.decide(
            "read_file",
            {"path": "src/foo.py"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert not (result.blocked)
        assert result.decision == "allow"
        assert len(cb.calls) == 0
        assert result.approval_result is not None
        assert result.approval_result.decision == "allow"
        assert result.approval_result.grant_id is not None

    def test_ask_new_permanent_grant_hit(self) -> None:
        """新格式 permanent 授权命中 → 回调不被调用，授权直接决定。"""
        store = InMemoryGrantStore((_grant("src/foo.py", decision="deny"),))
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            permanent_grant_store=store,
        )
        cb = self._RecordingCallback()
        result = engine.decide(
            "read_file",
            {"path": "src/foo.py"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert result.blocked
        assert result.decision == "deny"
        assert len(cb.calls) == 0
        assert result.approval_result is not None
        assert result.approval_result.decision == "deny"

    # ── ask + 无授权 → 回调 ──

    def test_ask_no_grant_calls_callback(self) -> None:
        """无命中授权 → 以 (tool_spec, tool_input) 调用回调。"""
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
        )
        cb = self._RecordingCallback(HITLResult("allow", "once"))
        tool = self._tool("read_file")
        result = engine.decide(
            "read_file",
            {"path": "src/unknown.py"},
            tool_spec=tool,
            approval_callback=cb,
        )
        assert not (result.blocked)
        assert result.decision == "allow"
        assert len(cb.calls) == 1
        called_spec, called_input = cb.calls[0]
        assert called_spec is tool
        assert called_input == {"path": "src/unknown.py"}

    # ── 回调返回 allow/once ──

    def test_callback_allow_once_no_store_write(self) -> None:
        """allow/once → 放行，不写入任何 store。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "once"))
        result = engine.decide(
            "read_file",
            {"path": "src/foo.py"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert not (result.blocked)
        assert result.decision == "allow"
        assert len(store.records()) == 0

    # ── 回调返回 allow/session（单目标）──

    def test_callback_allow_session_writes_store(self) -> None:
        """allow/session（单目标）→ 放行，session store 写入 GrantRecord。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "session"))
        result = engine.decide(
            "read_file",
            {"path": "src/foo.py"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert not (result.blocked)
        assert result.decision == "allow"
        assert len(store.records()) == 1
        record = store.records()[0]
        assert record.decision == "allow"
        assert record.scope == "session"
        assert record.target_pattern == "src/foo.py"

    # ── 回调返回 allow/session（apply_patch 多目标）──

    def test_callback_allow_session_apply_patch_no_store_write(self) -> None:
        """allow/session（多目标）→ 放行但不写入 store，元数据记录限制。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="apply_patch", decision="ask"),)
            ),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "session"))
        tool_input: dict[str, object] = {"paths": ["src/a.py", "src/b.py"]}
        result = engine.decide(
            "apply_patch",
            tool_input,
            tool_spec=self._tool("apply_patch"),
            approval_callback=cb,
        )
        assert not (result.blocked)
        assert result.decision == "allow"
        assert len(store.records()) == 0
        assert result.metadata is not None
        assert result.metadata.get("multi_target_restriction")
        assert result.metadata.get("requested_scope") == "session"
        assert result.metadata.get("effective_scope") == "once"

    # ── 回调返回 deny ──

    def test_callback_deny_no_store_write(self) -> None:
        """deny → 拒绝，不写入任何 store。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            session_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("deny", "once"))
        result = engine.decide(
            "read_file",
            {"path": "src/foo.py"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert result.blocked
        assert result.decision == "deny"
        assert len(cb.calls) == 1
        assert len(store.records()) == 0

    # ── 回调返回 allow/permanent ──

    def test_callback_allow_permanent_writes_store(self) -> None:
        """allow/permanent → 放行，permanent store 写入 GrantRecord。"""
        store = InMemoryGrantStore()
        engine = self._cutover_engine(
            static_policy=PermissionPolicy(
                (StaticPermission(tool="read_file", decision="ask"),)
            ),
            permanent_grant_store=store,
        )
        cb = self._RecordingCallback(HITLResult("allow", "permanent"))
        result = engine.decide(
            "read_file",
            {"path": "src/foo.py"},
            tool_spec=self._tool("read_file"),
            approval_callback=cb,
        )
        assert not (result.blocked)
        assert result.decision == "allow"
        assert len(store.records()) == 1
        record = store.records()[0]
        assert record.scope == "permanent"
        assert record.target_pattern == "src/foo.py"


class ShellCutoverTests:
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

    def test_build_shell_profile_allows_without_approval(self) -> None:
        """build 默认 profile 将 bash 作为工具整体放行。"""
        engine = PermissionEngine(
            PermissionEngineConfig(
                mode_ruleset=(Rule(action="bash", effect="allow"),),
                mode_fallback="ask",
            )
        )
        result = engine.decide(
            "bash",
            {"command": "echo hello"},
            tool_spec=self._bash_tool_spec(),
        )

        assert result.decision == "allow"
        assert not result.blocked

    def test_user_shell_rule_denies_without_callback(self) -> None:
        """用户配置的 shell 规则命中时可以拒绝命令。"""
        calls: list[object] = []

        def cb(_tool_spec: object, _tool_input: dict[str, object]) -> HITLResult:
            calls.append("called")
            return HITLResult(decision="allow", scope="once")

        engine = PermissionEngine(
            PermissionEngineConfig(
                user_ruleset=(
                    Rule(
                        action="bash",
                        command="rm",
                        flags_any={"-fr"},
                        effect="deny",
                    ),
                ),
                mode_fallback="ask",
            )
        )
        result = engine.decide(
            "bash",
            {"command": "rm -rf /"},
            tool_spec=self._bash_tool_spec(),
            approval_callback=cb,
        )

        assert result.decision == "deny"
        assert result.blocked
        assert len(calls) == 0

    def test_confirmation_rule_calls_callback_once_scope(self) -> None:
        """确认规则命中时 callback 被调用，once scope 不写 grant store。"""
        calls: list[HITLResult] = []

        def cb(_tool_spec: object, _tool_input: dict[str, object]) -> HITLResult:
            result = HITLResult(decision="allow", scope="once")
            calls.append(result)
            return result

        engine = PermissionEngine(PermissionEngineConfig(mode_fallback="ask"))
        tool_input: dict[str, object] = {"command": "rm some_file.py"}
        result = engine.decide(
            "bash",
            tool_input,
            tool_spec=self._bash_tool_spec(),
            approval_callback=cb,
        )
        assert result.decision == "allow"
        assert not (result.blocked)
        assert len(calls) == 1
        assert calls[0].scope == "once"

    def test_confirmation_rule_callback_deny(self) -> None:
        """确认规则命中且 callback 返回 deny 时 blocked=True。"""

        def cb(_tool_spec: object, _tool_input: dict[str, object]) -> HITLResult:
            return HITLResult(decision="deny", scope="once")

        engine = PermissionEngine(PermissionEngineConfig(mode_fallback="ask"))
        tool_input: dict[str, object] = {"command": "rm some_file.py"}
        result = engine.decide(
            "bash",
            tool_input,
            tool_spec=self._bash_tool_spec(),
            approval_callback=cb,
        )
        assert result.decision == "deny"
        assert result.blocked

    def test_non_shell_tool_unaffected(self) -> None:
        engine = PermissionEngine(PermissionEngineConfig(mode_fallback="allow"))
        tool_input: dict[str, object] = {"path": "some_file.py"}
        result = engine.decide(
            "read_file",
            tool_input,
            tool_spec=ToolSpec(
                name="read_file",
                description="test",
                input_hint="",
                handler=lambda _: "",
            ),
        )
        # mode_fallback="allow" → 默认 allow
        assert result.decision == "allow"
        assert not (result.blocked)

    def test_user_deny_rule_denies_without_metadata_flag(self) -> None:
        """用户配置的 deny 规则拒绝命令，不再附带特殊 metadata。"""
        engine = PermissionEngine(
            PermissionEngineConfig(
                user_ruleset=(
                    Rule(
                        action="bash",
                        command="rm",
                        flags_any={"-fr"},
                        effect="deny",
                    ),
                ),
                mode_fallback="ask",
            )
        )
        result = engine.decide(
            "bash",
            {"command": "rm -rf /"},
            tool_spec=self._bash_tool_spec(),
        )

        assert result.decision == "deny"
        assert result.blocked
        assert result.metadata is None


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


class HookInvariantTests:
    """验证 hook 约束不变量 (STEP 5 第 11.2-11.4 节)。"""

    maxDiff = None

    def test_hook_allow_cannot_override_static_deny(self) -> None:
        """Hook 返回 allow 不能覆盖 StaticPolicy 的 explicit deny。"""
        always_allow_hook: PolicyEvaluator = _AlwaysAllowHook()
        static_policy = PermissionPolicy(
            rules=(StaticPermission(tool="bash", decision="deny"),)
        )
        action = ActionExtractor().extract("bash", {"command": "echo hello"})

        constraints = evaluate_policy_constraints(
            action,
            execution_decision="allow",
            static_policy=static_policy,
            hook_constraint_providers=(always_allow_hook,),
        )

        verdict = PermissionResolver().resolve(constraints)
        assert verdict.decision == "deny"
        assert verdict.winning_constraint.source == "rule"

    def test_hook_allow_is_honored_when_no_other_deny(self) -> None:
        """无其他 evaluator 产生 deny 时，hook 的 allow 应被采纳。"""
        always_allow_hook: PolicyEvaluator = _AlwaysAllowHook()
        action = ActionExtractor().extract("bash", {"command": "echo hello"})

        constraints = evaluate_policy_constraints(
            action,
            execution_decision="allow",
            hook_constraint_providers=(always_allow_hook,),
        )

        verdict = PermissionResolver().resolve(constraints)
        assert verdict.decision == "allow"

    def test_hook_deny_can_override_builtin_allow(self) -> None:
        """Hook 返回 deny 可以拒绝内置 evaluator 放行的 action。"""
        always_deny_hook: PolicyEvaluator = _AlwaysDenyHook()
        action = ActionExtractor().extract("bash", {"command": "echo hello"})

        constraints = evaluate_policy_constraints(
            action,
            execution_decision="allow",
            hook_constraint_providers=(always_deny_hook,),
        )

        verdict = PermissionResolver().resolve(constraints)
        assert verdict.decision == "deny"
        assert verdict.winning_constraint.source == "hook:always_deny"


class StaticPolicyLastMatchWinsTests:
    """last-match-wins within StaticPolicyEvaluator static rule matching."""

    def test_last_match_wins_basic(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (
            StaticPermission(tool="bash", decision="deny"),
            StaticPermission(tool="bash", decision="allow"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 1
        assert constraints[0].decision == "allow"
        assert constraints[0].source == "rule"

    def test_last_match_wins_reversed(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (
            StaticPermission(tool="bash", decision="allow"),
            StaticPermission(tool="bash", decision="deny"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 1
        assert constraints[0].decision == "deny"

    def test_global_asterisk_rule(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (StaticPermission(tool="*", decision="deny"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 1
        assert constraints[0].decision == "deny"

    def test_input_regex_matches(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "curl example.com"})
        rules = (
            StaticPermission(tool="bash", decision="ask", input_regex=r"curl|wget"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 1
        assert constraints[0].decision == "ask"

    def test_input_regex_no_match(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "ls -la"})
        rules = (
            StaticPermission(tool="bash", decision="ask", input_regex=r"curl|wget"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 0

    def test_target_matches_action_target(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "src/foo.py"})
        rules = (
            StaticPermission(tool="read_file", decision="allow", target="src/foo.py"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 1
        assert constraints[0].decision == "allow"

    def test_target_no_match(self) -> None:
        action = ActionExtractor().extract("read_file", {"path": "src/other.py"})
        rules = (
            StaticPermission(tool="read_file", decision="allow", target="src/foo.py"),
        )
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 0

    def test_target_type_matches_command(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git status"})
        rules = (StaticPermission(tool="bash", decision="ask", target_type="command"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 1
        assert constraints[0].decision == "ask"

    def test_target_type_no_match(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "git status"})
        rules = (StaticPermission(tool="bash", decision="ask", target_type="path"),)
        evaluator = StaticPolicyEvaluator(rules)
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 0

    def test_global_default_applies_when_no_rule_matches(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        rules = (StaticPermission(tool="read_file", decision="allow"),)
        evaluator = StaticPolicyEvaluator(rules, global_default="ask")
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 1
        assert constraints[0].decision == "ask"
        assert "global_default" in constraints[0].reason

    def test_no_rules_no_global_default_emits_nothing(self) -> None:
        action = ActionExtractor().extract("bash", {"command": "echo hello"})
        evaluator = StaticPolicyEvaluator()
        constraints = evaluator.evaluate(action)
        assert len(constraints) == 0

    def test_deny_still_beats_static_allow(self) -> None:
        """PermissionResolver: deny > static allow."""
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="allow", source="rule", reason="static allows"),
                Constraint(
                    decision="deny",
                    source="safety",
                    reason="deny",
                ),
            )
        )
        assert verdict.decision == "deny"
        assert verdict.source == "safety"

    def test_boundary_deny_still_beats_static_allow(self) -> None:
        """PermissionResolver: boundary deny > static allow."""
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(decision="allow", source="rule", reason="static allows"),
                Constraint(
                    decision="deny", source="boundary", reason="boundary denies"
                ),
            )
        )
        assert verdict.decision == "deny"
        assert verdict.source == "boundary"

    def test_static_ask_beats_boundary_allow(self) -> None:
        """PermissionResolver: static ask > boundary allow."""
        resolver = PermissionResolver()
        verdict = resolver.resolve(
            (
                Constraint(
                    decision="allow", source="boundary", reason="boundary allows"
                ),
                Constraint(decision="ask", source="rule", reason="static asks"),
            )
        )
        assert verdict.decision == "ask"
        assert verdict.source == "rule"


class SessionGrantIsolationTests:
    """SessionGrantStoreManager + ToolGate provider + subagent isolation."""

    def test_same_session_id_reuses_store(self) -> None:
        manager = SessionGrantStoreManager()
        store_a1 = manager.get_for_session("session-A")
        store_a2 = manager.get_for_session("session-A")
        assert store_a1 is store_a2

    def test_different_session_id_different_store(self) -> None:
        manager = SessionGrantStoreManager()
        store_a = manager.get_for_session("session-A")
        store_b = manager.get_for_session("session-B")
        assert store_a is not store_b

    def test_grant_in_session_a_not_in_b(self) -> None:
        """Grant added to session A is not visible in session B."""
        manager = SessionGrantStoreManager()
        store_a = manager.get_for_session("session-A")
        store_b = manager.get_for_session("session-B")

        grant = _grant("src/main.py")
        store_a.add(grant)
        assert grant in store_a.records()
        assert grant not in store_b.records()

    def test_switch_back_reuses_store_with_grants(self) -> None:
        """Switching back to a previously-visited session reuses its store."""
        manager = SessionGrantStoreManager()
        store_a = manager.get_for_session("session-A")
        grant = _grant("src/main.py")
        store_a.add(grant)

        # "Switch" to B, then back to A
        manager.get_for_session("session-B")
        store_a_again = manager.get_for_session("session-A")
        assert store_a is store_a_again
        assert len(store_a_again.records()) == 1

    def test_subagent_fresh_empty_store(self) -> None:
        """Subagent receives a fresh empty InMemoryGrantStore."""
        subagent_store = InMemoryGrantStore(session_id="subagent-abcdef01")
        assert len(subagent_store.records()) == 0

    def test_subagent_does_not_inherit_parent_grants(self) -> None:
        """Subagent store is independent of parent session store."""
        manager = SessionGrantStoreManager()
        parent_store = manager.get_for_session("parent-session")
        parent_store.add(_grant("src/main.py"))

        subagent_store = InMemoryGrantStore(session_id="subagent-abcdef01")
        assert len(subagent_store.records()) == 0

    def test_session_restart_loses_grants(self) -> None:
        """New SessionGrantStoreManager = process restart = all stores lost."""
        manager1 = SessionGrantStoreManager()
        store = manager1.get_for_session("session-A")
        store.add(_grant("src/main.py"))
        assert len(store.records()) == 1

        manager2 = SessionGrantStoreManager()
        store_new = manager2.get_for_session("session-A")
        assert len(store_new.records()) == 0

    def test_repl_hitl_handler_no_grant_lookup(self) -> None:
        """ReplHITLHandler does not call lookup on any grant store."""
        from xcode.cli.repl_hitl import ReplHITLHandler

        handler = ReplHITLHandler()
        assert not (hasattr(handler, "_session_store"))
        assert not (hasattr(handler, "_permanent_store"))

    def test_repl_hitl_handler_no_write_grants_method(self) -> None:
        """ReplHITLHandler does not have _write_grants method."""
        from xcode.cli.repl_hitl import ReplHITLHandler

        handler = ReplHITLHandler()
        assert not (hasattr(handler, "_write_grants"))

    def test_toolgate_snapshot_uses_provider(self) -> None:
        """ToolGate resolves session store from provider at snapshot time."""
        from xcode.harness.agent_runtime.execution_modes import ExecutionModeState
        from xcode.harness.agent_runtime.tool_gate import ToolGate

        manager = SessionGrantStoreManager()
        mode = ExecutionModeState()

        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            hook_manager=None,
            audit_logger=None,
            session_id="test",
            session_grant_store_provider=lambda: manager.get_for_session(
                "provider-test"
            ),
        )
        snap = gate.snapshot()
        assert isinstance(snap.session_grant_store, InMemoryGrantStore)
        assert snap.session_grant_store._session_id == "provider-test"

    def test_toolgate_provider_resolves_after_session_change(self) -> None:
        """Provider resolves new store after session_id changes."""
        from xcode.harness.agent_runtime.execution_modes import ExecutionModeState
        from xcode.harness.agent_runtime.tool_gate import ToolGate

        manager = SessionGrantStoreManager()
        mode = ExecutionModeState()

        # Simulate changing session_id between turns
        current_id = ["session-A"]

        def provider() -> InMemoryGrantStore:
            return manager.get_for_session(current_id[0])

        gate = ToolGate(
            mode_state=mode,
            approval_callback=None,
            permission_policy=None,
            hook_manager=None,
            audit_logger=None,
            session_id="test",
            session_grant_store_provider=provider,
        )

        snap_a = gate.snapshot()
        assert snap_a.session_grant_store is not None
        snap_a.session_grant_store.add(_grant("src/a.py"))

        # "Switch" to session B
        current_id[0] = "session-B"
        snap_b = gate.snapshot()
        assert snap_b.session_grant_store is not None
        assert snap_a.session_grant_store is not snap_b.session_grant_store
        assert len(snap_b.session_grant_store.records()) == 0

        # Switch back to A — store with grant is reused
        current_id[0] = "session-A"
        snap_a2 = gate.snapshot()
        assert snap_a.session_grant_store is snap_a2.session_grant_store


if __name__ == "__main__":
    pytest.main()
