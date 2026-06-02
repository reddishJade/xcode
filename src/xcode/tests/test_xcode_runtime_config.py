from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

from xcode.harness.config import (
    discover_runtime_config,
    load_runtime_config,
    resolve_config_path,
    to_agent_config,
)


class XcodeRuntimeConfigTests(unittest.TestCase):
    def test_missing_config_uses_defaults(self) -> None:
        config = load_runtime_config(None)

        self.assertEqual(
            config.provider.model_profiles["main"].transport, "openai_chat"
        )
        self.assertEqual(
            config.provider.model_profiles["main"].chat_model, "deepseek-v4-flash"
        )
        self.assertEqual(
            config.provider.model_profiles["main"].base_url,
            "https://api.deepseek.com",
        )
        self.assertEqual(
            config.provider.model_profiles["subagent"],
            config.provider.model_profiles["main"],
        )
        self.assertTrue(config.skills.auto_trigger)
        self.assertEqual(config.agent.max_steps, 20)
        self.assertIsNone(config.paths.sessions_dir)
        self.assertFalse(config.daemon.enabled)
        self.assertEqual(config.daemon.interval_seconds, 30)

    def test_loads_runtime_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                '{"provider":{"model_profiles":{'
                '"main":{"transport":"openai_responses",'
                '"chat_model":"main-test","base_url":"https://main.test"},'
                '"subagent":{"chat_model":"subagent-test"}}},'
                '"agent":{"max_steps":9,"compact_threshold":20,'
                '"compact_token_threshold":3000,"max_recent_messages":7,'
                '"tool_workers":2,"watchdog_repeated_tool_limit":4},'
                '"paths":{"sessions_dir":"sessions",'
                '"skills_dir":"skills"},'
                '"observability":{"audit_path":"audit.jsonl"},'
                '"tools":{"network_commands":"deny"},'
                '"skills":{"auto_trigger":false},'
                '"prompt":{"modules":["identity","tools"]},'
                '"daemon":{"enabled":true,"interval_seconds":15}}',
                encoding="utf-8",
            )

            config = load_runtime_config(path)

            self.assertEqual(
                config.provider.model_profiles["main"].transport, "openai_responses"
            )
            self.assertEqual(
                config.provider.model_profiles["main"].chat_model, "main-test"
            )
            self.assertEqual(
                config.provider.model_profiles["main"].base_url,
                "https://main.test",
            )
            self.assertEqual(
                config.provider.model_profiles["subagent"].chat_model,
                "subagent-test",
            )
            self.assertFalse(config.skills.auto_trigger)
            self.assertEqual(config.prompt.modules, ("identity", "tools"))
            self.assertEqual(config.agent.max_steps, 9)
            self.assertEqual(config.agent.tool_workers, 2)
            self.assertEqual(config.paths.sessions_dir, Path("sessions"))
            self.assertEqual(config.paths.skills_dir, Path("skills"))
            self.assertEqual(config.observability.audit_path, Path("audit.jsonl"))
            self.assertTrue(config.daemon.enabled)
            self.assertEqual(config.daemon.interval_seconds, 15)

    def test_loads_chatglm_profile_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                '{"provider":{"model_profiles":{"main":{'
                '"transport":"chatglm",'
                '"chat_model":"glm-4.7",'
                '"base_url":"https://open.bigmodel.cn/api/paas/v4/",'
                '"thinking":false,'
                '"reasoning_effort":null,'
                '"clear_thinking":true,'
                '"tool_stream":false'
                "}}}}",
                encoding="utf-8",
            )

            config = load_runtime_config(path)
            profile = config.provider.model_profiles["main"]

            self.assertEqual(profile.transport, "chatglm_chat")
            self.assertEqual(profile.chat_model, "glm-4.7")
            self.assertFalse(profile.thinking)
            self.assertIsNone(profile.reasoning_effort)
            self.assertTrue(profile.clear_thinking)
            self.assertFalse(profile.tool_stream)

    def test_discovers_project_root_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "xcode.config.json").write_text(
                '{"agent":{"max_steps":13}}',
                encoding="utf-8",
            )

            config = discover_runtime_config(root)

            self.assertEqual(config.agent.max_steps, 13)

    def test_runtime_config_converts_to_core_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                '{"agent":{"max_steps":8,"tool_workers":1}}',
                encoding="utf-8",
            )

            config = load_runtime_config(path)
            agent = to_agent_config(config)

            self.assertEqual(agent.max_steps, 8)
            self.assertEqual(agent.tool_workers, 1)
            self.assertEqual(agent.execution_mode, "act")

    def test_resolve_config_path_keeps_absolute_and_roots_relative(self) -> None:
        root = Path("project").resolve()
        absolute = root / "absolute"

        self.assertEqual(resolve_config_path(root, None), None)
        self.assertEqual(resolve_config_path(root, absolute), absolute)
        self.assertEqual(resolve_config_path(root, Path("docs")), root / "docs")

    def test_repl_entry_consumes_explicit_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "xcode.config.json"
            config_path.write_text(
                '{"agent":{"max_steps":11,"tool_workers":3},'
                '"paths":{"sessions_dir":"sessions",'
                '"skills_dir":"skills"},'
                '"observability":{"audit_path":"audit.jsonl"}}',
                encoding="utf-8",
            )

            args = SimpleNamespace(
                prompt=None,
                project_root=root,
                sessions_dir=None,
                config=config_path,
                resume=False,
                setup=False,
            )

            captured: dict[str, object] = {}

            def fake_build_app(**kwargs):
                captured.update(kwargs)
                return object()

            def fake_run_repl(app, sessions_dir, **_kwargs):
                captured["app"] = app
                captured["sessions_dir"] = sessions_dir
                return 0

            with (
                patch("xcode.main.parse_args", return_value=args),
                patch(
                    "xcode.main.build_app",
                    side_effect=fake_build_app,
                ),
                patch("xcode.main.run_repl", side_effect=fake_run_repl),
                patch("xcode.main.has_valid_config", return_value=True),
            ):
                import xcode.main as main_module

                self.assertEqual(main_module.main(), 0)

            runtime: Any = captured["runtime_config"]
            self.assertEqual(runtime.agent.max_steps, 11)
            self.assertEqual(runtime.agent.tool_workers, 3)
            self.assertEqual(runtime.paths.skills_dir, Path("skills"))
            self.assertEqual(runtime.observability.audit_path, Path("audit.jsonl"))
            self.assertEqual(captured["sessions_dir"], root / "sessions")

    def test_repl_entry_discovers_project_root_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "xcode.config.json").write_text(
                '{"agent":{"max_steps":12},"paths":{"sessions_dir":"sessions"}}',
                encoding="utf-8",
            )
            args = SimpleNamespace(
                prompt=None,
                project_root=root,
                sessions_dir=None,
                config=None,
                resume=False,
                setup=False,
            )
            captured: dict[str, object] = {}

            def fake_build_app(project_root, runtime_config):
                captured["project_root"] = project_root
                captured["runtime_config"] = runtime_config
                return object()

            def fake_run_repl(app, sessions_dir, **_kwargs):
                captured["app"] = app
                captured["sessions_dir"] = sessions_dir
                return 0

            with (
                patch("xcode.main.parse_args", return_value=args),
                patch(
                    "xcode.main._build_app_from_config",
                    side_effect=fake_build_app,
                ),
                patch("xcode.main.run_repl", side_effect=fake_run_repl),
                patch("xcode.main.has_valid_config", return_value=True),
            ):
                import xcode.main as main_module

                self.assertEqual(main_module.main(), 0)

            runtime: Any = captured["runtime_config"]
            self.assertEqual(runtime.agent.max_steps, 12)
            self.assertEqual(captured["sessions_dir"], root / "sessions")

    def test_prompt_entry_uses_project_root_and_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                prompt="hello",
                project_root=root,
                sessions_dir=None,
                config=None,
                resume=False,
                setup=False,
            )
            captured: dict[str, object] = {}

            def fake_build_app(project_root, runtime_config):
                captured["project_root"] = project_root
                captured["runtime_config"] = runtime_config
                return SimpleNamespace(ask_stream=lambda prompt: iter([]))

            with (
                patch("xcode.main.parse_args", return_value=args),
                patch("xcode.main.has_valid_config", return_value=True),
                patch(
                    "xcode.main._build_app_from_config",
                    side_effect=fake_build_app,
                ),
                patch("builtins.print"),
            ):
                import xcode.main as main_module

                self.assertEqual(main_module.main(), 0)

            self.assertEqual(captured["project_root"], root)

    def test_resume_flag_opens_repl_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                prompt=None,
                project_root=root,
                sessions_dir=None,
                config=None,
                resume=True,
                setup=False,
            )
            captured: dict[str, object] = {}

            def fake_build_app(project_root, runtime_config):
                captured["project_root"] = project_root
                captured["runtime_config"] = runtime_config
                return object()

            def fake_run_repl(app, sessions_dir, **kwargs):
                captured["app"] = app
                captured["sessions_dir"] = sessions_dir
                captured.update(kwargs)
                return 0

            with (
                patch("xcode.main.parse_args", return_value=args),
                patch(
                    "xcode.main._build_app_from_config",
                    side_effect=fake_build_app,
                ),
                patch("xcode.main.run_repl", side_effect=fake_run_repl),
                patch("xcode.main.has_valid_config", return_value=True),
            ):
                import xcode.main as main_module

                self.assertEqual(main_module.main(), 0)

            self.assertEqual(captured["project_root"], root)
            self.assertEqual(captured["sessions_dir"], root / ".local" / "sessions")
            self.assertTrue(captured["resume_latest"])


if __name__ == "__main__":
    unittest.main()
