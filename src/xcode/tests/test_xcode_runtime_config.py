from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
from typing import Any

from xcode.harness.config import (
    DEFAULT_PROMPT_MODULES,
    discover_runtime_config,
    load_runtime_config,
    resolve_config_path,
)
import pytest


class XcodeRuntimeConfigMergeSemanticsTests:
    """验证配置合并语义：显式设置的默认值不会被静默丢弃。"""

    def test_global_overridden_by_project_with_default_value(self) -> None:
        """项目 config 显式设置与默认值相同的字段应覆盖全局。"""
        root = Path(tempfile.mkdtemp())
        try:
            (root / "xcode.config.json").write_text(
                '{"agent":{"max_steps":20}}',
                encoding="utf-8",
            )
            global_home = root / "global_home"
            global_home.mkdir(parents=True, exist_ok=True)
            global_config = global_home / ".xcode" / "settings.json"
            global_config.parent.mkdir(parents=True, exist_ok=True)
            global_config.write_text(
                '{"agent":{"max_steps":10}}',
                encoding="utf-8",
            )

            with (
                patch.object(Path, "home", return_value=global_home),
            ):
                config = discover_runtime_config(root)

            # Project explicitly set max_steps=20 (same as default of 20),
            # but it must override global's 10.
            assert config.agent.max_steps == 20
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_explicit_default_valued_field_survives_merge(self) -> None:
        """显式设回默认值的字段在合并后应正确保留。"""
        root = Path(tempfile.mkdtemp())
        try:
            config_path = root / "xcode.config.json"
            config_path.write_text(
                '{"provider":{"model_profiles":{"main":{"thinking":true}}}}',
                encoding="utf-8",
            )
            local_path = root / ".local" / "settings.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(
                '{"agent":{"max_steps":20}}',
                encoding="utf-8",
            )
            config = discover_runtime_config(root)

            # thinking=true 是默认值，但用户显式设置了它，不应被丢弃
            assert config.provider.model_profiles["main"].thinking
            assert config.agent.max_steps == 20
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_missing_key_does_not_override_lower_layer(self) -> None:
        """高层级配置中不存在的键不应覆盖低层级的值。"""
        root = Path(tempfile.mkdtemp())
        try:
            config_path = root / "xcode.config.json"
            config_path.write_text(
                '{"agent":{"max_steps":15}}',
                encoding="utf-8",
            )
            local_path = root / ".local" / "settings.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # 只写 provider，不写 agent
            local_path.write_text(
                '{"provider":{"model_profiles":{"main":{"transport":"deepseek_chat"}}}}',
                encoding="utf-8",
            )
            config = discover_runtime_config(root)

            # agent.max_steps 不在 local 中出现，应从 project 继承
            assert config.agent.max_steps == 15
            assert config.provider.model_profiles["main"].transport == "deepseek_chat"
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_env_override_still_wins(self) -> None:
        """环境变量优先级仍然最高。"""
        root = Path(tempfile.mkdtemp())
        try:
            config_path = root / "xcode.config.json"
            config_path.write_text(
                '{"security":{"sandbox_mode":false}}',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"XCODE_SANDBOX_MODE": "true"}):
                config = discover_runtime_config(root)
            assert config.security.sandbox_mode
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_unknown_key_handling_unchanged(self) -> None:
        """未知键被静默忽略（与旧行为一致）。"""
        root = Path(tempfile.mkdtemp())
        try:
            config_path = root / "xcode.config.json"
            config_path.write_text(
                '{"unknown_key":"value"}',
                encoding="utf-8",
            )
            config = discover_runtime_config(root)
            # 不抛出异常即为通过
            assert config is not None
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_nested_profile_merge_preserves_explicit_default(self) -> None:
        """model_profiles 中显式设为默认值的字段正确保留。"""
        root = Path(tempfile.mkdtemp())
        try:
            config_path = root / "xcode.config.json"
            config_path.write_text(
                '{"provider":{"model_profiles":{'
                '"main":{"transport":"deepseek_chat","reasoning_effort":null}'
                "}}}",
                encoding="utf-8",
            )
            config = discover_runtime_config(root)
            profile = config.provider.model_profiles["main"]

            assert profile.transport == "deepseek_chat"
            assert profile.reasoning_effort is None
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_project_hook_list_replaces_global_hook_list(self) -> None:
        """高优先级 hooks.entries 按列表替换语义覆盖全局声明。"""
        root = Path(tempfile.mkdtemp())
        try:
            global_home = root / "global_home"
            global_path = global_home / ".xcode" / "settings.json"
            global_path.parent.mkdir(parents=True)
            global_path.write_text(
                '{"hooks":{"entries":[{"event":"post_tool",'
                '"command":["global-hook"]}]}}',
                encoding="utf-8",
            )
            project_path = root / "xcode.config.json"
            project_path.write_text(
                '{"hooks":{"entries":[{"event":"pre_tool",'
                '"command":["project-hook"],"matcher":"bash",'
                '"timeout":2.5,"enabled":false,'
                '"failure_policy":"fail",'
                '"inherit_to_subagents":true}]}}',
                encoding="utf-8",
            )

            with patch.object(Path, "home", return_value=global_home):
                config = discover_runtime_config(root)

            assert len(config.hooks.entries) == 1
            hook = config.hooks.entries[0]
            assert hook.event == "pre_tool"
            assert hook.command == ("project-hook",)
            assert hook.matcher == "bash"
            assert hook.timeout == 2.5
            assert not (hook.enabled)
            assert hook.failure_policy == "fail"
            assert hook.inherit_to_subagents
            assert hook.source == str(project_path)
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


class XcodeRuntimeConfigTests:
    def test_missing_config_uses_defaults(self) -> None:
        config = load_runtime_config(None)

        assert config.provider.model_profiles["main"].transport == "openai_chat"
        assert config.provider.model_profiles["main"].chat_model == "deepseek-v4-flash"
        assert (
            config.provider.model_profiles["main"].base_url
            == "https://api.deepseek.com"
        )
        assert (
            config.provider.model_profiles["subagent"]
            == config.provider.model_profiles["main"]
        )
        assert config.agent.max_steps == 20
        assert config.prompt.modules == DEFAULT_PROMPT_MODULES
        assert config.paths.sessions_dir is None
        assert not (config.daemon.enabled)
        assert config.daemon.interval_seconds == 30
        assert config.hooks.entries == ()

    def test_loads_runtime_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                '{"provider":{"model_profiles":{'
                '"main":{"transport":"openai_chat",'
                '"chat_model":"main-test","base_url":"https://main.test"},'
                '"subagent":{"chat_model":"subagent-test"}}},'
                '"agent":{"max_steps":9,"compact_threshold":20,'
                '"compact_token_threshold":3000,"max_recent_messages":7,'
                '"tool_workers":2,"subagent_workers":3,'
                '"watchdog_repeated_tool_limit":4},'
                '"paths":{"sessions_dir":"sessions",'
                '"skills_dir":"skills"},'
                '"observability":{"audit_path":"audit.jsonl"},'
                '"tools":{"network_commands":"deny"},'
                '"skills":{},'
                '"prompt":{"modules":["identity","tools"]},'
                '"daemon":{"enabled":true,"interval_seconds":15}}',
                encoding="utf-8",
            )

            config = load_runtime_config(path)

            assert config.provider.model_profiles["main"].transport == "openai_chat"
            assert config.provider.model_profiles["main"].chat_model == "main-test"
            assert (
                config.provider.model_profiles["main"].base_url == "https://main.test"
            )
            assert (
                config.provider.model_profiles["subagent"].chat_model == "subagent-test"
            )
            assert config.prompt.modules == ("identity", "tools")
            assert config.agent.max_steps == 9
            assert config.agent.tool_workers == 2
            assert config.agent.subagent_workers == 3
            assert config.paths.sessions_dir == Path("sessions")
            assert config.paths.skills_dir == Path("skills")
            assert config.observability.audit_path == Path("audit.jsonl")
            assert config.daemon.enabled
            assert config.daemon.interval_seconds == 15

    def test_loads_external_hook_config(self) -> None:
        """外部 hook 声明转换为类型化 argv 配置。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                '{"hooks":{"entries":['
                '{"event":"before_provider_request",'
                '"command":["python","hook.py"],'
                '"matcher":"main","timeout":3,'
                '"failure_policy":"ignore"}'
                "]}}",
                encoding="utf-8",
            )

            config = load_runtime_config(path)

        assert len(config.hooks.entries) == 1
        hook = config.hooks.entries[0]
        assert hook.command == ("python", "hook.py")
        assert hook.timeout == 3
        assert hook.source == str(path)

    @pytest.mark.parametrize(
        "entry",
        [
            {"event": "unknown", "command": ["hook"]},
            {"event": "pre_tool", "command": "hook"},
            {"event": "pre_tool", "command": []},
            {"event": "pre_tool", "command": [""]},
            {"event": "pre_tool", "command": ["hook"], "matcher": ""},
            {"event": "pre_tool", "command": ["hook"], "timeout": 0},
            {"event": "pre_tool", "command": ["hook"], "enabled": "yes"},
            {
                "event": "pre_tool",
                "command": ["hook"],
                "failure_policy": "sometimes",
            },
            {
                "event": "pre_tool",
                "command": ["hook"],
                "inherit_to_subagents": "yes",
            },
        ],
    )
    def test_rejects_invalid_external_hook_entries(self, entry: dict) -> None:
        """无效 event、argv、timeout 和策略会 fail-fast。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                json.dumps({"hooks": {"entries": [entry]}}),
                encoding="utf-8",
            )
            with pytest.raises(ValueError):
                load_runtime_config(path)

    def test_loads_chatglm_profile_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                '{"provider":{"model_profiles":{"main":{'
                '"transport":"chatglm_chat", '
                '"chat_model":"glm-4.7",'
                '"base_url":"https://open.bigmodel.cn/api/paas/v4/",'
                '"thinking":false,'
                '"reasoning_effort":null,'
                '"clear_thinking":true,'
                '"tool_stream":false,'
                '"response_format":{"type":"json_object"}'
                "}}}}",
                encoding="utf-8",
            )

            config = load_runtime_config(path)
            profile = config.provider.model_profiles["main"]

            assert profile.transport == "chatglm_chat"
            assert profile.chat_model == "glm-4.7"
            assert not (profile.thinking)
            assert profile.reasoning_effort is None
            assert profile.clear_thinking
            assert not (profile.tool_stream)
            assert profile.response_format == {"type": "json_object"}

    def test_discovers_project_root_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "xcode.config.json").write_text(
                '{"agent":{"max_steps":13}}',
                encoding="utf-8",
            )

            config = discover_runtime_config(root)

            assert config.agent.max_steps == 13

    def test_runtime_config_converts_to_core_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xcode.config.json"
            path.write_text(
                '{"agent":{"max_steps":8,"tool_workers":1}}',
                encoding="utf-8",
            )

            config = load_runtime_config(path)
            agent = config.agent

            assert agent.max_steps == 8
            assert agent.tool_workers == 1
            assert agent.execution_mode == "act"

    def test_resolve_config_path_keeps_absolute_and_roots_relative(self) -> None:
        root = Path("project").resolve()
        absolute = root / "absolute"

        assert resolve_config_path(root, None) is None
        assert resolve_config_path(root, absolute) == absolute
        assert resolve_config_path(root, Path("docs")) == root / "docs"

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
                continue_=False,
                session=None,
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

                assert main_module.main() == 0

            runtime: Any = captured["runtime_config"]
            assert runtime.agent.max_steps == 11
            assert runtime.agent.tool_workers == 3
            assert runtime.paths.skills_dir == Path("skills")
            assert runtime.observability.audit_path == Path("audit.jsonl")
            assert captured["sessions_dir"] == root / "sessions"

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
                continue_=False,
                session=None,
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

                assert main_module.main() == 0

            runtime: Any = captured["runtime_config"]
            assert runtime.agent.max_steps == 12
            assert captured["sessions_dir"] == root / "sessions"

    def test_prompt_entry_uses_project_root_and_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                prompt="hello",
                project_root=root,
                sessions_dir=None,
                config=None,
                resume=False,
                continue_=False,
                session=None,
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

                assert main_module.main() == 0

            assert captured["project_root"] == root

    def test_resume_flag_opens_repl_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                prompt=None,
                project_root=root,
                sessions_dir=None,
                config=None,
                resume=True,
                continue_=False,
                session=None,
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

                assert main_module.main() == 0

            assert captured["project_root"] == root
            assert captured["sessions_dir"] == root / ".local" / "sessions"
            assert captured["resume_latest"]


if __name__ == "__main__":
    pytest.main()
