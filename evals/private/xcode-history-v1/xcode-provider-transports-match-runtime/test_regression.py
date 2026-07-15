"""Agent 不可见的已支持 provider transport 配置回归。"""

import json
from pathlib import Path

import pytest

from xcode.harness.config import load_runtime_config


@pytest.mark.parametrize(
    "transport",
    ["openai_chat", "chatglm_chat", "deepseek_chat", "mimo_chat"],
)
def test_supported_transport_survives_config_loading(
    tmp_path: Path,
    transport: str,
) -> None:
    config_path = tmp_path / "xcode.config.json"
    config_path.write_text(
        json.dumps(
            {"provider": {"model_profiles": {"main": {"transport": transport}}}}
        ),
        encoding="utf-8",
    )

    loaded = load_runtime_config(config_path)

    assert loaded.provider.model_profiles["main"].transport == transport
