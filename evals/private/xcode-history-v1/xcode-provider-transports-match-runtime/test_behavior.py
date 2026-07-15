"""Agent 不可见的 provider transport 配置行为 oracle。"""

import json
from pathlib import Path
from typing import get_args

import pytest

from xcode.ai.providers import PROVIDER_REGISTRY
from xcode.harness.config import ProviderTransport, load_runtime_config


def test_declared_transports_equal_constructible_runtime_providers() -> None:
    declared = set(get_args(ProviderTransport))
    production = set(PROVIDER_REGISTRY) - {"faux_chat"}

    assert declared == production


@pytest.mark.parametrize("transport", ["anthropic_messages", "openai_typo"])
def test_unsupported_config_transport_fails_fast(
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

    with pytest.raises(ValueError, match="Unsupported provider transport"):
        load_runtime_config(config_path)
