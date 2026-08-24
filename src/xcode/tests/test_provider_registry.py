"""Provider 工厂配置传播测试。"""

from typing import Any, cast

from xcode.ai.providers._runtime import RateLimitPolicy, RetryPolicy
from xcode.ai.providers.registry import (
    ModelProfileConfig,
    ProviderSettings,
    build_provider_bundle,
)


def test_provider_settings_runtime_reaches_all_profiles() -> None:
    retry = RetryPolicy(max_attempts=7, initial_delay_seconds=0.1)
    rate_limit = RateLimitPolicy(min_interval_seconds=0.25)
    profile = ModelProfileConfig(api_key="test-key", chat_model="test-model")

    bundle = build_provider_bundle(
        ProviderSettings(
            env_files=(),
            model_profiles={"main": profile, "subagent": profile},
            retry=retry,
            rate_limit=rate_limit,
        )
    )

    runtimes = []
    for provider in bundle.llms.values():
        runtime = cast(Any, provider).runtime
        runtimes.append(runtime)
        assert runtime.retry == retry
        assert runtime.rate_limit == rate_limit
    assert len({id(runtime) for runtime in runtimes}) == len(runtimes)
