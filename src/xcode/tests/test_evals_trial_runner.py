"""Trial 终止原因与有效性分类测试。"""

from xcode.evals.schema import ErrorCategory
from xcode.evals.trial_runner import _termination_error


def test_provider_error_is_invalid_infrastructure_trial() -> None:
    assert _termination_error("provider_error") is ErrorCategory.PROVIDER_FAILURE


def test_cancelled_run_is_invalid_agent_trial() -> None:
    assert _termination_error("cancelled") is ErrorCategory.AGENT_FAILURE


def test_capability_terminations_remain_independently_verifiable() -> None:
    assert _termination_error("completed") is None
    assert _termination_error("step_limit") is None
    assert _termination_error("watchdog") is None
