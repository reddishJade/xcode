"""Provider 运行时单元测试。"""

from __future__ import annotations

import pytest

from xcode.ai.providers._runtime import (
    classify_api_error,
    _is_transient_error,
    _try_extract_status_code,
    API_ERROR_MESSAGES,
    ProviderRuntime,
    RetryPolicy,
)


class TestClassifyApiError:
    def test_known_status_code(self) -> None:
        for code, msg in API_ERROR_MESSAGES.items():
            exc = _make_error(code, "detail")
            result = classify_api_error(exc)
            assert msg.split("(")[0].strip() in result
            assert str(code) in result

    def test_unknown_status_code(self) -> None:
        exc = _make_error(418, "teapot")
        result = classify_api_error(exc)
        assert "abnormal" in result
        assert "418" in result

    def test_no_status_code(self) -> None:
        exc = ValueError("connection failed")
        result = classify_api_error(exc)
        assert "Request failed" in result


class TestIsTransientError:
    def test_retryable_status_codes(self) -> None:
        for code in (429, 500, 502, 503, 529):
            assert _is_transient_error(_make_error(code, ""))

    def test_non_retryable_status_codes(self) -> None:
        for code in (400, 401, 403, 404):
            assert not _is_transient_error(_make_error(code, ""))

    def test_timeout_error(self) -> None:
        assert _is_transient_error(TimeoutError("timeout"))

    def test_connection_error(self) -> None:
        assert _is_transient_error(ConnectionError("connection reset"))

    def test_keyword_detection(self) -> None:
        assert _is_transient_error(RuntimeError("rate limit exceeded"))
        assert _is_transient_error(RuntimeError("Too many requests"))
        assert _is_transient_error(RuntimeError("temporary failure"))

    def test_ordinary_exception(self) -> None:
        assert not _is_transient_error(ValueError("bad value"))


class TestExtractStatusCode:
    def test_status_code_attr(self) -> None:
        exc = _make_error(429, "")
        assert _try_extract_status_code(exc) == 429

    def test_status_attr(self) -> None:
        class Exc(Exception):
            def __init__(self) -> None:
                self.status = 503

        assert _try_extract_status_code(Exc()) == 503

    def test_code_attr(self) -> None:
        class Exc(Exception):
            def __init__(self) -> None:
                self.code = 400

        assert _try_extract_status_code(Exc()) == 400

    def test_invalid_values(self) -> None:
        class Exc(Exception):
            def __init__(self) -> None:
                self.status_code = "abc"

        assert _try_extract_status_code(Exc()) is None

    def test_no_match(self) -> None:
        assert _try_extract_status_code(ValueError()) is None


class TestProviderRuntime:
    def test_run_success(self) -> None:
        runtime = ProviderRuntime()
        result = runtime.run(lambda: 42)
        assert result == 42

    def test_run_non_retryable_error_raises(self) -> None:
        runtime = ProviderRuntime()
        with pytest.raises(RuntimeError, match="Request failed"):
            runtime.run(lambda: (_ for _ in ()).throw(ValueError("bad")))

    def test_run_retry_then_success(self) -> None:
        """Verify retry logic: fail twice, succeed on third."""
        attempts: list[int] = []

        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise _make_error(500, "server error")
            return "ok"

        runtime = ProviderRuntime(
            retry=RetryPolicy(max_attempts=5, initial_delay_seconds=0.001)
        )
        result = runtime.run(flaky)
        assert result == "ok"
        assert len(attempts) == 3

    def test_run_max_retries_exhausted(self) -> None:
        attempts: list[int] = []

        def always_fails() -> str:
            attempts.append(1)
            raise _make_error(500, "server error")

        runtime = ProviderRuntime(
            retry=RetryPolicy(max_attempts=3, initial_delay_seconds=0.001)
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            runtime.run(always_fails)
        assert len(attempts) == 3

    def test_rate_limit_waits_between_calls(self) -> None:
        call_times: list[float] = []

        def tracking_now() -> float:
            return call_times[-1] if call_times else 0.0

        def fast_forward(seconds: float) -> None:
            t = call_times[-1] + seconds if call_times else seconds
            call_times.append(t)

        runtime = ProviderRuntime(
            retry=RetryPolicy(max_attempts=1),
            rate_limit=type("RL", (), {"min_interval_seconds": 0.1})(),
            now=tracking_now,
            sleeper=fast_forward,
        )

        # first call: no wait (last_call_at is None)
        runtime._wait_for_rate_limit()

        # second call: should wait because interval hasn't elapsed
        runtime._wait_for_rate_limit()
        assert len(call_times) >= 1  # sleeper was called


def _make_error(status_code: int, detail: str) -> Exception:
    class HttpError(Exception):
        def __init__(self, code: int, msg: str) -> None:
            super().__init__(msg)
            self.status_code = code

    return HttpError(status_code, detail)
