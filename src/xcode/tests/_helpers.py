from __future__ import annotations

from unittest import TestCase as _TestCase

# Direct reuse of unittest's assertLogs logic — works correctly because
# assertLogs does not depend on TestCase instance state.
assert_logs = _TestCase().assertLogs

# assertNoLogs (Python 3.10+)
_inst = _TestCase()
assert_no_logs = _inst.assertNoLogs  # type: ignore[attr-defined]  # assertNoLogs 在 typeshed 3.9 以下不存在，运行时需要 Python 3.10+
