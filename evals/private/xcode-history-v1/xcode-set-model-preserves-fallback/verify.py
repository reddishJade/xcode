"""隐藏 verifier：运行新增行为 oracle 和父版本原有回归套件。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _run(
    command: tuple[str, ...], *, workspace: Path
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace / "src")
    return subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def main() -> int:
    workspace = Path(sys.argv[1]).resolve()
    hidden_root = Path(__file__).parent
    behavior = _run(
        (
            sys.executable,
            "-m",
            "pytest",
            str(hidden_root / "test_behavior.py"),
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ),
        workspace=workspace,
    )
    regression = _run(
        (
            sys.executable,
            "-m",
            "pytest",
            str(workspace / "src/xcode/tests/test_xcode_model_modes.py"),
            (
                str(workspace / "src/xcode/tests/test_xcode_providers.py")
                + "::XcodeProviderEnvTests"
            ),
            (
                str(workspace / "src/xcode/tests/test_xcode_providers.py")
                + "::XcodeProviderRuntimeTests"
            ),
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
        ),
        workspace=workspace,
    )
    payload = {
        "resolved": behavior.returncode == 0,
        "regression_free": regression.returncode == 0,
        "policy_clean": False,
        "details": {
            "behavior_exit_code": behavior.returncode,
            "regression_exit_code": regression.returncode,
            "behavior_stdout": behavior.stdout[-4000:],
            "behavior_stderr": behavior.stderr[-4000:],
            "regression_stdout": regression.stdout[-4000:],
            "regression_stderr": regression.stderr[-4000:],
        },
    }
    (hidden_root / "verifier-result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
