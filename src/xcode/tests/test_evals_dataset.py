"""版本化 Task 数据集契约测试。"""

from pathlib import Path

from xcode.evals.dataset import load_tasks


def test_xcode_history_dataset_loads_without_hidden_material() -> None:
    repository = Path(__file__).parents[3]
    tasks = load_tasks(repository / "evals/datasets/xcode-history-v1")

    assert len(tasks) == 7
    task = next(
        task for task in tasks if task.task_id == "xcode-set-model-preserves-fallback"
    )
    assert task.task_id == "xcode-set-model-preserves-fallback"
    payload = task.model_dump_json()
    assert "337b989" not in payload
    assert "replace_primary" not in payload
    assert "verifier.py" not in payload

    watchdog = next(
        task
        for task in tasks
        if task.task_id == "xcode-watchdog-preserves-tool-error-signal"
    )
    watchdog_payload = watchdog.model_dump_json()
    assert "da58a39" not in watchdog_payload
    assert "tool_results" not in watchdog_payload

    thinking = next(
        task for task in tasks if task.task_id == "xcode-thinking-off-overrides-effort"
    )
    thinking_payload = thinking.model_dump_json()
    assert "6c1a27f" not in thinking_payload
    assert "_build_thinking_params" not in thinking_payload

    session = next(
        task
        for task in tasks
        if task.task_id == "xcode-session-reset-clears-temporary-grants"
    )
    session_payload = session.model_dump_json()
    assert "797bce1" not in session_payload
    assert "clear_session_grants" not in session_payload

    observer = next(
        task
        for task in tasks
        if task.task_id == "xcode-observer-hooks-do-not-block-agent"
    )
    observer_payload = observer.model_dump_json()
    assert "dd296ea" not in observer_payload
    assert "register_background" not in observer_payload

    mcp_override = next(
        task
        for task in tasks
        if task.task_id == "xcode-mcp-unknown-overrides-are-diagnosed"
    )
    mcp_payload = mcp_override.model_dump_json()
    assert "994bc24" not in mcp_payload
    assert "_warn_unknown_overrides" not in mcp_payload

    worker_limit = next(
        task
        for task in tasks
        if task.task_id == "xcode-parallel-tools-respect-worker-limit"
    )
    worker_payload = worker_limit.model_dump_json()
    assert "5691546" not in worker_payload
    assert "asyncio.Semaphore" not in worker_payload

    metadata = repository / "evals/datasets/xcode-history-v1/dataset.json"
    assert metadata.is_file()
    assert '"task_count": 7' in metadata.read_text(encoding="utf-8")
