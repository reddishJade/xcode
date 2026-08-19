from pathlib import Path

from xcode.harness.memory import MemoryManager, build_memory_block, build_memory_tools


def _manager(tmp_path: Path) -> MemoryManager:
    project = tmp_path / "project"
    project.mkdir()
    user_file = tmp_path / "user" / "MEMORY.md"
    return MemoryManager(project, user_memory_file=user_file)


def test_markdown_is_the_only_source_of_truth(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.memory_file.write_text(
        "# Project memory\n\n"
        "## Provider retry rule\n"
        "Retry provider timeouts at most twice.\n\n"
        "## Architecture\n"
        "The harness owns session persistence.\n",
        encoding="utf-8",
    )

    records = manager.read_memory_records(layer="project")

    assert [record.title for record in records] == [
        "Provider retry rule",
        "Architecture",
    ]
    assert not (manager.root / ".xcode" / "memory_lru.json").exists()


def test_bm25_search_supports_code_and_chinese(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.memory_file.write_text(
        "# Project memory\n\n"
        "## Provider timeout\n"
        "ProviderClient retries connection timeout twice.\n\n"
        "## 终端规则\n"
        "终端超时后不得无限重试。\n",
        encoding="utf-8",
    )

    english = manager.search_memory_records("ProviderClient timeout")
    chinese = manager.search_memory_records("终端超时")

    assert english[0].title == "Provider timeout"
    assert chinese[0].title == "终端规则"


def test_project_and_user_layers_are_explicit(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    assert manager.add_memory_block(
        build_memory_block("Architecture", "Use a layered harness."),
        layer="project",
    )
    assert manager.add_memory_block(
        build_memory_block("Preference", "Prefer concise answers."),
        layer="user",
    )

    assert [record.layer for record in manager.read_memory_records()] == [
        "project",
        "user",
    ]
    assert manager.search_memory_records("concise", layer="project") == []
    assert manager.search_memory_records("concise", layer="user")[0].title == (
        "Preference"
    )


def test_duplicate_title_is_rejected_without_rewriting_existing_file(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    assert manager.add_memory_block(
        build_memory_block("Retry rule", "Retry twice."),
    )
    original = manager.memory_file.read_text(encoding="utf-8")

    assert not manager.add_memory_block(
        build_memory_block("retry RULE", "Retry three times."),
    )
    assert manager.memory_file.read_text(encoding="utf-8") == original


def test_budgeted_read_never_exceeds_budget(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.memory_file.write_text(
        "# Project memory\n\n"
        + build_memory_block("Short", "Keep this durable fact.")
        + "\n"
        + build_memory_block("Long", "token " * 200),
        encoding="utf-8",
    )

    packets = manager.read_budgeted(max_tokens=30)

    assert len(packets) == 1
    assert "Short" in packets[0]


def test_memory_tool_is_small_and_read_only(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.memory_file.write_text(
        "# Project memory\n\n"
        + build_memory_block("Retry rule", "Retry provider timeouts twice."),
        encoding="utf-8",
    )
    (tool,) = build_memory_tools(manager)

    result = tool.handler({"query": "provider timeout", "limit": 3})

    assert "Retry rule" in result
    assert "path=" in result
    assert set(tool.schema["properties"]) == {"query", "limit", "scope", "layer"}


def test_legacy_governance_metadata_does_not_survive_runtime(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    manager.memory_file.write_text(
        "# Project memory\n\n"
        "## Active rule\n"
        "- Solution: Retry twice.\n"
        "- status: active\n"
        "- utility: 42\n"
        "- success-count: 99\n\n"
        "## Retired rule\n"
        "- Solution: Retry forever.\n"
        "- status: superseded\n",
        encoding="utf-8",
    )

    records = manager.read_memory_records(layer="project")

    assert [record.title for record in records] == ["Active rule"]
    assert "utility" not in records[0].block
    assert "success-count" not in records[0].block
