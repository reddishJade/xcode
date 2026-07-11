"""结构化上下文块收集、组装与预算裁剪单元测试。"""

from __future__ import annotations

from xcode.agent.context import (
    ContextBlock,
    ContextBlockSource,
    ContextPriority,
    ContextExpiry,
    ContextBlockTarget,
    ContextCollectionInput,
    ContextAssemblyInput,
    DefaultContextAssembler,
    trim_to_budget,
    _is_expired,
    _block_to_text,
    _apply_size_budget,
    _utf8_prefix,
    _condense_manifest,
    _extract_key_sections,
    _drop_fenced_blocks,
    _prepare_manifest,
    MANIFEST_MAX_BYTES,
    ContextCollectorRegistry,
)
from xcode.agent.messages import SystemMessage, UserMessage


# ── ContextBlock ──


class TestContextBlock:
    def test_get_token_count_returns_cached(self) -> None:
        block = ContextBlock(
            source=ContextBlockSource.INSTRUCTION,
            priority=ContextPriority.CRITICAL,
            content="hello",
            token_count=5,
        )
        assert block.get_token_count() == 5

    def test_get_token_count_estimates(self) -> None:
        block = ContextBlock(
            source=ContextBlockSource.INSTRUCTION,
            priority=ContextPriority.CRITICAL,
            content="hello world",
        )
        assert block.get_token_count() > 0

    def test_expiry_never(self) -> None:
        e = ContextExpiry()
        assert e.never
        e2 = ContextExpiry(max_turns=5)
        assert not e2.never


# ── _is_expired ──


class TestIsExpired:
    def test_no_expiry(self) -> None:
        block = ContextBlock(
            source=ContextBlockSource.INSTRUCTION,
            priority=ContextPriority.CRITICAL,
            content="x",
        )
        assert not _is_expired(block, 10, 10)

    def test_never_expiry(self) -> None:
        block = ContextBlock(
            source=ContextBlockSource.INSTRUCTION,
            priority=ContextPriority.CRITICAL,
            content="x",
            expiry=ContextExpiry(),
        )
        assert not _is_expired(block, 10, 10)

    def test_expired_by_turns(self) -> None:
        block = ContextBlock(
            source=ContextBlockSource.INSTRUCTION,
            priority=ContextPriority.CRITICAL,
            content="x",
            expiry=ContextExpiry(max_turns=3),
            created_turn=0,
        )
        assert _is_expired(block, 5, 0)


# ── _block_to_text ──


class TestBlockToText:
    def test_basic(self) -> None:
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.MEDIUM,
            content="some notes",
        )
        text = _block_to_text(block)
        assert "[notes]" in text
        assert "some notes" in text

    def test_with_metadata(self) -> None:
        block = ContextBlock(
            source=ContextBlockSource.ACTIVE_DIFF,
            priority=ContextPriority.HIGH,
            content="diff content",
            metadata={"files": 3},
        )
        text = _block_to_text(block)
        assert "[active_diff]" in text
        assert "files=3" in text


# ── trim_to_budget ──


class TestTrimToBudget:
    def test_negative_budget_returns_all(self) -> None:
        blocks = [
            ContextBlock(
                source=ContextBlockSource.INSTRUCTION,
                priority=ContextPriority.CRITICAL,
                content="x",
            )
        ]
        used, dropped = trim_to_budget(blocks, budget=-1, base_tokens=0)
        assert len(used) == 1
        assert len(dropped) == 0

    def test_budget_exceeded_by_base(self) -> None:
        blocks = [
            ContextBlock(
                source=ContextBlockSource.INSTRUCTION,
                priority=ContextPriority.CRITICAL,
                content="x",
            )
        ]
        used, dropped = trim_to_budget(blocks, budget=5, base_tokens=10)
        assert len(used) == 0
        assert len(dropped) == 1

    def test_budget_selects_higher_priority_first(self) -> None:
        blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.LOW,
                content="x" * 100,
                token_count=50,
            ),
            ContextBlock(
                source=ContextBlockSource.INSTRUCTION,
                priority=ContextPriority.CRITICAL,
                content="important",
                token_count=20,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=30, base_tokens=0)
        assert len(used) == 1
        assert used[0].priority == ContextPriority.CRITICAL


# ── DefaultContextAssembler ──


class TestDefaultContextAssembler:
    def test_no_blocks_passthrough(self) -> None:
        assembler = DefaultContextAssembler()
        result = assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[],
            )
        )
        assert len(result.messages) == 1
        assert result.total_tokens > 0

    def test_system_blocks_inserted_after_system_message(self) -> None:
        assembler = DefaultContextAssembler()
        result = assembler.assemble(
            ContextAssemblyInput(
                messages=[
                    SystemMessage(content="You are helpful."),
                    UserMessage(content="hello"),
                ],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="Extra instruction",
                    ),
                ],
            )
        )
        assert len(result.messages) == 3
        assert isinstance(result.messages[1], SystemMessage)
        assert result.messages[1].content == "Extra instruction"

    def test_expired_blocks_dropped(self) -> None:
        assembler = DefaultContextAssembler()
        result = assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.LOW,
                        content="old note",
                        expiry=ContextExpiry(max_turns=1),
                        created_turn=0,
                    ),
                ],
                current_turn=10,
            )
        )
        assert len(result.blocks_dropped) == 1

    def test_budget_drops_excess_blocks(self) -> None:
        assembler = DefaultContextAssembler()
        result = assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hi")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.LOW,
                        content="x" * 10000,
                    ),
                ],
                token_budget=50,
            )
        )
        assert len(result.blocks_dropped) == 1


# ── _apply_size_budget ──


class TestApplySizeBudget:
    def test_within_budget(self) -> None:
        assert _apply_size_budget("hello", 100, "...") == "hello"

    def test_exceeds_budget(self) -> None:
        result = _apply_size_budget("hello world", 5, "...")
        assert "(truncated)" not in result
        assert result.endswith("...") or len(result) <= 8

    def test_empty_content(self) -> None:
        assert _apply_size_budget("", 100, "...") == ""


# ── _utf8_prefix ──


class TestUtf8Prefix:
    def test_short_text(self) -> None:
        assert _utf8_prefix("hello", 100) == "hello"

    def test_truncated(self) -> None:
        result = _utf8_prefix("hello world", 5)
        assert len(result) <= 5


# ── _condense_manifest ──


class TestCondenseManifest:
    def test_short_text_returns_as_is(self) -> None:
        text = "Short content here"
        assert _prepare_manifest(text) == text

    def test_long_manifest_gets_truncated_and_tagged(self) -> None:
        text = "x" * (MANIFEST_MAX_BYTES + 1000)
        result = _condense_manifest(text)
        assert "<manifest-truncated>" in result

    def test_key_sections_preserved(self) -> None:
        text = (
            "Opening context\n"
            "## Priority\n"
            "- item 1\n\n"
            "## Checklist\n"
            "- check A\n\n" + "x" * 50000
        )
        result = _condense_manifest(text)
        assert "Priority" in result
        assert "Checklist" in result

    def test_non_key_section_not_in_key_sections(self) -> None:
        text = "Opening\n## Random Section\n- stuff\n\n" + "y" * 50000
        sections = _extract_key_sections(text)
        assert not any("random section" in s.lower() for s in sections)


# ── _extract_key_sections ──


class TestExtractKeySections:
    def test_extracts_matching_sections(self) -> None:
        text = "## Priority\n- high\n\n## Git Safety\n- never rebase\n"
        sections = _extract_key_sections(text)
        assert len(sections) >= 1
        assert any("priority" in s.lower() for s in sections)

    def test_ignores_non_matching_sections(self) -> None:
        text = "## Unrelated Topic\n- stuff\n"
        assert _extract_key_sections(text) == []


# ── _drop_fenced_blocks ──


class TestDropFencedBlocks:
    def test_removes_fenced_content(self) -> None:
        lines = ["line1", "```", "hidden", "```", "line2"]
        result = _drop_fenced_blocks(lines)
        assert result == ["line1", "line2"]


# ── _prepare_manifest ──


class TestPrepareManifest:
    def test_short_passthrough(self) -> None:
        assert _prepare_manifest("small") == "small"

    def test_long_condenses(self) -> None:
        result = _prepare_manifest("x" * (MANIFEST_MAX_BYTES + 100))
        assert "<manifest-truncated>" in result


# ── ContextCollectorRegistry ──


class _DummyCollector:
    def __init__(self, blocks: list[ContextBlock]) -> None:
        self._blocks = blocks

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        return self._blocks


class TestContextCollectorRegistry:
    def test_empty_registry(self) -> None:
        registry = ContextCollectorRegistry()
        assert len(registry) == 0

    def test_collect_aggregates(self) -> None:
        registry = ContextCollectorRegistry()
        registry.register(
            _DummyCollector(
                [
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.LOW,
                        content="a",
                    ),
                ]
            )
        )
        registry.register(
            _DummyCollector(
                [
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        priority=ContextPriority.CRITICAL,
                        content="b",
                    ),
                ]
            )
        )
        blocks = registry.collect(ContextCollectionInput())
        assert len(blocks) == 2

    def test_collector_exception_skipped(self) -> None:
        class BrokenCollector:
            def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
                raise RuntimeError("boom")

        registry = ContextCollectorRegistry()
        registry.register(BrokenCollector())
        registry.register(
            _DummyCollector(
                [
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.LOW,
                        content="ok",
                    ),
                ]
            )
        )
        blocks = registry.collect(ContextCollectionInput())
        assert len(blocks) == 1
