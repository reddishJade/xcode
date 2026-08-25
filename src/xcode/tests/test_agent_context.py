"""结构化上下文块收集、组装与预算裁剪单元测试。"""

from __future__ import annotations

from xcode.agent.context import (
    MANIFEST_MAX_BYTES,
    ContextAssemblyInput,
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextCollectionInput,
    ContextCollectorRegistry,
    ContextState,
    ContextExpiry,
    ContextPriority,
    DefaultContextAssembler,
    InstructionCollector,
    make_collector_section,
    make_state_section,
    _apply_size_budget,
    _block_to_text,
    _is_expired,
    _prepare_manifest,
    _utf8_prefix,
    trim_to_budget,
)
from xcode.agent.messages import SystemMessage, UserMessage
from xcode.agent.types import ToolSpec, ToolSpecAdapter

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
        used, _dropped = trim_to_budget(blocks, budget=30, base_tokens=0)
        assert len(used) == 1
        assert used[0].priority == ContextPriority.CRITICAL

    def test_same_priority_preserves_collector_order(self) -> None:
        blocks = [
            ContextBlock(
                source=ContextBlockSource.INSTRUCTION,
                priority=ContextPriority.CRITICAL,
                content="first",
                token_count=5,
                block_id="first",
            ),
            ContextBlock(
                source=ContextBlockSource.INSTRUCTION,
                priority=ContextPriority.CRITICAL,
                content="second",
                token_count=1,
                block_id="second",
            ),
        ]

        used, _dropped = trim_to_budget(blocks, budget=6, base_tokens=0)

        assert [block.block_id for block in used] == ["first", "second"]


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

    def test_budget_includes_prompt_and_tool_tokens(self) -> None:
        tool = ToolSpecAdapter(
            ToolSpec(
                name="read_file",
                description="Read a file.",
                input_hint="path",
                handler=lambda _data, _update=None: "contents",
                schema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        )
        base_input = ContextAssemblyInput(
            system_prompt="system instructions " * 20,
            messages=[UserMessage(content="history " * 20)],
            tools=[tool],
        )
        assembler = DefaultContextAssembler()
        base = assembler.assemble(base_input).total_tokens
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.LOW,
            content="keep this note",
            token_count=1,
        )

        result = assembler.assemble(
            ContextAssemblyInput(
                system_prompt=base_input.system_prompt,
                messages=base_input.messages,
                tools=base_input.tools,
                context_blocks=[block],
                token_budget=base,
            )
        )

        assert result.blocks_dropped == [block]
        assert result.base_tokens == base


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


# ── _prepare_manifest ──


class TestPrepareManifest:
    def test_short_text_returns_as_is(self) -> None:
        text = "Short content here"
        assert _prepare_manifest(text) == text

    def test_long_manifest_keeps_byte_limited_prefix(self) -> None:
        text = "x" * (MANIFEST_MAX_BYTES + 1000)
        result = _prepare_manifest(text)
        assert result == text[:MANIFEST_MAX_BYTES]
        assert len(result.encode("utf-8")) == MANIFEST_MAX_BYTES

    def test_long_manifest_keeps_only_prefix(self) -> None:
        text = (
            "Opening context\n"
            "## Priority\n"
            "- item 1\n\n"
            "## Checklist\n"
            "- check A\n\n" + "x" * 50000
        )
        result = _prepare_manifest(text)
        assert result == text[:MANIFEST_MAX_BYTES]

    def test_long_condenses(self) -> None:
        result = _prepare_manifest("x" * (MANIFEST_MAX_BYTES + 100))
        assert result == "x" * MANIFEST_MAX_BYTES


# ── InstructionCollector ──


class TestInstructionCollector:
    def test_instruction_sources_share_one_byte_budget(self, tmp_path) -> None:
        first = tmp_path / "first.md"
        first.write_bytes(b"a" * 20_000)
        (tmp_path / "AGENTS.md").write_bytes(b"b" * 20_000)

        collector = InstructionCollector(
            sources=({"type": "file", "path": "first.md"},),
            project_root=tmp_path,
        )
        blocks = collector.collect(ContextCollectionInput())

        assert len(blocks) == 2
        assert blocks[0].content == "a" * 20_000
        assert blocks[1].content == "b" * (MANIFEST_MAX_BYTES - 20_000)

    def test_fenced_content_is_preserved(self, tmp_path) -> None:
        agents = tmp_path / "AGENTS.md"
        content = "```powershell\n重要命令\n```\n"
        agents.write_bytes(content.encode("utf-8"))

        collector = InstructionCollector(project_root=tmp_path)
        blocks = collector.collect(ContextCollectionInput())

        assert len(blocks) == 1
        assert blocks[0].content == content

    def test_hierarchy_is_loaded_from_root_to_cwd_and_override_wins(
        self, tmp_path
    ) -> None:
        child = tmp_path / "child"
        nested = child / "nested"
        nested.mkdir(parents=True)
        (tmp_path / "AGENTS.md").write_text("root", encoding="utf-8")
        (child / "AGENTS.md").write_text("child", encoding="utf-8")
        (child / "AGENTS.override.md").write_text("override", encoding="utf-8")
        (nested / "AGENTS.md").write_text("nested", encoding="utf-8")

        blocks = InstructionCollector(project_root=tmp_path).collect(
            ContextCollectionInput(cwd=nested)
        )

        assert [block.content for block in blocks] == ["root", "override", "nested"]
        assert all(block.scope == "project" for block in blocks)


class TestWorldState:
    def test_unchanged_section_is_not_rendered_twice(self) -> None:
        class _Collector:
            def __init__(self) -> None:
                self.calls = 0

            def collect(self, _input: ContextCollectionInput) -> list[ContextBlock]:
                self.calls += 1
                return [
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        priority=ContextPriority.CRITICAL,
                        target=ContextBlockTarget.SYSTEM,
                        content="stable instruction",
                        block_id="stable",
                    )
                ]

        collector = _Collector()
        registry = ContextCollectorRegistry()
        registry.register_section(make_collector_section("agents", collector))
        assembler = DefaultContextAssembler()
        state = ContextState()
        from xcode.agent.config import AgentContext
        from xcode.agent.request import DefaultRequestAssembler

        request_assembler = DefaultRequestAssembler(
            context_collectors=registry,
            context_assembler=assembler,
        )
        context = AgentContext(context_state=state)

        first = request_assembler.assemble(context, current_step=1, options=None)
        second = request_assembler.assemble(context, current_step=2, options=None)

        assert collector.calls == 2
        assert "stable instruction" in str(first.messages)
        assert "stable instruction" in str(second.messages)
        assert [trace.block_id for trace in first.context_trace] == ["stable"]
        assert second.context_trace == ()

    def test_changed_section_renders_a_replacement_notice(self) -> None:
        class _Collector:
            def __init__(self) -> None:
                self.value = "before"

            def collect(self, _input: ContextCollectionInput) -> list[ContextBlock]:
                return [
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.MEDIUM,
                        target=ContextBlockTarget.USER_CONTEXT,
                        content=self.value,
                    )
                ]

        collector = _Collector()
        registry = ContextCollectorRegistry()
        registry.register_section(make_collector_section("notes", collector))
        from xcode.agent.config import AgentContext
        from xcode.agent.request import DefaultRequestAssembler

        context = AgentContext(context_state=ContextState())
        request_assembler = DefaultRequestAssembler(context_collectors=registry)
        request_assembler.assemble(context, current_step=1, options=None)
        collector.value = "after"
        assembly = request_assembler.assemble(context, current_step=2, options=None)

        assert assembly.context_trace[0].block_id == "notes"
        assert 'status="updated"' in str(assembly.messages)

    def test_state_section_reports_removal(self) -> None:
        section = make_state_section(
            "mode",
            "mode",
            ContextBlockSource.MODE,
        )
        state = ContextState()
        input_with_mode = ContextCollectionInput(state={"mode": {"current": "act"}})
        assert state.world_state.render((section,), input_with_mode)
        removed = state.world_state.render((section,), ContextCollectionInput())

        assert removed[0].content == '<context-section id="mode" status="removed" />'


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
