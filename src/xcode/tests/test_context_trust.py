"""Context trust contract tests."""

from __future__ import annotations

from xcode.agent.context_assembly import (
    ContextAssemblyInput,
    ContextAuthority,
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextPriority,
    ContextProvenance,
    ContextScope,
    ContextTrust,
    DefaultContextAssembler,
)
from xcode.agent.messages import UserMessage


def test_workspace_instruction_defaults_to_workspace_policy() -> None:
    block = ContextBlock(
        source=ContextBlockSource.INSTRUCTION,
        content="Repository instructions",
        priority=ContextPriority.HIGH,
    )

    assert block.authority is ContextAuthority.WORKSPACE_POLICY
    assert block.trust is ContextTrust.WORKSPACE_UNTRUSTED
    assert block.scope is ContextScope.REPOSITORY
    assert block.provenance.origin == "instruction"


def test_memory_defaults_to_non_system_memory_authority() -> None:
    block = ContextBlock(
        source=ContextBlockSource.MEMORY,
        content="Prior validated result",
        priority=ContextPriority.MEDIUM,
    )

    assert block.target is ContextBlockTarget.USER_CONTEXT
    assert block.authority is ContextAuthority.MEMORY
    assert block.scope is ContextScope.SESSION


def test_explicit_context_contract_is_preserved() -> None:
    provenance = ContextProvenance(
        origin="memory_store",
        locator="memory:abc123",
        evidence_ids=("evidence:tool:1",),
        content_hash="sha256:example",
    )
    block = ContextBlock(
        source=ContextBlockSource.MEMORY,
        content="Validated procedure",
        priority=ContextPriority.MEDIUM,
        authority=ContextAuthority.MEMORY,
        trust=ContextTrust.VERIFIED_TOOL,
        scope=ContextScope.REPOSITORY,
        scope_key="repo:example",
        provenance=provenance,
    )

    assert block.provenance == provenance
    assert block.scope_key == "repo:example"
    assert block.trust is ContextTrust.VERIFIED_TOOL


def test_user_context_render_includes_typed_contract() -> None:
    block = ContextBlock(
        source=ContextBlockSource.MEMORY,
        content="Use the verified command.",
        priority=ContextPriority.MEDIUM,
        scope=ContextScope.REPOSITORY,
        scope_key="repo:example",
        provenance=ContextProvenance(locator="memory:abc123"),
    )
    result = DefaultContextAssembler().assemble(
        ContextAssemblyInput(
            messages=[UserMessage(content="continue")],
            context_blocks=[block],
        )
    )

    rendered = result.messages[0].content
    assert "authority=memory" in rendered
    assert "trust=runtime_internal" in rendered
    assert "scope=repository" in rendered
    assert "scope_key=repo:example" in rendered
    assert "locator=memory:abc123" in rendered
