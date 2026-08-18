"""Provider RequestAssembly 单一组装边界测试。"""

from __future__ import annotations

from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.context import (
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextCollectionInput,
    ContextCollectorRegistry,
    ContextExpiry,
    ContextPriority,
)
from xcode.agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.request import DefaultRequestAssembler, RequestHygiene
from xcode.agent.types import ToolCallContent, ToolSpec, ToolSpecAdapter


class _Collector:
    def collect(self, _input: ContextCollectionInput) -> list[ContextBlock]:
        return [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.HIGH,
                target=ContextBlockTarget.SYSTEM,
                content="current architecture note",
                block_id="note-current",
            ),
            ContextBlock(
                source=ContextBlockSource.ACTIVE_DIFF,
                priority=ContextPriority.LOW,
                content="expired diff",
                block_id="diff-old",
                expiry=ContextExpiry(max_steps=1),
                created_step=0,
            ),
        ]


def test_request_assembly_is_the_complete_provider_envelope() -> None:
    collectors = ContextCollectorRegistry()
    collectors.register(_Collector())
    tool = ToolSpecAdapter(
        ToolSpec(
            name="read_file",
            description="Read one local file.",
            input_hint="path",
            handler=lambda _data, _update=None: "contents",
            schema={"type": "object", "properties": {}},
        )
    )
    assembler = DefaultRequestAssembler(
        context_collectors=collectors,
        hygiene=RequestHygiene(enabled=False),
    )

    assembly = assembler.assemble(
        AgentContext(
            request_prefix=[SystemMessage(content="identity")],
            messages=[UserMessage(content="inspect")],
            tools=[tool],
        ),
        current_step=2,
        options=None,
    )

    assert [message["role"] for message in assembly.wire_messages] == [
        "system",
        "system",
        "user",
    ]
    assert assembly.wire_messages[1]["content"] == "current architecture note"
    assert [tool.name for tool in assembly.tools] == ["read_file"]
    assert [(trace.block_id, trace.included) for trace in assembly.context_trace] == [
        ("note-current", True),
        ("diff-old", False),
    ]
    assert all(len(trace.content_sha256) == 64 for trace in assembly.context_trace)


def test_request_hygiene_changes_assembly_not_session_surface() -> None:
    long_output = "ordinary output line\n" * 20
    surface = [
        AssistantMessage(
            content=[
                ToolCallContent(
                    id="call-1",
                    name="bash",
                    arguments={"command": "y" * 100},
                )
            ]
        ),
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="bash",
            content=long_output,
        ),
    ]
    assembler = DefaultRequestAssembler(
        hygiene=RequestHygiene(
            max_tool_result_bytes=20,
            max_tool_arg_length=10,
            keep_head_lines=1,
            keep_tail_lines=1,
        )
    )

    assembly = assembler.assemble(
        AgentContext(messages=surface),
        current_step=1,
        options=None,
    )

    request_call = assembly.messages[0]
    assert isinstance(request_call, AssistantMessage)
    block = request_call.content[0]
    assert isinstance(block, ToolCallContent)
    assert block.arguments == {"command": "<truncated, 100 chars>"}
    request_result = assembly.messages[1]
    assert isinstance(request_result, ToolResultMessage)
    assert "lines omitted" in str(request_result.content)
    assert surface[0].content[0].arguments == {"command": "y" * 100}
    assert surface[1].content == long_output


def test_agent_loop_config_has_one_request_assembly_entrypoint() -> None:
    config = AgentLoopConfig()

    assert isinstance(config.request_assembler, DefaultRequestAssembler)
    assert "transform_context" not in type(config).model_fields
    assert "convert_to_llm" not in type(config).model_fields
