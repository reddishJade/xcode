# Citation Marker Support Design

## Status

Design only. This document defines the implementation boundary for adding local
file/search citations later; it does not enable citations by itself.

## Problem

`read_file` and `grep_search` return useful local evidence, but the model has no
stable source IDs to cite. If final answers include claims based on those tool
outputs, downstream rendering cannot distinguish supported statements from
uncited text.

OpenAI models are most reliable when citations use the familiar marker shape:

```text
\ue200cite\ue202<source_id>\ue202<locator>\ue201
```

The implementation should expose that marker to the model while keeping source
metadata structured enough for audit, rendering, and future non-OpenAI
providers.

## Existing Boundaries

- Product tools live in `src/xcode/coding_agent/tools/`.
- `read_file` currently returns text or `ToolOutput` metadata for images.
- `grep_search` currently returns plain ripgrep-style text lines.
- `ToolOutput.metadata` is available to `ToolSpecAdapter`, but only structured
  content blocks under `AGENT_CONTENT_BLOCKS_METADATA_KEY` are forwarded today.
- `ToolResultMessage` currently stores content, tool name, and tool call id, but
  no general metadata.
- Provider conversion flattens `ToolResultMessage.content` into text before it
  reaches OpenAI.

## Design Goals

- Give each citable local source a deterministic source ID in the
  `turn{n}{kind}{i}` family.
- Preserve raw source metadata separately from the model-visible marker text.
- Support line locators for text file evidence.
- Keep citation handling provider-neutral until the final provider conversion
  step.
- Avoid citing tool outputs that are not explicitly marked citable.

## Non-Goals

- Do not cite `write_file`, `edit_file`, `bash`, image payloads, or directory
  listings in the first implementation.
- Do not parse final assistant citations in this step.
- Do not add external web citations; this design only covers local file/search
  outputs.

## Source Model

Add a small citation model in harness runtime, not in provider-specific code:

```python
@dataclass(frozen=True)
class CitationSource:
    """模型可引用的本地证据来源。"""

    kind: Literal["file", "search"]
    path: str
    start_line: int
    end_line: int
    text: str
```

Use a metadata key such as `citation_sources` on `ToolOutput.metadata`.

`read_file` should emit one `CitationSource(kind="file", ...)` for the text
span returned to the model. `offset` and `limit` determine `start_line` and
`end_line`; a full read uses the rendered span after truncation.

`grep_search` should emit one `CitationSource(kind="search", ...)` per matched
or context result line. The source text is the displayed result line, and the
line locator comes from the parsed `path:line:` prefix.

## Source ID Assignment

Source IDs should be assigned after tool execution, when the full prompt history
is known:

```text
turn0file0
turn1search0
turn1search1
```

`turnN` increments once per citable tool result in prompt order. `file0` and
`search0` are item indexes within that tool result. This matches OpenAI's
recommended family without requiring product tools to know provider turn state.

The mapping should be generated in a harness citation decorator before provider
conversion:

```text
ToolOutput.metadata
  -> AgentToolResult.details
  -> ToolResultMessage.metadata
  -> provider-visible decorated tool result text
```

This requires extending `AgentToolResult` and `ToolResultMessage` metadata
plumbing before adding markers to any tool output.

## Model-Visible Rendering

Decorate citable tool results only at the provider boundary. The transcript and
audit logs can retain raw tool output plus structured citation metadata.

Each citable source should be rendered as:

```text
Citation Marker: \ue200cite\ue202turn0file0\ue201
Path: src/xcode/harness/config.py
Lines: L20-L32

[L20] PROFILE_MAIN = "main"
[L21] PROFILE_SUBAGENT = "subagent"
```

The model instruction should say:

```text
Use \ue200cite\ue202<source_id>\ue202Lx-Ly\ue201 when citing marked local file
or search evidence. Do not cite unmarked tool output.
```

Line locators are mandatory for `file` and `search` sources because local files
are mutable and users need precise verification points.

## Implementation Sequence

1. Add `CitationSource` and metadata constants in harness runtime.
2. Carry `ToolOutput.metadata["citation_sources"]` through
   `ToolSpecAdapter`, `AgentToolResult`, and `ToolResultMessage`.
3. Add a citation decorator that scans tool results in prompt order and renders
   marker headers plus line-numbered source text.
4. Update `read_file` to produce one citable file source for returned text.
5. Update `grep_search` to produce citable search sources for parsed result
   lines.
6. Add stable prompt instructions for citation syntax only when citable tools
   are enabled.
7. Add parser/rendering support for final assistant citation markers.

## Test Plan

- `read_file` with `offset` and `limit` emits one `file` source with the correct
  line range.
- Truncated `read_file` output cites only the visible line span.
- `grep_search` emits one `search` source per visible result line and ignores
  truncation notices.
- Provider conversion decorates citable tool results with `\ue200cite...`
  markers and leaves non-citable tool results unchanged.
- Multiple citable tool calls receive monotonically increasing `turnN` IDs in
  prompt order, including parallel tool batches.
- Existing Chat Completions and Responses flows receive the same decorated text.
