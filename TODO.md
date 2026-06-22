# TODO

## Tool Governance — Remaining Migration

已实现：`RegisteredTool`、`ToolSurfacePolicy`、`ToolOrigin`、`ToolActionProfile`、`ToolSelector` 数据结构和 registry filter API（`tools_visible_to_root`、`tools_primary_agent_invocable`、`tools_for_subagent`、`resolve_selector`）。Mode governance filter（`governance_registry_for_mode`）已接入。

剩余步骤：
1. `/tool` list、completion、help、direct execution 全部切换到 `RegisteredTool.surface_policy`，通过 selector resolver 原子访问。feature flag `use_registered_tool_governance` 同时门控 UI 层和执行网关
2. Primary agent Plan/Build/Act assembly 完全从旧 `PlanPolicy`/`BuildPolicy` 切换到 `primary_agent_invocable` + mode 约束 + `ToolActionProfile` 存在性检查
3. Subagent delegation 从 `group != "mcp"` 切换到 `subagent_policy` 过滤 + `delegation_context_constraints`。subagent 独立授权
4. LLM provider-facing function-name adapter：从 canonical id 生成 provider 合法标识符
5. Permission model：移除 `capability == "mcp"` 和 `mcp__` 前缀匹配。capability 从 `ToolActionProfile.capability` 推导
6. 最终审计：清除所有 `group == "mcp"` 分支。通过完整 test matrix

## Context cost display

`/context` 命令已可以利用 registry 中 `Model.cost` 显示费用估算。

待后续扩展：
- 跟踪实际 output token 数（从 `AssistantMessage.usage` 提取）使 cost 更准确
- OpenAI long-context 分段计价（>272K input 触发 2x/1.5x）
