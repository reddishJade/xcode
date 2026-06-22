# TODO

## Citation marker

根据 `docs/citation-marker-design.md` 实现本地文件/搜索结果的引用标记。

前提：ADR 已锁定以下设计决策——`CitationSource.text` 为内容快照（非引用），`grep_search` 最多 30 个 per-line source 后聚合，turn ID 在 decorator 阶段按 conversation 顺序分配，OpenAI marker 格式仅在 provider boundary 序列化。

步骤：
1. 添加 `CitationSource` 数据模型和 `citation_sources` metadata 常量
2. 扩展 `ToolOutput.metadata` 接受 `citation_sources` key；给 `AgentToolResult.details` 加 `citation_sources: list[CitationSource]` 字段
3. 将 `citation_sources` 贯穿 `ToolResultMessage.metadata`，走 `AGENT_CONTENT_BLOCKS_METADATA_KEY` 转发路径
4. 添加 citation 装饰器：按 prompt 顺序扫描 tool result，分配 turn ID 并生成 `\ue200cite\ue202turnNkindI\ue201` 标记头 + 行号文本
5. `read_file` 产出 `kind="file"` 的 citation source
6. `grep_search` 产出 `kind="search"` 的 citation source（最多 30 个 per-line source，超出后聚合为一个 range source）
7. 添加 stable prompt 指令（仅在 citable tools 启用时）
8. 添加 final assistant citation 解析/渲染支持

## Tool Governance

根据 `docs/tool-governance.md` 实现 ToolSpec 六关注点分离。

步骤：
1. 引入 `RegisteredTool`、`ToolSurfacePolicy`、`ToolOrigin`、`ToolActionProfile` 数据结构和 canonical internal id 方案。registry 存储 `RegisteredTool`，不改变行为。注册时校验 `public_selector` 唯一性
2. Registry filter API 和 selector resolver：分别回答 `/tool` 可见性、user-invocability、primary agent、subagent 的工具集合。Selector resolver 将 public selector 映射到 canonical id
3. `/tool` list、completion、help、direct execution 全部切换到 `RegisteredTool.surface_policy`，通过 selector resolver 原子访问。feature flag `use_registered_tool_governance` 同时门控 UI 层和执行网关
4. Primary agent Plan/Build/Act assembly 从手写 allowlist 切换到 `primary_agent_invocable` + mode 约束 + `ToolActionProfile` 存在性检查
5. Subagent delegation 从 `group != "mcp"` 切换到 `subagent_policy` 过滤 + `delegation_context_constraints`。subagent 独立授权
6. LLM provider-facing function-name adapter：从 canonical id 生成 provider 合法标识符
7. Permission model：移除 `capability == "mcp"` 和 `mcp__` 前缀匹配。capability 从 `ToolActionProfile.capability` 推导
8. 最终审计：清除所有 `group == "mcp"` 分支。通过完整 test matrix

## Context cost display

`/context` 命令已可以利用 registry 中 `Model.cost` 显示费用估算。

待后续扩展：
- 跟踪实际 output token 数（从 `AssistantMessage.usage` 提取）使 cost 更准确
- OpenAI long-context 分段计价（>272K input 触发 2x/1.5x）
