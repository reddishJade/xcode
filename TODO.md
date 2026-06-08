# Xcode TODO

按优先级排序，仅保留未完成计划。

## 边界

- 默认路径：REPL/CLI → StructuredAgent → core tools → permission/risk/audit
- 默认工具组：`tools.enabled_groups=["core"]`
- 新能力以 opt-in group 或 `experimental.*` 进入
- 不做默认 MCP 全量注入，不做不可观测 swarms，不做绕过权限的外部工具直连，不做企业级 RBAC/Grafana/Phoenix/RAGAS

---

## OpenAI Responses API parity

已完成：

- [x] 将 `StreamOptions.cache_retention` 映射到 OpenAI `prompt_cache_retention`。

计划：

- [ ] 添加 `FileContent` 内容块类型，并补全 agent 消息到 Responses file input 的类型链路。
- [ ] 添加 `/responses/input_tokens` token counting API 集成，避免只依赖本地估算。
- [ ] 添加服务端压缩 `context_management` 集成，保留现有本地压缩作为明确路径。
- [ ] 添加 background mode 的结果获取能力，支持后台响应轮询或恢复读取。
- [ ] 将 `tool_catalog_fingerprint` 动态接入 `prompt_cache_key`，减少工具目录变化导致的缓存错配。

## OpenAI native shell / skills alignment

- Design the minimal native OpenAI shell path using `ToolDefinition.builtin` with `{"type": "shell", "environment": {"type": "local"}}`.
- Add an agent/tool bridge for the Responses builtin shell without removing the existing `bash` function tool.
- Execute Responses `shell_call` requests locally and return official `shell_call_output` items with `stdout`, `stderr`, and `outcome`.
- Attach `SkillLoader.to_local_shell_skills()` metadata to the native local shell environment when skills are enabled.
- Add targeted tests for builtin shell tool definitions, local shell output round trips, and skill metadata attachment.
- Validate each implementation step with targeted `ruff`, `mypy`, `pyright`, and `unittest` commands.
- Commit each logical step separately with exact-path staging only.

## OpenAI prompting and citation alignment

- [ ] Step 1: Map internal `system` instructions to OpenAI Responses `developer` role while preserving Chat Completions behavior.
- [ ] Step 2: Move stable Responses API instructions into the `instructions` request parameter instead of mixing them into `input`.
- [ ] Step 3: Adjust `previous_response_id` incremental input filtering so system/developer instructions are not duplicated across turns.
- [ ] Step 4: Reorder prompt modules toward `Identity -> Instructions -> Tools -> Context` while keeping stable prompt content first for cache hits.
- [ ] Step 5: Add a Markdown `# Identity` section heading to the core identity prompt.
- [ ] Step 6: Replace prompt stable-cache mtime keys with content hashes for `AGENTS.md` and `CLAUDE.md`.
- [ ] Step 7: Design citation marker support for file/search outputs using OpenAI's `\ue200cite\ue202...\ue201` format before implementation.
- [ ] Validate and commit each logical step separately with exact-path staging only.

## OpenAI reasoning API alignment

已完成：

- [x] Step 1: 从 `THINKING_LEVELS` 中移除无效的 `"max"` 值（OpenAI 不支持该值）。

计划：

- [ ] Step 2: 为 `ThinkingBudgets` 数据类添加 `xhigh` 字段（与 `THINKING_LEVELS` 保持一致）。
- [ ] Step 3: 在 `StreamOptions` 和 `OpenAIResponsesProvider` 中添加 `reasoning.summary` 支持。
- [ ] Step 4: 扩展 `_reasoning_output_items` 保留 `function_call` 类型输出项（无状态回灌完整性）。
- [ ] Step 5: 在 `AssistantMessage`、`to_responses_input` 中添加 `phase` 字段透传。
- [ ] Step 6: 在 Responses API 中将 system 消息转为 developer role（推理模型推荐做法）。
- [ ] Step 7: 在 `responses_stream_to_events` 中添加 `incomplete` 状态检测（max_output_tokens 耗尽检测）。
