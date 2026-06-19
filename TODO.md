# Xcode 已知问题与待办

基准复核日期：2026-06-18。每项“现状”均已按当前源码重新核对；已修复或已不
成立的描述不保留在本文件中。

优先级定义：

- P0：安全边界或可能导致错误授权。
- P1：核心能力不可用、状态不一致或协议行为错误。
- P2：资源稳定性、正式能力完善和可观测性。
- P3：维护性、质量治理和低风险一致性。

同一优先级内按依赖顺序排列。Skill 和 MCP 是核心能力；Memory 是正式但可选的
能力。现有 Python Plugin 系统不作为产品能力保留。

## P1 · LLM-as-judge eval 未实际生效

`src/xcode/evals/graders.py:run_llm_judge()` 从 `app.agent.provider` 获取 judge
provider，但 `ModelProvider` 协议仅保证 `stream()`，而 judge 只检查 `ask()`
和 `run()`，导致普通 provider 路径返回空 tuple。内置 eval 套件的
`llm_judge_criteria` 不参与评分。

需要：

- 为 judge 定义明确协议并单独注入 provider，或基于 `ModelProvider.stream()`
  构建 judge 调用。
- judge 未执行时在 report 中显式记录 skipped，而不是静默返回空结果。
- 添加真实 provider protocol 的离线替身测试。

## P1 · PROVIDER_REGISTRY 不完整

`ProviderTransport` 和 factory 分支接受 `anthropic_messages`，但
`PROVIDER_REGISTRY` 没有对应 provider。合法配置会在运行时失败。

需要：

- 若 Anthropic provider 已确定进入当前版本，完成实现和注册。
- 否则删除 transport literal、API key 映射和 factory 特殊分支。
- 添加“所有声明 transport 均可解析”的注册表一致性测试。

## P2 · MCP tools/list 不支持分页和动态刷新

`McpClient.list_tools()` 只发送一次空参数请求，忽略 `nextCursor`。
`notifications/tools/list_changed` 也未处理。

需要：

- 循环处理 `tools/list` cursor pagination。
- server 声明 `tools.listChanged` 时处理 list-changed notification。
- 刷新 schema cache 和 runtime tool registry。
- 对重复 cursor 和异常分页设置保护。

## P2 · MCP timeout 未发送取消通知，关闭流程不符合完整生命周期

请求超时后客户端直接停止等待，没有发送 MCP cancellation notification。
`stop()` 直接关闭所有 stream 并 terminate process，没有先关闭 stdin 并等待
server 自行退出。

需要：

- request timeout 后发送 `notifications/cancelled`。
- graceful shutdown：关闭 stdin、等待 server、再 TERM/KILL。
- `LazyClientRef` 对失败连接进行有限重连并保留 last error。
- 为 timeout、cancel、server 自行退出和强制 kill 添加测试。

## P2 · MCP 现代 tool result 支持不完整

当前 handler 主要拼接 text content。image、audio、resource link 和 embedded
resource 只产生 placeholder；`structuredContent`、`outputSchema` 和 annotations
没有进入宿主结果模型。

需要：

- 支持 `structuredContent`。
- 保留 `outputSchema`，在可行时验证 structured result。
- 为非文本 content 定义结构化宿主映射；暂不支持的类型必须返回完整、可诊断
  的 unsupported result。
- 不因第一个未知 block 丢失后续 content。

## P2 · Skill 缺少用户显式激活入口

模型自动选择之外，用户应能直接激活 skill。

需要：

- 支持 `$skill-name`、`/skill skill-name` 或等价语法。
- 在 REPL 中提供 skill name 补全。
- 显式激活通过同一 activation 状态和 compaction 保护路径。
- 未知或被禁用 skill 返回明确错误。

## P2 · Skill frontmatter 与 Agent Skills 规范不完整

当前解析 name、description、hidden，但未完整校验名称格式、长度、连续 hyphen
和目录名一致性，也未保留 `license`、`compatibility`、`metadata`、
`allowed-tools`。

需要：

- 对 cosmetic 问题采用 warn + load。
- 缺少 description 或 YAML 完全无法解析时 skip。
- 保留 compatibility、license 和 metadata。
- activation 时向模型提供 compatibility。
- 暂不让 `allowed-tools` 绕过 PermissionEngine；该字段只作为提示或忽略。

## P2 · 缺少用户可配置的显式 Hook 系统

Xcode 已有内部 `HookManager` 和固定事件，但 `build_app()`、运行时配置与 REPL
均没有用户级 hook 注册、禁用、查看或诊断入口。原有 Python Plugin 外部注入
路径已经删除。

需要：

- 在配置中提供显式 hooks 列表，至少支持
  `pre_tool`、`post_tool`、`on_error`、`on_compact`、
  `before_agent_start` 和 `before_provider_request`。
- 每项声明 event、可选 matcher、command、timeout、enabled 和 failure policy。
- Hook 作为受信任的外部子进程执行，通过 JSON stdin/stdout 交换结构化数据；
  不使用 `exec()`、动态 import 或隐式 shell。
- pre hook 只允许返回结构化 allow/deny/ask 或参数变换；任何权限放宽仍必须经过
  PermissionEngine，不能覆盖 non-bypassable deny。
- 支持 `/hooks` 或等价诊断入口，显示来源、启用状态和最近错误。
- 主 agent 与 subagent 的继承规则必须显式配置，默认不向 subagent 传播外部
  command hook。
- 增加超时、非零退出、无效 JSON、敏感字段脱敏和配置合并测试。

不提供任意进程内 Python callback 配置。库调用方仍可通过明确的编程接口注入
`HookManager`，但配置文件能力必须保持进程隔离和权限边界。

## P2 · 缺少轻量 TodoWrite 会话工具

现有 `tasks` 和 `progress` 是持久化任务图、依赖和长任务租约系统，不适合作为
一次编码会话中的轻量执行清单。当前没有模型可直接维护并向用户展示的
TodoWrite 等价能力。

需要：

- 增加单一 `update_todo` 工具，以完整列表替换当前会话清单。
- item 至少包含稳定 id、content 和
  `pending` / `in_progress` / `completed` 状态。
- 强制最多一个 `in_progress`，拒绝空内容、重复 id 和无效状态。
- 清单进入 session state、resume 和 compaction 保护，并通过结构化事件在 REPL
  中渲染。
- 默认仅主 agent 可用；subagent 默认排除，但允许通过明确 allowlist 手动启用。
- 不复用 TaskStore 的依赖图、租约和跨任务状态机，避免把轻量清单耦合到长期任务
  编排。

## P2 · tool_workers 未限制工具并发

`AgentConfig.tool_workers` 可配置并被传入 runtime，但工具执行路径没有消费该
值。parallel batch 的并发数量由模型一次返回的 tool calls 决定。

需要：

- 使用有界 semaphore 或 worker pool 限制 parallel tool call。
- 保持 sequential 工具的顺序语义。
- 将取消信号和异常收集行为纳入并发限制测试。
- 首版只限制总 parallel tool calls，不设计资源标签或依赖图。

## P2 · subagent 缺少独立并发上限

`ManagedSubagentRunner` 可以持续提交独立 worker job，没有明确的 active job
额度。它与普通工具并发属于不同资源，应单独治理。

需要：

- 增加可配置的最大 active subagent 数。
- 超限时返回明确 busy 状态或排队，不无限创建任务。
- shutdown、cancel、timeout 和 finished job 清理必须释放额度。

## P2 · observability 缺少基础关联字段

Audit record 已有时间戳，但 HookRecord、provider/tool event 和 final result
尚无统一的 session、turn、request/tool-call 关联信息。

需要：

- 为 hook/event 增加 UTC timestamp。
- 统一传递 session_id、turn_id、request_id 和 tool_call_id。
- final result 提供本轮模型与工具耗时汇总。
- 暂不引入 OpenTelemetry；先保证 JSONL audit、hook subscriber 和 eval trace
  可以关联同一轮执行。

## P3 · tool_catalog.py 新增 builder 无代码强制

`src/xcode/cli/tool_catalog.py` 的 docstring 要求 `build_*_tools()` 必须注册入
`_builders()`，但无类型系统或测试约束。新增 builder 后工具目录会遗漏。

需要：

- 优先添加 registry/catalog 一致性测试。
- 只有出现多个实际调用方时再考虑引入通用 builder registry。

## P3 · slash command 与 @file 仅支持前缀补全

`ReplCompleter` 对 slash command、tool name 和当前目录层级的 `@file` 使用
`startswith()`。`@file` 不做跨目录候选检索，输入 `@` 本身也不返回候选；
command dispatch 则只接受精确命令。

需要：

- slash command 和 `/tool` 补全增加轻量 fuzzy ranking，但提交执行仍要求唯一、
  精确命令。
- `@file` 使用项目文件索引进行 basename、路径片段和子序列匹配。
- 文件候选遵循与 glob 相同的 `.gitignore`、hidden 和 blocked path 规则。
- 精确前缀结果优先于模糊结果，并限制候选数量和扫描时间。
- 不缓存失效的完整项目树；使用短生命周期缓存或文件索引版本。
- 添加重名文件、深层路径、Windows 分隔符、ignored file 和 typo command 测试。

## 明确不进入近期范围

- 通过 JSON/YAML 配置直接定义任意可执行 Tool；外部工具扩展统一使用 MCP。
- 模糊匹配后直接执行 slash command；模糊能力只用于候选提示，执行必须精确。
- 配置文件加载进程内 Python Hook callback。
- Skill marketplace。
- Harness 侧向量语义匹配。
- `enabled.txt` 专用激活格式。
- Skill 独立脚本执行 runtime。
- MCP Streamable HTTP、OAuth、resources、prompts、sampling、roots 和
  elicitation；出现明确 server 用例后再进入 TODO。
- MCP 旧 HTTP+SSE 和私有 WebSocket transport。
- 通用工具资源标签、依赖图和读写锁调度器。
- Subagent 递归和 child-to-child 通信。
- 完整全系统错误 taxonomy。
- 仅为架构完整性引入 OpenTelemetry。
