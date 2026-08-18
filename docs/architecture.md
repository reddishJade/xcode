# Xcode 架构

## 定位

Xcode 是本地运行的 Python coding-agent harness。它不是一组工具的薄包装，
而是负责模型输入、运行状态、工具权限、会话事实、生命周期和终端交互的
软件运行时。

当前架构可以概括为：

```text
agent
= model provider
+ append-only session ledger
+ local capability graph
+ execution and permission policy
+ lifecycle control
+ typed interaction surface
+ product composition
```

## 核心不变量

1. 模型可见即已记录。实际发给 provider 的 messages、tools 和 provider
   参数必须先形成 `provider_request` envelope，能够审计和比对。
2. session transcript 是事实账本。用户消息、稳定运行事件、压缩 epoch、
   子代理生命周期和最终回答只能追加，不能原地改写历史。
3. 内存状态是日志投影。resume、fork 和 restart 从 transcript surface
   重建，不把 CLI/TUI 对象当成事实来源。
4. 工具呈现属于协议。terminal、diff、location、subagent 等语义由工具产生
   类型化 intent，宿主只负责投影，不从输出字符串猜测。
5. 正确性以组装后的产品为准。局部单测不能替代真实 `build_app()`、真实
   registry、session 落盘和重建路径。
6. 迁移直接完成。当前预发布阶段不保留旧签名、双写、别名适配器或旧 schema
   分支；调用方、测试和文档在同一提交中一次迁移。
7. 一个 run 只使用一个 composition generation。provider、工具表、agent 配置、
   静态权限策略、请求组装器和上下文入口必须在 run 开始前一起发布，运行中
   不得从多个可变对象分别读取。

## 分层

| 层 | 路径 | 所有权 |
|---|---|---|
| Provider | `src/xcode/ai/` | provider 协议、流式事件和厂商适配 |
| Agent | `src/xcode/agent/` | 消息模型、loop、工具执行和 provider 请求 |
| Harness | `src/xcode/harness/` | session、权限、观测、MCP、记忆和运行策略 |
| Coding product | `src/xcode/coding_agent/` | coding 工具、产品 registry 和应用装配 |
| Host | `src/xcode/cli/` | REPL/TUI 输入输出与交互控制 |

依赖应朝更低层稳定协议流动。CLI/TUI 不拥有 session 语义；工具不直接拥有
provider；provider 不感知产品工具。

## 一次回合的数据流

```text
user input
  -> SessionInbox.inbox/inserted
  -> active run claims next_step / next_turn
  -> SessionInbox.inbox/claimed
  -> CodingAgentHarness / Agent loop
  -> RequestAssembly
       - scoped prefix + session surface
       - context collection and injection
       - request hygiene
       - wire messages + tool schemas + options
  -> provider_request envelope
  -> provider stream
  -> typed assistant/tool events
  -> local Shell or local FileSystem
  -> permission and audit hooks
  -> append-only session events
  -> REPL/TUI projection
```

`SessionInbox` 是所有模型输入的统一所有者，`SessionRecorder` 记录运行输出。
编程式 `ask()`、REPL 和 TUI 必须经过同一条路径。`harness/session/replay.py`
负责从当前 branch 恢复 message history、run metadata、Goal 和 contextual
state；未 claim 的输入由 inbox 自身恢复。

普通 agent 请求只有一个 `RequestAssembler` 入口。provider stream 与审计 hook
消费同一个 `RequestAssembly`，其中包含最终 wire messages、tool schemas、options、
step 和动态 context provenance。禁止在发送前通过通用 transformer 隐式改写
messages；请求卫生是 assembly 内的显式确定性阶段，且不修改 session surface。

## Agent composition

`AgentComposition` 是发布 agent 行为的不可变 generation，包含主/备 provider、
冻结工具 schema、`AgentConfig`、静态 gate 策略、`RequestAssembler` 和 runtime
context 入口。`AgentRuntimeConfig` 只保存 session inbox、取消、压缩器、hook、
审计和 grant store 等有生命周期的服务；这些对象不伪装成产品配置。

每个 run 在取得 active-run 所有权后原子捕获一次 composition 与有效 provider，
后续 step 不再重新读取产品装配。`/model` 和静态 permission policy 变更必须发布
新的 generation；active run 存在时拒绝替换。旧的 provider setter、fallback
包装器原地换主和私有 gate 字段写入均不存在。`provider_request` 保存
`composition_id`，因此一次实际请求可以回溯到完整装配代际。

## Session 事实模型

稳定记录包括：

- `inbox/inserted`、`inbox/claimed`、`inbox/discarded`：输入内容、lane、来源和
  消费生命周期；
- `assistant`：最终用户可见回答；
- `provider_request`：provider 实际收到的输入和请求指纹；
- `assistant`、`tool_use`、`tool_result`、`final`：运行语义；
- `compaction`：追加式压缩 epoch，原 transcript 保持不变；
- `subagent_run`：子运行的 started/completed/failed/cancelled 生命周期。

`compaction` 保存完整、类型化的 surface replacement、来源 entry IDs、generation
和指纹。replayer 只按日志顺序应用 replacement，不读取第二份 checkpoint 状态。
只有 `inbox/claimed` 中的 typed message 会进入模型 surface；普通命令记录为
`command` event，不会伪装成用户消息。

## 本地执行边界

Xcode 只支持本地执行，不提供容器、远程 workspace 或远程 shell 抽象。
`bash` 直接依赖 `Shell`，文件工具直接依赖 `FileSystem`；生产实现分别是
`SubprocessShell` 和 `LocalFileSystem`。这些窄协议用于测试本地行为，不代表
可切换的远程执行世界。

Xcode 的权限引擎、审批和 shell 效果分析不是 OS sandbox。需要进程级隔离时，
应由运行 Xcode 的外部环境提供，Xcode 自身不负责创建或管理该环境。

## 工具呈现

`ToolRenderIntent` 当前包含：

- `terminal`：命令与本地工作目录；
- `diff`：patch、文件集合和首个变更行；
- `location`：文件或目录及行范围；
- `subagent`：batch ID 与 child run IDs。

intent 随 `ToolResultMessage` 进入 runtime event 和 session log。新增呈现类型时，
必须同时更新事件编码、回放解码、CLI/TUI 投影和契约测试。

## 子代理

子代理的身份是独立 durable session，不是父工具调用中的临时 `Agent` 对象。每个
child 拥有自己的 session log、surface、inbox、composition generation 和 provider
request envelope；`subagent/descriptor` 记录 child/parent session ID、one-shot 或
continuable 模式、persona、provider model 和初始 composition ID。session index 的
`parent_id` 提供无需激活 child 的 lineage 枚举。

`subagent` 创建新 child。并行 batch 只允许 one-shot；需要后续对话时显式创建
continuable child，再用 `subagent_continue` 按 child session ID 提交 FIFO turn。
进程中没有 activation 时，manager 从 child log 重建 surface 后冷恢复同一个
session。`subagent_list` 只读取 descriptor，不启动模型。Xcode 当前只实现 spawn，
不会复制父 transcript；任务 prompt 必须自包含。

每次 child turn 仍创建 batch/run ID，父 session 的 `subagent_run` 事件记录 child
session ID、模式、状态、摘要或错误；父 tool result 的 subagent intent 保存 run
关联。child 模型失败被解析为 completed/failed/cancelled 结果，基础设施不会通过
共享 `messages[]` 假装成普通父对话。

## 组合根

`build_app()` 是产品组合根，按顺序构造：

1. 已解析配置；
2. 共享同一 store 的 session recorder/inbox、memory、compactor 和 cancellation；
3. provider bundle；
4. 本地工具、MCP、memory/history 和 subagent registry；
5. 冻结的 `AgentComposition`、会话级 runtime services 和
   `CodingAgentHarness`；
6. `XcodeApp` 生命周期句柄。

任何新能力都应进入拥有该行为的层，并在真实组合测试中证明最小应用仍可
启动、请求、落盘和恢复。

## 明确非目标

- 容器执行、远程 shell、远程 filesystem 或远程 workspace；
- 为未发布调用面保留兼容包装、双写或旧格式解析；
- CLI/TUI 各自维护一套运行或 session 语义；
- 未写入 session event 的隐式模型上下文；
- 仅靠单元覆盖率声明真实产品路径正确。
