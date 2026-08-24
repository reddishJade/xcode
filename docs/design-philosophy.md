# Xcode 设计理念与核心实现

Xcode 并不是一组大模型 API 与本地脚本的简单组合（薄包装），而是一个面向生产环境的 **Python Coding Agent 运行时骨架（Runtime Harness）**。

在构建 Coding Agent 时，最常见的痛点往往不是模型的能力上限，而是运行时的不可控性：状态黑盒无法复现、文件被破坏后难以回滚、长上下文导致幻觉与 Token 浪费、命令缺乏隔离存在安全隐患，以及多工具调用时的串行低效。

Xcode 从第一天起就围绕 **可测试性（Testability）**、**可观测性（Observability）** 与 **纵深防御（Defense-in-depth）** 进行架构设计。

---

## 1. 核心设计理念

### 1.1 模型可见即已记录：可回放事实账本 (Append-only Ledger)
* **核心痛点**：传统 Agent 往往将内存对象（如 Python dict 或临时 List）作为状态源，崩溃或重启后上下文丢失，无法确定模型某一轮究竟看到了什么输入。
* **Xcode 方案**：
  * **事实账本（Session Transcript）**：所有用户输入、Provider 请求 Envelopes、工具执行语义（`tool_use` / `tool_result`）、压缩代际（`compaction epoch`）与子代理生命周期，均以 Append-only 的结构化 JSONL 事件流形式落盘。
  * **状态皆为日志投影**：会话的恢复（`resume`）、分叉（`fork`）与重播（`replay`）完全基于磁盘日志重放，内存对象不拥有状态所有权。
  * **确定性请求组装（RequestAssembly）**：发给 Provider 的最终 Wire Messages、Tool Schemas 和 Options 在发送前固化为 `provider_request` 事件，保证每一次模型调用的输入可比对、可审计。

### 1.2 执行模式与分权管理 (Plan / Build / Act)
* **核心痛点**：全自动执行容易造成不可逆破坏，而每一步都弹窗询问用户又严重打断心流。
* **Xcode 方案**：
  * **Plan 模式（只读规划）**：仅开放只读工具及 `.xcode/plans/*.md` 规划文件写入权限，禁止对业务代码的写操作与 Shell 执行，确保在深入调研与架构设计阶段绝不产生副作用。
  * **Build 模式（自动构建）**：允许项目内的结构化文件读写；对 Shell 命令和敏感边界动作，交由独立的 `reviewer` 模型（Reviewer Profile）进行自动化语义审批与风险评估，低风险操作秒级放行，无需打扰用户。
  * **Act 模式（人机协作）**：对所有写操作与 Shell 执行严格请求人工确认（HITL，Human-in-the-Loop）。
  * **findLast 规则覆盖引擎**：支持按工具名、主命令、子命令、参数 Flag 及路径通配符配置精细规则，最后一条匹配的规则生效，用户自定义规则优先于内置规则。

### 1.3 纵深防御安全体系 (Defense-in-depth)
* **核心痛点**：仅靠大模型自觉或字符串匹配无法提供可靠的安全隔离。
* **Xcode 方案**：
  * **决策层（PermissionEngine & Reviewer）**：统一拦截工具调用，进行路径边界检查、敏感文件匹配与基于上下文的风险判定。
  * **隔离层（Linux Bubblewrap Sandbox）**：Linux 环境下 Agent 的 `bash` 命令默认在 bubblewrap 命名空间沙箱中运行：
    * 宿主系统根目录为只读；
    * 仅项目根目录与 `/tmp` 挂载为可写；
    * `.git`、`.xcode`、`.agents` 自动重新挂载为只读，阻止篡改版本控制与运行时数据；
    * 密钥目录、私钥文件与 `.env*` 敏感配置全面遮蔽（Masking）；
    * 默认启用网络命名空间隔离（`network_access: deny`）。

### 1.4 副作用感知与工具并发分区 (Tool Concurrency Partitioning)
* **核心痛点**：Agent 发起多个工具调用时，强制串行执行导致耗时翻倍；盲目并发执行则会导致写冲突与竞态条件。
* **Xcode 方案**：
  * **读写分类调度**：调度器自动将并发安全的只读工具（`read_file`、`glob_files`、`grep_search`、`search_memory` 等）放入并发线程池并行调度，显著降低多文件阅读延迟；
  * **写操作串行屏障**：涉及文件写入或 Shell 命令的工具保持严格串行执行，确保前后步骤的状态确定性；
  * **Read-before-edit 指纹校验**：`edit_file` 必须基于读取时的 SHA256 指纹进行校验，若文件在编辑前被外部或并行操作修改，则拒绝写入，防止脏写。

### 1.5 分层上下文压缩与滚动 Checkpoint (Layered Compaction)
* **核心痛点**：多轮长程对话中，陈旧工具输出与历史文件内容会迅速耗尽上下文窗口，导致费用激增甚至超出模型窗口。
* **Xcode 方案**：
  * **70% 智能水位线触发**：在达到模型上下文窗口预留上限前（默认约 70% 使用率），自动触发微压缩与分层裁剪；
  * **分层裁剪（Layered Compactor）**：优先裁剪过期无效的 `read_file` 内容、大型工具输出截断以及陈旧工具结果；
  * **滚动 Checkpoint**：生成结构化的上下文摘要与状态 Replacement，按 Session 写入快照。会话恢复时通过 `Checkpoint + 原文 Tail` 极速重建上下文，既保留关键任务状态，又彻底解决 Token 膨胀问题。

### 1.6 长期记忆与子代理谱系 (Memory & Subagents)
* **长期事实源**：项目根目录 `MEMORY.md` 记录持久化架构决策与约定，用户级 `~/.xcode/memory/` 记录个人偏好。Agent 通过轻量 BM25 检索工具（`search_memory`）按需读取，不在每轮强行注入冗余背景。
* **独立子代理（Durable Subagent）**：子代理并非父会话中的临时对象，而是拥有独立 Session ID、事实账本与上下文生命周期的实体，避免子任务输出污染主会话上下文。
* **Child-first 逆序生命周期回收**：退出或中断时严格逆序等待子代理平稳收敛（Settle），防止孤儿进程与账本损坏。

### 1.7 中途可打断性与取消安全 (Mid-run Interruptibility & Cancellation Containment)
* **核心痛点**：Agent 在长篇大论或执行漫长脚本时，用户按下中断键往往导致事件循环挂起、HTTP 异步流泄露或临时文件残留。
* **Xcode 方案**：
  * **异步流中止**：收到中断信号时，底层的 Provider HTTP 流式连接瞬间被主动 Cancel，停止后续 Token 产生与计费；
  * **工具进程优雅终止**：正在运行的后台子进程（包括沙箱 Shell 和工具线程池）收到终止信号并在毫秒级安全退出；
  * **状态幂等性保证**：中断事件作为 `run_aborted` 记入账本，未完成的半途修改不会破坏已有的上下文一致性。

### 1.8 类型化渲染意图 (Typed Render Intents)
* **核心痛点**：传统 Agent 将工具输出粗暴转换为单一字符串输出，TUI/CLI 无法区分代码变更、终端输出还是进度状态。
* **Xcode 方案**：
  * 工具返回结果携带结构化的 `RenderIntent`（如 `DiffIntent`、`TerminalIntent`、`TodoIntent`、`SubagentIntent` 等）；
  * 展示层根据渲染意图在 TUI 中分屏呈现差异对比高亮、折叠终端视窗或动态进度条，将计算、I/O 与视觉呈现完全解耦。

---

## 2. 运行时分层架构

```
┌──────────────────────────────────────────────────────────┐
│ CLI / TUI (src/xcode/cli/)                               │  交互层：REPL、全屏终端、Slash 命令、配置浏览器
├──────────────────────────────────────────────────────────┤
│ Coding Agent (src/xcode/coding_agent/)                   │  产品装配：文件工具、代码编辑、Bash、Subagent、TODO
├──────────────────────────────────────────────────────────┤
│ Harness (src/xcode/harness/)                             │  运行时：Session 账本、权限引擎、沙箱、记忆、MCP、Hook
├──────────────────────────────────────────────────────────┤
│ Agent Loop (src/xcode/agent/)                            │  合约层：事件类型、上下文压缩、工具并发调度、看门狗
├──────────────────────────────────────────────────────────┤
│ AI Providers (src/xcode/ai/)                             │  适配层：OpenAI 协议基类、DeepSeek/MiMo/ChatGLM 适配
└──────────────────────────────────────────────────────────┘
```

数据流向始终由高层向底层稳定协议单向流动，保证了每个核心层均可独立编写单元测试与契约测试。

---

## 3. 系统核心不变量守卫 (Invariants & Postmortem Guardrails)

Xcode 在工程实践中沉淀了 5 条绝对不可违背的系统级不变量（System Invariants）：

1. **不可丢弃历史**：任何会话状态均可从 JSONL 事实账本完全重建，禁止任何原地修改或隐式状态丢失；
2. **不可越界执行**：非只读操作与跨工作区写入必须经过权限引擎与沙箱边界校验，禁止任何绕过审批的副作用；
3. **可见即已记录**：大模型接收到的每一条 Wire Message 必须与账本中落盘的 `provider_request` 记录完全一致；
4. **编辑幂等防脏写**：所有文件修改必须基于读取时的 SHA256 指纹校验，杜绝竞态脏覆盖；
5. **经验量化验证**：所有关键架构决策（如压缩算法、工具调度优化）必须拥有对应的真实消融 Benchmark 进行可量化追踪。

