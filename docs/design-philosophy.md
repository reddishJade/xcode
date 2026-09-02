# Xcode 设计与实现的哲学与理念

Xcode 是一个运行在本地工作区中的编码 Agent harness。用户给出目标，Agent 通过模型、工具、权限、会话、上下文和验证完成一轮或多轮工作。系统把模型输出转换为可执行的工具调用，把工具结果转换为下一次推理的证据，把运行事实转换为可恢复的会话账本。

Xcode 的核心判断可以概括为：

> **让模型负责推理，让运行时负责边界，让会话负责连续性，让证据负责结论。**

这四句话贯穿循环、装配、权限、文件操作、持久化和交互界面。

---

## 1. 核心设计理念

### 1.1 能力与所有权同步清晰

Xcode 把每项能力放入明确的拥有者：

1. Agent 核心循环拥有模型调用、工具调度、重试、续写和终止编排。
2. AI 层拥有 provider 协议、服务适配、模型元数据、费用和流事件。
3. 通用 harness 拥有运行生命周期、取消、输入调度、结果翻译和运行时组合。
4. `coding_agent` 拥有编码工具、执行模式、技能激活、记忆、todo 和目标验收。
5. session 层拥有事实记录、分支、surface、inbox、恢复和文件快照。
6. security 层拥有 Action 提取、权限裁决、授权、审批和边界策略。
7. CLI、TUI 和 Web 拥有输入、审批交互、事件展示和视口管理。

能力可以丰富，拥有关系保持清晰；抽象可以存在，抽象对应真实的替换点、边界或生命周期；运行路径保持单一而可追踪。

### 1.2 契约先行

关键接缝使用协议、值对象、显式序列化、类型收窄、事件数据类和模块边界：

- `ModelProvider`、`StreamProvider` 定义模型调用边界。
- `AgentTool`、`ToolSpec` 定义工具执行与工具注册边界。
- `AgentMessage`、内容块和 `ProviderEvent` 定义消息与流事件边界。
- `RequestAssembly` 定义一次完整模型请求的快照。
- `PermissionEngineResult`、`Constraint`、`Verdict` 定义权限裁决边界。
- `SessionStore`、`SessionRepo` 定义会话存储边界。
- `HookManager`、`AuditLogger` 定义观测边界。
- `CommandSandbox`、`Shell`、`FileSystem` 定义进程和文件系统边界。

Pydantic 模型负责外部输入、事件、消息和配置的结构校验；冻结 dataclass 负责稳定值对象和快照；Protocol 负责跨层依赖；普通可变对象集中在确实需要生命周期管理的运行状态中。

### 1.3 用收敛维护架构

Xcode 通过删除重复路径、回收兼容层、移除未使用功能、拆出单一职责模块来控制维护面。当前架构质量体现在持续收敛：

- 一个行为拥有一个主要入口。
- 一个策略拥有一个决策点。
- 一个事实拥有一个持久化表达。
- 一个跨层对象拥有明确的类型形状。
- 一个前端行为复用运行时事件，并由运行时统一执行语义。

---

## 2. 分层架构与所有权

当前运行路径可以抽象为：

```text
CLI / TUI / Web
       │
       ▼
    XcodeApp
       │
       ▼
 coding_agent 装配与 CodingAgentHarness
       │
       ├── execution modes / tool registry / skills / memory
       ├── ToolGate / PermissionEngine
       ├── session / snapshot / MCP / hooks
       ▼
   通用 AgentHarness
       │
       ▼
      Agent
       │
       ├── request assembly / context / window rollover
       ├── provider stream / tool execution / watchdog
       ▼
       AI provider layer
       │
       ▼
  OpenAI-compatible gateways and provider APIs
```

### 2.1 AI 层：把服务差异压缩在 provider 边界

`ai` 层拥有模型元数据、费用、思考级别、请求选项、流式事件、缓存统计和 provider 工厂。

当前运行时支持以下 transport：

- `openai_chat`
- `deepseek_chat`
- `chatglm_chat`
- `mimo_chat`
- `custom`

四个 OpenAI Chat Completions 风格 provider 共享 `OpenAICompatProvider`。子类负责 provider 专属请求参数、reasoning 内容清理、tool stream 和 usage 字段；公共基类负责消息归一化、工具编码、流读取、usage 拦截、thinking 参数、超时中止和累计成本。

外部 chunk 被转换为统一事件：文本增量、推理增量、工具调用、usage、最终消息和 provider failure。Agent 核心消费统一事件，provider 差异停留在适配边界。

### 2.2 Agent 层：把循环编排保持为纯运行机制

Agent 层拥有：

- 类型化消息和内容块。
- `run_agent_loop` 核心循环。
- 工具参数校验、串行/并行调度和执行结果整理。
- provider 调用和流事件收集。
- token 估算、上下文压缩触发和请求卫生处理。
- 运行指标、终止原因和结果对象。

`Agent` 类保存工具列表与最近结果，并把具体运行配置通过 `AgentLoopConfig` 注入。权限、审计、执行模式、技能、记忆和前端行为由外层提供 hook、工具适配器和上下文，而核心循环保持领域中立。

### 2.3 Harness 层：拥有运行时生命周期

通用 `AgentHarness` 负责：

- 发布和读取当前 Agent composition。
- 建立 active run，拒绝同一 session 的重叠运行。
- 连接取消 token、durable inbox、会话上下文和工具门控。
- 将 Agent 事件翻译为 `AgentHarnessEvent`。
- 把同步 API 与异步流 API 接合。
- 处理 provider fallback、hook、审计和结果构建。

`CodingAgentHarness` 在此基础上加入 plan/build/act、技能激活、todo、目标验收和编码产品状态。领域能力位于产品层，通用生命周期位于 harness 层。

### 2.4 装配层：在运行前组合能力，在运行中消费快照

`build_app` 依次完成配置解析、共享基础设施、provider bundle、工具注册、安全策略、外部 hook、自动 reviewer、compactor 和 Agent 装配。

`AgentComposition` 是一次发布后的能力 generation，包含：

- 主 provider 与可选 fallback provider。
- 工具注册表。
- Agent 配置。
- Gate 配置。
- Request assembler。
- 运行时上下文 provider。

发布时复制并冻结规则、映射、工具 schema 和配置。模型切换、权限策略切换都会生成新的 generation；活动 run 保持当前 generation 的一致视图。

### 2.5 依赖方向：外层适配，内层稳定

当前包级依赖方向遵循由产品和界面指向运行内核的结构：

- `ai` 依赖基础类型和 provider 生态。
- `agent` 使用 AI 协议，拥有循环和消息。
- `harness` 使用 Agent，拥有生命周期、安全、会话、观测和执行环境。
- `coding_agent` 组合 harness 与编码工具。
- `cli` 和 `server` 使用产品 API 与结构化事件。

这使得同一个 Agent 行为能够被单次调用、CLI、TUI、WebSocket、子代理和恢复流程重复使用。

---

## 3. Agent 循环：模型推理与运行时纪律的分工

Xcode 的一次 Agent run 由外层 step 循环和内层模型循环组成。

### 3.1 外层 step 循环

每个 step 依次处理：

1. 取消检查。
2. 从 durable inbox 接收 next-step 输入。
3. 发出 turn 边界事件。
4. 根据实测或估算 token 判断压缩。
5. 进入模型调用、错误重试和续写。
6. 追加 assistant 消息与工具结果。
7. 执行重复工具和空闲步骤看门狗。
8. 运行 `prepare_next_turn`。
9. 运行 `should_stop_after_turn`。
10. 继续工具循环、消费末轮 steer，或构建最终结果。

`max_steps` 默认保持开放；调用方可以显式设置正整数。终止原因被统一编码为 completed、cancelled、step_limit、watchdog、provider_error。

### 3.2 内层模型循环

模型请求之前，`RequestAssembler` 构建完整请求快照。provider 流返回后，循环执行以下规则：

- provider 返回错误时使用指数退避，默认最多三次 Agent step retry。
- provider 流在 `max_tokens` 结束且续写开启时追加 `continue` 用户消息。
- 续写内容低于默认 500 token 时累计低产出次数。
- 连续三次低产出触发结构化 error，结束续写回路。
- 模型完成纯文本回应时进入完成验收或结束 turn。
- 存在工具调用时先执行全部门控，再执行实际 handler。

provider 层自身使用 `ProviderRuntime` 处理临时 HTTP、连接和超时错误的重试；Agent 层处理一次模型回应在循环层面的恢复。这两个层次各自拥有清晰的失败范围。

### 3.3 工具调度

工具默认以并行模式调度，`tool_workers` 默认 4。工具可以声明 `sequential` 或 `parallel`：

- 并行工具合并到同一批次。
- 串行工具切成单调用批次。
- 混合调用保持模型给出的调用顺序进行结果整理。
- 每个工具调用具备开始、更新和结束事件。
- 工具参数先按 JSON Schema 校验，再进入 handler。
- 每个工具拥有独立超时；默认工具超时为 120 秒。

`ToolExecutionEndEvent` 始终携带结构化结果、错误状态、metadata 和 render intent，前端据此选择展示方式。

### 3.4 进度与停机保护

重复工具看门狗按工具名和规范化参数构造签名，连续三次重复调用触发停止。空闲步骤看门狗按工具结果成功状态累计，连续四个结果均为 error 的步骤触发停止。配置可以声明重复检测豁免工具和自定义 productive 判定。

这套机制把模型回路当作需要运行时治理的控制系统：模型可以继续推理，运行时始终拥有结束循环的确定性出口。

### 3.5 目标验收

`GoalController` 保存自然语言停止条件，并在主 Agent 准备结束时调用独立 provider 验收：

- judge 只接收 transcript 和实际工具结果。
- judge 要求返回单个 JSON 对象。
- verdict 包含 `ok`、`reason`，并可声明 `impossible`。
- 判定结果为待完成时生成下一轮反馈。
- 反应次数超过上限时生成明确的 re-entry limit。
- goal 支持 active、paused、resume、clear，并进入可恢复运行状态。

完成声明因此经过独立证据检查，目标状态与模型自我描述保持分离。

---

## 4. 请求与上下文：把 context 当作有预算的运行资源

### 4.1 单一请求组装边界

`DefaultRequestAssembler` 是 provider 请求的唯一组装入口。它按固定顺序执行：

1. 修复历史中的工具调用配对。
2. 收集配置指令、活动 diff、最近验证失败、笔记、技能摘要和运行状态。
3. 对会话级 section 执行 snapshot/diff 渲染。
4. 把 system context、user context、历史消息和工具定义组合起来。
5. 应用确定性的 request hygiene。
6. 编码为 provider wire messages。
7. 生成 context trace、tool trace、token 预算和剩余预算。
8. 将同一份 `RequestAssembly` 提交给 provider 和 before-provider hook。

请求快照记录 source、target、block id、是否纳入、token 数、内容 SHA-256、provenance、截断信息和 scope。provider 与 before-provider hook 因此使用同一组可追踪事实。

### 4.2 结构化 ContextBlock

每个上下文块包含：

- 来源：instruction、skill、active_diff、notes、recent_validation、environment、tools、permissions、mode 等。
- 目标：system 或 user_context。
- 优先级：critical、high、medium、low、background。
- 生命周期：turn/step expiry。
- 元数据：来源、截断和运行信息。
- scope：user、project、runtime。

组装器先过滤过期块，再按优先级排序；相同优先级保持 collector 注册顺序。预算包含 system prompt、历史消息、工具 schema 和上下文块，优先级决定可纳入内容。

### 4.3 World state 与增量投影

`ContextSection` 为动态上下文提供 snapshot、full render 和 diff render。`WorldState` 保存每个 section 的指纹：

- 首次出现时注入完整内容。
- 指纹保持稳定时跳过重复注入。
- 状态变化时注入带 section id 的 replacement。
- section 消失时注入 removed notice。
- 压缩完成后清除 baseline，下一窗口重新建立完整投影。

会话级 `ContextState` 同时保存 persistent messages、request prefix fingerprint 和 world state。一次请求看到的动态环境因此具有增量语义，压缩和恢复具有明确的重新建立点。

### 4.4 当前上下文来源

当前编码 Agent 使用以下上下文来源：

- 项目与用户指令：按 root 到 cwd 的层级发现 `AGENTS.override.md`、`AGENTS.md`、`agents.md`、`AGENTS.txt`，共享累计 32 KB 字节预算。
- 活动 Git diff：提供 staged/unstaged 状态、统计和有限 diff excerpt，预算 8 KB。
- 最近验证错误：保留最近一次相关 shell 错误，预算 4 KB。
- `.xcode/notes`：读取 Markdown 与文本笔记，预算 4 KB。
- 技能目录：注入轻量 skill catalog，正文按激活加载。
- 运行快照：环境、工具、权限和执行模式按状态 section 注入。
- Git preflight：提供状态、最近提交、脏 diff 统计和用户已有修改提示，并使用短 TTL 缓存。
- 当前任务相关文件与工具结果：通过 LRU 记录 active file、recent files 和 recent tool results。

稳定 prompt 区域包含身份、工具纪律、引用说明、工具清单和搜索策略；动态区域包含环境与 cwd；易变区域包含 Git、检索状态和 session notices。稳定区域和 cwd 区域使用缓存，易变区域按轮次重建。

### 4.5 请求卫生与事实保留

请求卫生作用于 provider 投影，保留会话 surface 的完整历史：

- 工具结果文本默认限制 8000 字节。
- 工具参数中的长字符串默认折叠为长度标记，默认上限 1000 字符。
- 长结果保留头部 50 行与尾部 50 行。
- base64 内容转换为数据摘要。
- 工具调用与工具结果配对由历史修复器维护。

会话历史依然保留原始结构，模型请求获得预算可控的投影。这种分离让上下文效率与恢复完整性同时成立。

---

## 5. 会话：事实账本与当前 surface 分离

### 5.1 Append-only tree JSONL

每个 session 使用 JSONL 记录 entry。entry 拥有 id、parent_id、type、content 和 created_at；session index 保存 metadata 与当前 head_id。追加 entry 更新 head，file lock 保护并发写入。

当前会话结构具备：

- 从 head 回溯得到当前 branch。
- 从任意用户消息 fork。
- 完整 clone 当前 session。
- 通过 head 移动实现 rewind 与 tree navigation。
- spawn 空 child session，并记录 parent lineage。
- 用 session title、summary、project path、时间和 parent id 提供列表视图。

### 5.2 Surface 是从事实计算出的模型历史

`SessionSurface` 依次应用 inbox claim、assistant 事实、tool use、tool result 和 context-window reset replacement，生成模型可见消息。

surface 使用带显式 `kind` 与 `payload` 的类型标签编码。解码时校验消息类型、字段和工具配对：

- 每个 tool call 都需要对应 result。
- orphan result 触发 surface 错误。
- 重复 tool call id 触发 surface 错误。
- surface 结尾保留完整的工具调用闭合关系。

换窗 replacement 带有 generation、source entry ids 和 surface SHA-256。重建时验证 replacement 对应整个 branch prefix，并验证新 surface 指纹。

这套结构把两个时间尺度分开：

- transcript 保存发生过的事实。
- surface 保存当前模型窗口的工作投影。

模型窗口可以丢弃并重开，事实账本持续追加；恢复流程重新计算当前 branch，再恢复运行状态、目标、技能和上下文检索状态。

### 5.3 Durable inbox 与 active run

`SessionInbox` 将输入生命周期写成 `inbox/inserted`、`inbox/claimed`、`inbox/discarded` 事件，提供两条 lane：

- `NEXT_STEP`：当前 run 的下一次模型边界消费。
- `NEXT_TURN`：当前 run 结束后启动新的 turn。

`ActiveRunHandle` 管理 running、cancelling、finishing、finished 四个状态，并以锁保护 step input 的接收窗口。末轮输入通过关闭入口后原子 claim，避免生成完成瞬间产生竞态。

用户输入、runtime reminder、steer、follow-up 和 interrupt 都经过 SessionRunController。一个 session 只有一个 active run，消息的去向和唤醒行为被结构化返回。

### 5.4 记录粒度

会话记录链路将稳定语义事件写入 session：

- assistant
- tool use
- tool result
- context window reset
- final
- provider request
- goal state
- subagent descriptor、activation 和 run lineage

流式碎片保持运行时展示属性；持久化层记录足以重建语义和状态的事件。事件 schema 当前版本为 2，provider request envelope 保存实际消息、工具、provider、options、组装预算和 context trace。

### 5.5 文件变更快照与撤销

Git 工程中的每个用户 turn 可以建立 pre/post tree snapshot。快照仓库存放在 `.xcode/snapshots/<session>` 下，使用隐藏 Git tree 保存文件状态；用户工作区的 `.git`、index、stash、HEAD 和 refs 保持独立。

撤销流程依次检查：路径边界、turn changed_files 归属、post snapshot 冲突、PermissionEngine 权限和实际恢复。已被用户继续修改的文件进入 skipped 状态，原始快照保持可追踪。

---

## 6. 工具：把模型意图变成受约束的本地动作

### 6.1 ToolSpec 与 AgentTool 双表示

`ToolSpec` 面向产品注册，包含名称、说明、参数 schema、prompt snippet、prompt guidelines、action profile 和 path extractor。`AgentTool` 面向 Agent 循环，提供异步 execute、取消信号、更新回调、执行模式和 examples。

`ToolSpecAdapter` 把同步 handler 放入独立 daemon 线程，在异步循环中轮询结果；生产 ToolGate 使用带输出脱敏的 adapter。工具因此可以保持简单的同步 I/O 代码，Agent 循环保持异步事件流。

所有发布工具都要求 JSON Schema。模型输入、工具执行输入和 provider 工具定义共享同一份参数形状；冻结 composition 中的 schema 以只读映射发布。

### 6.2 文件工具：精确、可解释、可回退

文件工具集中执行路径、编码、大小、换行、BOM、二进制和输出策略：

- 相对路径锚定项目 root，绝对路径保留规范化结果；ToolGate 的路径边界决定项目外访问，内置规则阻断 `.git`、`.venv`、`__pycache__` 和环境文件模式。
- `read_file` 支持文件、目录、1-based offset/limit、行号、50 KB 输出预算和目录分页。
- 图片按 magic bytes 检测，使用 Pillow 读取，最大边缩放至 2000 像素，数据保留在 metadata。
- `write_file` 用于新文件或有意的整文件替换，返回 unified diff。
- `edit_file` 使用精确 `old_text`/`new_text`，默认要求唯一匹配；`replace_all` 仅在单编辑时启用。
- 编辑保持 BOM 和原始换行风格，Python 文件写入后尝试格式化。
- 单文件写操作经过路径 mutex，减少并发写入交错。
- 文件写入内容默认上限 1 MB。

工具结果同时包含人类可读文本、结构化 metadata 和 `LocationRenderIntent`/`DiffRenderIntent`。前端可以展示位置或差异，模型也能获得同一动作的文本证据。

### 6.3 apply_patch：先解析与计划，再应用

`apply_patch` 支持 add、update、delete、move。处理分为：

1. 解析完整 patch envelope。
2. 校验每个 hunk、操作前缀、锚点和目标路径。
3. 读取所有目标文件并构建 `FileChange` 计划。
4. 逐个精确匹配上下文，失败时给出 verification error。
5. 在文件 mutation queue 中写入变更。
6. 输出文件摘要、unified diff、增删统计和 render intent。

多文件修改因此拥有整体校验、清晰失败和统一结果；单文件小改动保留 `edit_file` 的精确路径。

### 6.4 搜索：发现与阅读分工

`glob_files`、`find_files`、`list_dir` 负责文件发现，`grep_search` 负责内容检索。ripgrep 可用时优先使用，Python walk/grep 提供确定性回退；`.gitignore`、隐藏目录、敏感目录和二进制文件在搜索路径层过滤。

结果拥有数量上限、长行截断、尾部截断和 metadata。项目文件补全使用最多 5000 个文件与 75 ms 时间预算的短生命周期索引。

搜索策略在 Agent 身份中被明确表达为：先做词法发现，再阅读完整相关文件，定位根因，执行聚焦修改，验证修改行为。

### 6.5 Bash：真实 shell 与可见过程

Bash 工具根据平台发现 `pwsh`、PowerShell、Git Bash、bash、zsh、sh、dash、ksh 或 cmd，并把 command、workdir、timeout_ms 转为确定的执行计划：

- workdir 经过项目 root 解析，路径逃逸回退到项目 root。
- 默认超时 30 秒，最大 300 秒。
- `shell=False` 启动进程包装器，POSIX 使用进程组，Windows 使用隐藏窗口进程。
- stdout/stderr 由独立线程排空，避免管道阻塞。
- 取消先终止进程，超时先 SIGTERM，必要时升级到 SIGKILL 或 taskkill。
- OutputAccumulator 保留完整字节统计、行统计、最近预览和滚动窗口。
- 输出超限后惰性 spill 到临时 `.log` 文件，模型看到尾部摘要与完整输出路径。
- `on_progress` 将实时文本转为工具 update 事件。

Shell 结果带有 `TerminalRenderIntent`，CLI、TUI、Web 可以把它呈现为一次终端执行。

### 6.6 交互工具、Web 工具和子代理

`question` 把用户选择和自由文本统一为结构化回答，并允许自定义选项。交互输入受限时返回明确的交互环境提示。

`webfetch` 限定 HTTP(S)，默认输出 Markdown，支持 text/html，响应体上限 5 MB；图片和二进制文件进入明确错误路径。`websearch` 通过 Exa 或 Parallel MCP HTTP endpoint 获取当前外部信息，并对结果进行字节限制。

`subagent` 创建独立 child session：

- one-shot child 用于有界独立工作。
- continuable child 进入 durable inbox，支持后续 turn。
- 批量任务最多 16 个，默认并发 4 个。
- child 只接收显式任务和自身工具集合。
- child 拥有独立 session id、correlation、recorder 和 ToolGate。
- parent 记录 descriptor、activation、run status、summary 和 error。
- child activation 可以释放，durable session 保持可冷恢复。

---

## 7. 安全：语义权限与 OS 隔离共同形成边界

### 7.1 Action 是权限判断的基本对象

权限系统先把工具输入转换为 `Action`：

- `tool`：工具名。
- `capability`：read、write、edit、patch、shell、skill、mcp 等。
- `operation`：具体动作。
- `targets`：path、command、domain、mcp、skill 等目标。
- `unresolved_effects`：变量、glob、命令替换、wrapper、解析错误和危险命令。

工具可以通过 `action_profile` 声明能力和 target kind，通过 `path_extractor` 提取多文件目标。`apply_patch` 因此能在真正写入前暴露 patch 中的全部目标路径。

### 7.2 PermissionEngine 的唯一决策路径

ToolGate 在每个 turn 创建冻结 snapshot，把当前模式、规则、审批者、grant store、目录边界和工具映射交给 PermissionEngine。决策顺序为：

1. `restricted_dirs`、项目边界、敏感路径、Git metadata 和危险命令形成硬约束。
2. Path boundary、execution mode、静态 policy、hook constraint 和 shell policy 形成约束集合。
3. RuleMatcher 对规则执行 last-match-wins；用户规则追加在默认规则之后。
4. PermissionResolver 按 deny、ask、allow 的顺序选择最严格约束。
5. ask 进入 session/permanent grant 查询。
6. `approval_policy` 决定 ask 是否进入 reviewer。
7. user 或 auto reviewer 返回最终 HITLResult。

每次裁决携带 decision、blocked、reason、reason code、overrideable、remediation、matched rule、source、metadata、approval result 和 action。拒绝结果因此同时具备机器可读诊断和用户可读修复方向。

### 7.3 Plan / Build / Act

三种执行模式分别表达各自的自主性边界：

- **Plan**：模型可见只读探索、搜索、问答和 Web 能力；技能可以通过显式激活进入会话；`write_file`、`edit_file` 的允许范围限定为 `.xcode/plans/*.md`；模式 fallback 为 deny。
- **Build**：全部工具可见；项目内结构化写入直接执行；规则覆盖范围外的 shell 与动作进入自动 reviewer；模式 fallback 为 ask。
- **Act**：全部工具可见；只读工具直接执行；写入与 shell 默认进入用户审批；模式 fallback 为 ask。

Plan 具有最大 investigation turn 计数，达到上限后自动进入 Build 并发出模式通知。执行模式存入 `CodingRunState`，恢复 session 时一并恢复。

### 7.4 路径、凭据和外部目录

路径边界通过规范化、resolve、符号链接检查和 root containment 判断：

- 项目内路径获得 boundary allow，`.git` metadata 直接拒绝。
- `.env` 与 `.env.*` 进入敏感路径策略；`.env.example` 拥有独立写入语义。
- `.aws`、`.ssh`、`.kube`、`.docker`、`.npmrc`、`.pypirc`、密钥文件名等凭据路径直接拒绝。
- `.venv` 与 `__pycache__` 属于内置 blocked workspace path。
- 项目外路径需要 `external_directories` 中配置目录的 access 覆盖读、写或读写权限。
- `sensitive_path_overrides` 只接受精确路径，并且只允许受限环境文件的明确访问例外。
- `non_workspace_access` 关闭后，用户配置的外部目录白名单停止生效；Xcode 自身的 `~/.xcode` 与 `~/.agents` 保留只读基础设施访问。

### 7.5 Shell 分析器保持保守语义

POSIX、PowerShell 和 cmd 拥有对应分析器。分析器识别只读命令、纯观察命令、写命令、文件路径、管道、重定向、变量、glob、命令替换和组合控制语法。

`rm -rf /`、主机级关机/重启、权限提升、`git reset --hard`、强制 `git clean` 等危险命令直接拒绝。未知命令、动态路径和待确认的 wrapper 根据当前模式进入 ask 或 deny。静态分析表达“已确认的效果”，OS sandbox 负责实际进程边界。

### 7.6 审批、授权和自动 reviewer

审批范围由动作真实 target 决定：

- once：当前执行。
- session：当前 session 的同类 target。
- permanent：项目级持久授权。
- 多 target 动作只提供 once。
- auto reviewer 只允许 once，自动决策永远获得单次授权。

session grant 使用进程内 session store；permanent grant 使用 `.xcode/approval_grants.json`，通过 file lock、临时文件、fsync 和 replace 写入。用户界面负责展示与选择，PermissionEngine 负责 grant 查询和写入。

Build 的自动 reviewer 在独立 provider 会话中工作，接收有界 transcript、精确 action、工作目录和 turn id。reviewer 的证据规则把 system/user 内容视作授权证据，把 assistant、tool call、tool result、approval reason 和 planned arguments 视作需要审查的证据；高风险动作需要足够授权，critical 风险拒绝，超时和 provider failure 进入 failed-closed 路径。

### 7.7 Linux OS sandbox

执行环境提供 `CommandSandbox` 接口，Linux 使用 bubblewrap：

- 默认模式为 `workspace-write`，默认网络为 deny。
- `read-only` 提供只读文件系统视图。
- `workspace-write` 绑定项目 root 与获准可写 root。
- `danger-full-access` 提供完整访问模式。
- 用户、PID、IPC、UTS namespace 被隔离；网络 deny 时创建独立网络 namespace。
- 进程能力全部 drop。
- `.git`、`.agents`、`.xcode` 等 protected workspace paths 以只读方式挂载。
- 凭据与环境文件通过 `/dev/null` 或 tmpfs 遮蔽。
- 缺失的 protected path 使用临时 placeholder，并在结束时校验 inode/device 后清理。
- cwd 必须位于项目 root 内。

Linux 之外的环境使用本地 `SubprocessShell`；语义权限边界仍由 ToolGate 和 PermissionEngine 执行。

---

## 8. 观测：每个重要动作都留下可定位的关联关系

### 8.1 Hook 事件

当前 hook 事件包括：

- `before_agent_start`
- `before_provider_request`
- `pre_tool`
- `post_tool`
- `on_error`
- `on_context_window_reset`

`SignalHookManager` 同时支持同步注册、结构化订阅和后台 hook。后台队列把外部观察从主循环中分离，`drain_background` 提供受控收尾点。

### 8.2 Correlation

`RuntimeCorrelation` 为 session、turn、request 和 tool call 分配关联身份，并附加 UTC timestamp。相同关联字段进入：

- AgentHarnessEvent。
- HookRecord。
- provider request envelope。
- tool audit record。
- session provider request event。
- WebSocket 事件。

一次动作可以沿着 session → turn → request → tool call 追踪完整路径。

### 8.3 审计与脱敏

可配置 `JsonlAuditLogger` 保存工具输入、输出、动态决策、policy decision、授权范围、grant id、reviewer、risk、authorization、rationale 和最终状态。常见 API key、token、secret、password 和 `sk-` 形式在工具、MCP、hook 和审计边界执行脱敏。

### 8.4 外部命令 hook

外部 hook 由配置声明 event、argv、matcher、timeout、failure policy 和 subagent inheritance。执行采用 `shell=False`、JSON stdin/stdout 和输出长度上限。

`pre_tool` hook 可以返回参数变换与决策，但决策合并采用更严格方向；外部 hook 可以收紧 allow 为 ask 或 deny，现有安全边界始终保持有效。每个 hook 保存 run count、last status、last error 和 last run time，运行状态可由交互界面查询。

---

## 9. 记忆与技能：渐进披露，保持当前任务清晰

### 9.1 技能采用 catalog-first、body-on-demand

SkillRegistry 先发现 `SKILL.md` frontmatter，保存名称、描述、来源、兼容性、allowed-tools、references、scripts 和 assets 元数据。技能目录按 explicit、project、user 优先级搜索；重复名称 first-wins。

模型上下文只接收可见技能摘要。任务匹配后通过 `load_skill` 显式加载正文；reference 文件另行按名加载，单个引用拥有 50 KB 读取上限。激活内容带 `skill-activation-state`，进入 session surface，并在压缩和恢复时成对保留。

技能的 `allowed-tools` 以 advisory 方式披露，权限 bypass 明确保持关闭。项目技能默认等待 `trust_project_skills` 开启，显式目录拥有最高优先级。

### 9.2 记忆承担跨 session 的可复用事实

当前记忆实现使用两个 Markdown 事实文件：项目 `MEMORY.md` 与用户 `~/.xcode/memory/MEMORY.md`。每个 H2 section 形成 `MemoryRecord`，记录 layer、title、body 和稳定 memory id。

检索使用确定性的 BM25：

- 英文、代码、路径、中文字符和中文 bigram 进入 token 序列。
- exact match 与 token overlap 增强结果分数。
- 项目层在相同条件下优先于用户层。
- 结果最多返回 10 条。
- 文件 inode、mtime、size 组成索引签名，变化触发重建。

记忆写入支持显式 add、update、delete；标题或正文重复被拒绝；文件锁、临时文件、fsync 和 replace 保证原子更新。恢复 session 时可以按最多 6000 token 读取记忆概览；普通 turn 只接收记忆使用协议，模型通过 `search_memory` 按需检索。session surface 承担当前任务连续性，MemoryManager 承担跨 session 的规则、架构决策、验证事实和可复用方案。

### 9.3 MCP 采用延迟发现与运行时快照

MCP 配置以 server 为单位校验，server 工具以 `mcp__<server>__<tool>` 规范化命名。运行时支持：

- server 配置 hash 与协议/server identity 校验的工具缓存。
- `defer_loading` 延迟工具列表读取。
- fetch tools 与 mcp tool search 引导冷启动。
- MCP SDK stdio client 的单 owner task 生命周期。
- protocol capability 与 roots 协商。
- tools/list 分页，最多 100 页。
- tools/list_changed 后的刷新。
- 连接失败的有限重连。
- call progress、取消、超时和优雅关闭。
- text、image、audio、resource link、embedded resource 与 structuredContent 转换。
- outputSchema 校验与 MCP 结果脱敏。
- host tool id collision 检测，冲突工具整体停用。

MCP 的外部能力通过普通 ToolSpec 进入同一个 ToolGate、同一个参数校验流程和同一个 session event pipeline。

---

## 10. 前端：事件消费者与运行时的多个观察面

### 10.1 CLI 与 TUI 共享行为合同

CLI 使用 prompt_toolkit、questionary、Rich 和统一命令注册表；TUI 使用同一组命令、工具、HITL 预览、补全、会话、事件和渲染辅助。

CLI 提供：

- `/plan`、`/build`、`/act` 模式切换。
- `/model`、`/thinking`、`/effort` 动态模型控制。
- `/sessions`、`/resume`、`/continue`、`/fork`、`/clone`、`/tree`、`/rewind`。
- `/new-context`、`/context`、`/goal`、`/permissions`、`/hooks`、`/mcp`、`/memory`。
- `/tool` 直接工具入口和 `!command` shell 快捷入口。
- `$skill` 显式技能激活与 `@file` 文件引用。
- Tab 补全、参数暗示、实时 Markdown、推理预览和工具摘要。

TUI 使用 inline transcript 形式呈现运行过程：步骤轨道、思考区、assistant 内容、工具卡片、授权列表、问题选择器、滚动视口和状态栏。流式刷新采用节流与缓存；已完成消息保留 ANSI 渲染缓存，长输出按视口分页。

Ctrl+C 的语义按状态分层：先清空输入，再取消活动 run，空闲状态连续触发退出；工具、provider、snapshot 和 child run 各自执行协作式收尾。

### 10.2 Web：单一运行舞台，多浏览器观察

Web server 通过 REST 提供 info、stats、model、git branches、workspaces、sessions 和 transcript；WebSocket `/ws` 推送 hello、user message、run lifecycle、structured agent event、approval request、workspace/session switching 和错误。

`WebRunHub` 持有单一 `XcodeApp`、单一活动回合和多个 sink：

- 一个浏览器提交的事件广播给所有连接。
- 工具线程产生的审批请求通过线程安全队列进入 asyncio loop。
- 浏览器返回 approval decision、scope 和 suggestion。
- cancel 可以终止审批等待或活动回合。
- 模型、effort、工作区和 Git 分支在活动回合外切换。
- 工作区切换重新装配完整 app，再替换旧 app。
- 历史会话提供只读 transcript 展示，并支持恢复后继续对话。

Web 前端直接消费结构化 event type，把 text_delta、reasoning_delta、tool_use、tool_update、tool_result、context_window_reset 和 final 映射为步骤、思考面板、工具卡片和终止统计。

---

## 11. 配置：用户表达意图，运行时完成合并与校验

运行时配置由 global、project、local 和 environment 层组成，按 raw dict 的显式键递归合并，再通过 Pydantic 完整校验。profile 先展开继承，环境覆盖获得字段来源提示，未知字段进入校验错误。

配置对象覆盖：

- provider profiles、模型、base URL、thinking、effort、context window。
- Agent max steps、自动换窗、reserve token、tool workers、watchdog。
- request hygiene。
- shell 与 tools。
- skills、prompt modules、instruction sources。
- session path、skills path、audit path。
- security、external directories、sensitive overrides、permission rules。
- plan/build/act rulesets、fallback 和 shell unresolved policy。
- external hooks。

交互式设置浏览器与首次启动向导共享同一份 `XcodeRuntimeConfig` 校验。用户通过语义选项编辑 approval policy，写入时展开为 `approval_policy` 与 `approval_router`；枚举、布尔、整数、浮点、路径和字符串列表拥有对应解析器。

配置的核心理念是把“用户想要的运行方式”转换为“可以逐字段验证、逐层覆盖、逐 generation 发布的运行事实”。

---

## 12. 这套实现最终形成的工程哲学

### 12.1 证据驱动的行动

搜索、文件读取、diff、验证命令、工具结果和 provider response 都进入上下文或 session 结构。Agent 身份要求技术结论建立在观察到的文件、命令输出、验证结果或 provider response 上；上下文还保留来源、行号、摘要、截断和指纹。

### 12.2 最小副作用路径

结构化文件工具承担清晰、可解释的修改；复杂 shell 保持审批；多文件变化使用 patch；写操作返回 diff；撤销检查冲突；敏感路径拥有硬边界；所有工具进入统一门控。副作用距离模型越近，描述越结构化，审计越具体。

### 12.3 运行时拥有最后的确定性

模型可以请求工具、继续生成、选择目标、建议动作和声明完成。运行时负责 schema、超时、取消、权限、OS sandbox、看门狗、goal judge、surface pairing 和持久化校验。模型拥有推理空间，系统拥有执行边界。

### 12.4 连续性来自可重建事实

上下文压缩、会话恢复、分支、fork、rewind、子代理和浏览器刷新都依赖持久化事实与可验证投影。当前窗口可以替换，输入可以排队，child 可以冷恢复，文件可以按 snapshot 回滚，provider request 可以追踪。

### 12.5 前端只改变观察方式

CLI、TUI、Web 和单次 prompt 使用同一个 XcodeApp、同一个 AgentHarness、同一个 ToolGate、同一个事件模型和同一个 session recorder。前端负责输入、审批、展示和交互节奏；执行语义保持一致。

### 12.6 复杂度通过生命周期治理

技能按需激活，MCP 按需发现，记忆按需检索，context 按预算纳入，工具输出按需 spill，provider 按故障回退，hook 按事件执行，子代理按 activation 管理。每种资源都有创建、使用、观察、取消、持久化和释放路径。

### 12.7 可替换性来自狭窄边界

provider 可以替换，shell 可以替换，filesystem 可以替换，sandbox 可以替换，审批 UI 可以替换，hook manager 可以替换，session store 可以替换，前端可以替换。替换发生在 Protocol、值对象、事件和 composition 边界上，核心循环保持稳定。

---

## 13. 结语

Xcode 的设计重心是一条完整的本地执行链：

```text
用户目标
  → 证据收集
  → 预算化上下文
  → 类型化模型请求
  → 流式推理
  → Action 提取
  → 权限与边界裁决
  → 工具执行
  → 验证与目标验收
  → 事件、审计、快照与 session 持久化
  → 可恢复的下一步
```

这条链路把 Agent 从一次性文本生成提升为可操作、可中断、可审批、可观察、可恢复的本地工程系统。它的轻量来自核心循环的清晰与依赖方向；它的可靠来自边界、事实、预算、类型和生命周期的共同约束；它的长期价值来自每次运行都能留下可继续工作的结构化状态。
