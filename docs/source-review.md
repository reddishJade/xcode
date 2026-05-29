# Xcode 源码级深度审查与架构分析报告

基于 2026-05-26 当前独立 checkout 源码分析。本文是一份全景式、高颗粒度的源码级能力地图与系统性审查报告，严格从 **Coding Agent 核心工程组件（常驻约束、ACI 工具、控制与拦截、物理隔离、状态自愈）** 的五个诊断层面进行解构，揭示 Xcode Harness 骨架的底层硬核实现与工程边界。

---

## 一、 系统定位与运行大图

Xcode 是一个轻量级 Python coding agent harness 骨架。其核心设计思想是**模型负责短期推理，Harness 负责长期约束、确定性拦截与状态自愈**。系统运行流程如下：

```plaintext
      人机协作/终端指令 (REPL Session Management)
                 │
                 ▼
     决策与事件主循环 (StructuredAgent Loop)  ◄─── [看门狗防陷入 (Watchdog)]
                 │                           ◄─── [协作取消 (CancellationToken)]
                 ▼
      上下文提示装配与注入 (SystemPromptBuilder)
   ┌─────────────┼─────────────┬─────────────┐
   ▼             ▼             ▼             ▼
[常驻契约]     [局部路径]     [动态滑窗]     [按需懒加载]
CLAUDE.md    .claude/rules  Contextual    Skills (SKILL.md)
                 │
                 ▼
       动作能力与 ACI 工具层 (ToolCatalog & MCP)
   ┌─────────────┴─────────────┐
   ▼                           ▼
[内置核心工具]               [外部 MCP stdio 连接] (Lazy Refs & Schema 缓存)
read/write/edit/bash/test   mcp__<server>__<tool>
                 │
                 ▼
        确定性拦截与安全屏障 (ToolExecutor)
   ┌─────────────┼─────────────┐
   ▼             ▼             ▼
[三态权限拦截] [HITL 人工审批] [Secret Redaction 脱敏] (sk- / key=)
allow/deny/ask   once/session  run_tool_result & audit两路拦截
                 │
                 ▼
        物理隔离沙箱开发 (Subagents & Worktrees)
   ┌─────────────┴─────────────┐
   ▼                           ▼
[物理会话分叉 (Plan Exit)]   [隔离物理沙箱 (Git Worktrees)]
fork_clean_into("isolate")   Cherry未合并检测/脏工作区拦截
                 │
                 ▼
        自愈式状态安全重建 (Context Restore)
  restore_read_versions 逆向会话历史扫描与物理哈希自愈校验
```

---

## 二、 项目拓扑与模块职责

```text
xcode/
├── src/xcode/main.py                         # CLI 入口：REPL、单次 prompt 及会话恢复选择器
├── src/xcode/cli/                            # 终端人机协作交互层
│   ├── repl.py                     # prompt_toolkit REPL 主循环、斜杠命令解析及 Ctrl+C 协作取消拦截
│   ├── session.py                  # SessionStore 管理（Fork 隔离会话、Plan Exit 方案物理落盘）
│   ├── completion.py               # 终端 Tab 自动补全（斜杠指令、@file 引用及可见工具）
│   ├── file_refs.py                # 动态展开会话中的 @file 物理文件引用
│   └── markdown.py                 # 终端 Markdown 高级渲染器
├── src/xcode/harness/                        # 核心运行与状态托管层
│   ├── app.py                      # 运行时唯一装配中心：build_real_app()
│   ├── config.py                   # 基础与运行时配置数据类
│   ├── event_bus.py                # 核心事件分发总线
│   ├── session.py                  # SessionStore 转发适配器
│   ├── skills.py                   # ToolSpec 工具定义、三层 HITL 审批流
│   ├── skill_loader.py             # SKILL.md 轻量目录扫描与 load_skill 按需正文加载工具
│   ├── tool_wrapper.py             # 工具底层包装拦截
│   ├── types.py                    # 基础运行类型定义
│   ├── agent_runtime/              # 核心决策与控制环
│   │   ├── structured.py           # StructuredAgent 解耦决策循环与单步 Watchdog 签名看门狗
│   │   ├── tool_executor.py        # 工具并发调度、审计过滤及结果格式化
│   │   ├── execution_modes.py      # Plan/Review/Act 三态变速执行策略与模式流转 notices
│   │   ├── subagent.py             # 子代理 spawn 隔离运行与 ManagedSubagentRunner 控制
│   │   ├── prompting.py            # SystemPromptBuilder 动态提示词装配工程
│   │   ├── contextual.py           # contextual-retrieval 动态滑窗状态（文件/工具结果缓存）
│   │   ├── compaction.py           # LayeredCompactor 五层上下文压缩及大输出预算裁剪
│   │   ├── git_preflight.py        # git status 与 dirty diff stat 动态注入器
│   │   └── cancellation.py         # CancellationToken 线程安全协作式取消广播
│   ├── tools/                      # 系统内置工具包
│   │   ├── file.py                 # read/write/edit（版本指纹校验及 restore_read_versions 会话状态重建）
│   │   ├── code_search.py          # glob_files、grep_search 词法代码搜索
│   │   ├── bash.py                 # 动态三级 CommandRiskEvaluator 评估及 Popen 异步生命周期管理
│   │   ├── shell_adapter.py        # 宿主系统 Shell 环境（pwsh/bash）自动探测与适配
│   │   └── operations.py           # 常规工具操作辅助
│   └── observability/              # 可观测性与审计拦截
│       ├── audit.py                # 结构化审计日志输出与 Key 敏感词脱敏
│       ├── permissions.py          # PermissionPolicy 权限规则拦截（allow/deny/ask 三态）
│       └── hooks.py                # HookManager 生命周期钩子
├── src/xcode/ai/                             # 传输与模型对接层
│   ├── types.py                    # 共享的强类型模型接口与响应定义
│   ├── stream.py                   # 响应流拦截与事件产生适配器
│   └── providers/                  # 模型连接与传输适配器
│       ├── factory.py              # build_provider_bundle() 统一工厂、重试与 429 抖动避退
│       ├── codec.py                # 统一的 OpenAI tool schema 与流式 reasoning_delta 编解码器
│       ├── deepseek.py             # DeepSeek 增强传输与 thinking mode 支持
│       ├── openai.py               # chat_completions 与 responses_stateful 推理传输
│       ├── anthropic.py            # Anthropic Messages 适配
│       ├── mimo.py                 # MiMo 适配层
│       ├── faux.py                 # 单元测试用 Mock provider
│       └── runtime.py              # 客户端执行期控制 (Retry & RateLimit)
├── src/xcode/experimental/                   # 可选扩展特性层（opt-in）
│   ├── mcp.py                      # stdio MCP Client，Content-Length 消息帧及 MD5 指纹缓存
│   ├── speculation.py              # SpeculationPlanner 步骤投机分类与 UI 预热事件调度
│   ├── tasks.py                    # 任务存储、依赖解析、终端 Kanban 视图与 tasks 工具组
│   ├── mailbox.py                  # 基于 filelock 与事件日志型的多进程子代理异步邮箱
│   ├── memory.py                   # 基于 H2 契约校验、BM25 召回、元数据重排和 eval 的 MEMORY.md 动态内存管理器
│   ├── worktree.py                 # WorktreeTaskRunner 隔离沙箱及脏状态/未合并 cherry 移除拦截
│   ├── plugins.py                  # PluginManager 扫描加载本地插件与 Hooks 注册
│   ├── bm25.py                     # 纯 Python BM25Okapi 核心算法实现
│   └── speculation.py              # 安全预热事件管理器（不触发副作用的安全预测）
└── docs/                                     # 深度架构与源码审查文档
```

---

## 三、 诊断层面一：Context Surface (上下文提示装配与 Prompt 缓存工程)

Xcode 将整个 Prompt 渲染流程围绕 **前缀精准匹配（Prompt Caching）** 与 **动态渐进式披露** 设计。

### 3.1 提示词 Caching 布局与前缀保护（`prompting.py`）
`SystemPromptBuilder.build()` 强制采用“静态内容在前、动态变动在后”的排版格局，最大化利用 Key-Value 推理缓存，削减 90% 的首轮 Token 计算开销：
1. **System Prompt & Identity (常驻静态)**：模型身份、思考约束，绝对锁定。
2. **Tool Definitions (常驻静态)**：按分组过滤后的工具 Schema，由于 MCP 连接数稳定，保证前缀匹配。
3. **Environment & Static conventions (半固定静态)**：OS、Python 版本、`CLAUDE.md` + `CLAUDE.md` 内置规则（各截断 4000 字符）。
4. **Git Preflight & CWD & dynamic notices (动态追加)**：Git 短状态、Commit 摘要、dirty diff stat（由 `git_preflight.py` 注入）、根目录文件列表视图。
5. **Contextual Retrieval (每轮动态)**：最近访问的 8 个文件和最近 6 条工具结果摘要（每条截断 240 字符）。

### 3.2 Git Preflight 代码感知注入（`git_preflight.py`）
避免模型漫无目的地扫描，每次推理开始前利用 `git status --short` 快速收集被改动的文件列表，利用 `git log -n 1 --oneline` 提供当前 HEAD 提交线索，并在 dirty 状态下计算 diff 摘要。该数据在提示词后方注入，使模型获得无感的环境代码态感知。

---

## 四、 诊断层面二：Action Surface (动作能力与 ACI 工具设计)

Harness 遵循 **ACI (Agent-Computer Interface)** 视角设计工具包，重点在于“工具能被用对”与“出错能被纠偏”。

### 4.1 内置核心工具与强一致性防御（`tools/file.py`, `tools/bash.py`）
- **Read-Before-Edit 强指纹保护**：`edit_file` 不允许模型临场盲改。系统在 `read_file` 执行时将 Path 的 SHA1 哈希、修改时间（`mtime`）和文件大小（`size`）捕获为 `read_versions` 指纹。在 `edit_file` 执行前进行强校验，若检测到外界改动（哈希冲突），强行阻断并报错，迫使模型重新读取，构建了严格的原子修改边界。
- **Popen 轮询 cancel 异步终止机制**：`bash` 工具在底层彻底抛弃了阻塞式的 `subprocess.run`，改用带有自旋轮询的 `subprocess.Popen`。通过将 Popen 对象的生命周期与 `CancellationToken` 及超时限制挂钩，在 cancel 信号被触发时执行 `terminate()` 并退避至 `kill()`，杜绝了执行挂起时长命令产生孤儿僵尸进程的隐患。
- **动态 CommandRiskEvaluator 三级判定**：
  - `deny`：阻断 `rm -rf`、`sudo`、网络请求且 `network_commands=deny` 等操作。
  - `ask`：匹配 `git push`、`pip install`、`npx` 等网络命令且 `network_commands=ask`，挂起触发 HITL 人工审批。
  - `allow`：允许静态只读或安全的系统探测前缀。

### 4.2 渐进式披露与 Skills 目录加载（`skill_loader.py`）
- **轻量目录扫描（`SkillLoader`）**：`SkillLoader` 在冷启动时只扫描 `skills/` 下 `SKILL.md` 的 frontmatter，保留 `name`、`description`、`use_when`、`dont_use_when`、`tools`、`risk` 和文件路径，不把正文提前放入内存或 system prompt。
- **模型自主选择**：`SystemPromptBuilder` 在启用 `"skills"` 模块时注入 `<skill-catalog>`，提供所有 skill 的轻量目录信息和 `load_skill({"name": "skill-name"})` 加载方式。Harness 不再用 Jaccard、BM25 或 `SkillRouter` 在主路径替模型硬选 skill。
- **`load_skill` 按需正文加载**：启用 `"skills"` 组后，系统注册 `load_skill` 工具。只有模型判断当前任务需要某项技能时，才读取对应 `SKILL.md` 正文并返回完整 SOP，将大型运行手册与常驻系统提示剥离。

### 4.3 外部 MCP stdio 延迟连接连接（`src/xcode/experimental/mcp.py`）
- **Content-Length 异步双向长连接**：在 stdio 管道上运行带有 `Content-Length: ...\r\n\r\n` 首部的消息帧进行高效的异步编解码。
- **MD5 配置指纹缓存**：首次加载 MCP 服务时，计算配置（command, args, env）的 MD5 摘要指纹。若与 `.local/mcp_cache.json` 的指纹一致，秒级免启动直接从缓存提取工具 Schema 字典；错配或缺失时冷启动服务获取 `tools/list` 并刷新指纹缓存，完美规避了冷启动大量外部 Node/Python 进程的延迟开销。
- **延迟连接（Lazy Connection）**：使用 `LazyClientRef`。在初始化注册阶段不建立任何 IO 连接；只有当模型判定并首次唤醒调用特定的 `mcp__<server>__<tool>` 命令时，底层进程管道才会正式 `start` 建连，杜绝了空闲进程对系统资源的白白侵占。
- **敏感词正则高风险判定**：`get_mcp_tool_risk` 采用正则敏感词智能扫描。当工具名或描述中包含 `write`/`delete`/`create`/`run`/`exec` 等变动命令词时，动态评定为 `"high"` 风险，其余根据 `read`/`view`/`list` 评定为 `"low"`，未知一律降级为安全中风险（`"medium"`），并无缝集成到 `run_tool_result` 审批和审计流程中。

---

## 五、 诊断层面三：Control Surface (确定性拦截、脱敏与安全门禁)

安全机制应当硬性编码化，而不依赖模型自身的临场发挥。

### 5.1 三层 HITL (Human-in-the-Loop) 校验决策链
`skills.py` 中的 `run_tool_result()` 是工具调度前的核心拦截卡点，其审查顺序如下：

```plaintext
1. 检查 Session/Persistent Policy ───► [判定为 deny] ───► 立即返回 ToolExecutionResult("denied")
2. 调用 tool.risk_evaluator()     ───► [判定为 deny] ───► 立即返回 ToolExecutionResult("denied")
3. 检查是否需要授权审批 ──────────► [ 静态 risk=="high" 或 策略/动态判定为 ask ]
                                             │
                                             ▼
                                     [触发 approval_callback]
                                 ┌───────────┴───────────┐
                                 ▼                       ▼
                           [用户选择 Allow]         [用户选择 Deny]
                           继续执行工具 handler     中止执行，返回 "denied" 状态
```

这一整套三层审批门禁靠近 Dispatch 侧，当新增工具时，仅需声明其 `risk="high"` 或绑定 `risk_evaluator`，Harness 行为即刻保持强一致。

### 5.2 确定性 Secret Redaction 脱敏审计（`audit.py`）
`redact_text()` 采用正则强力捕获并进行文本脱敏（Redact）：
1. 捕获并强制重写 OpenAI 等 API key（`sk-[A-Za-z0-9_-]{8,}`）为 `[REDACTED]`。
2. 捕获并屏蔽 key/secret/token 键值对分配（`api_key = "..."` 结构）为 `key=[REDACTED]`。
- **双路审查脱敏**：
  - **前路拦截**：`run_tool_result()` 对工具 handler 的输出结果进行脱敏后返回给 LLM，防止模型读取到敏感凭证并泄露给第三方。
  - **后路审计**：`ToolExecutor` 在向磁盘的 `JsonlAuditLogger` 写入审计日志时，再次对输入及输出执行 `redact_text()` 脱敏，保证审计日志绝对安全。

---

## 六、 诊断层面四：Isolation Surface (物理隔离与子 Agent 并行)

长周期任务最大的隐患在于大量试错的垃圾数据与噪声将注意力稀释，Xcode 采用会话与物理文件系统双向隔离设计。

### 6.1 物理会话隔离与 Plan Exit 状态流转（`session.py`, `repl.py`）
当用户执行 `/act --clear` 指令或在交互界面从 Plan Mode 确认切换为 Act Mode 时，系统自动触发 **Plan Exit**：
- **Plan 状态提取**：从当前会话的 transcript 事件流中反向遍历，提取出最后一条 `assistant` 生成的 Plan 设计方案内容。
- **方案物理落盘**：附加 Approved 元数据与日期，物理落盘于 `.local/session_artifacts/plan-{parent_id}.md` 保证归档。
- **Isolate Fork 无损分叉**：调用 `SessionStore.fork_clean_into("isolate", ...)` 新建一个 `.jsonl` 文件。继承 parent_id 引用链，但是**物理上彻底丢弃原会话的全部历史 transcript 会话消息**（彻底抹除 Token 债务）。
- **无痕前置注入**：将 Plan 内容缓存在 `state.approved_plan` 中。在进入新隔离会话首轮执行时，将其以 `<approved-plan>` 标记动态前置注入到 System Prompt 头部。
- **Transcript 历史零污染**：最关键的工程细节在于，这段被临时动态注入的 `<approved-plan>` **绝不写入新会话的 transcript 事件落盘中**，仅仅参与首轮 LLM 交互的上下文建构。使得会话在逻辑上继承方案指导，但在历史和 Token 负荷上达到极致的精简与纯粹。

### 6.2 分身沙箱隔离开发（`src/xcode/experimental/worktree.py`）
高自主度 Agent 应当被限定在物理沙箱工作树中工作：
- **沙箱隔离**：`WorktreeTaskRunner` 基于 `git worktree add -b <branch> <path> HEAD`，在不打扰开发者主开发区脏工作树的前提下，将当前提交克隆并在 `.xcode-worktrees/{task_id}` 目录内建立一个全新的分支开发工作树，达到极佳 of 沙箱环境。
- **脏文件拦截器**：为防止在尚未完成或未提交内容前误删工作，非强制模式移除（`remove_worktree_task`，`force=False`）提供了强大的拦截边界。首先调用 `git status --porcelain`，若检测到工作树中有任何未提交的脏改动（即跟踪/未跟踪的修改与暂存），将直接拒绝删除。
- **合并缺失审计**：如果在分身沙箱中产出了提交，移除器会尝试利用 `git rev-parse --abbrev-ref @{u}` 获取上游，运行 `git cherry` 比对；如果不存在 upstream，则智能获取主开发分支（如 `main` 或 `master`）作为基准，运行 `git cherry` 探测是否存在有差异却还未合并回主开发线的临时提交。如果发现此类“漂泊”的独立提交，将立刻阻断物理删除，杜绝因误动作或模型逻辑混乱而彻底丢失重要模型产出。

### 6.3 受控子代理协作机制（`agent_runtime/subagent.py`）
主 Agent 作为 Orchestrator 统筹全局，下挂子 Agent 独立跑扫库、测试等易产生大输出的任务。
- **独立上下文与摘要回收**：子 Agent 在 fresh messages 里运行，拥有独立的 Loop，运行细节留在子 Agent 消息历史中，主 Agent 仅回收其最终摘要结论，防止主线程上下文污染。
- **ManagedSubagentRunner 资源受控**：设置最大递归深度限制防止无限递归；在 `subagent` profile 下使用独立的轻量级 model_profile；限制并发 worker 上限（2 worker）与执行超时限额（120s），支持通过 job ID 异步状态查询及 cancel 取消机制。

---

## 七、 诊断层面五：Verification Surface (状态自愈与评测闭环)

### 7.1 分层上下文压缩与 Token 预算裁剪（`compaction.py`）
LayeredCompactor 实现了五级智能渐进式压缩：
1. **Stale Snip 冗余读取裁剪**：根据 `tool_use_id -> read_file path` 映射，扫描消息流。同一路径的 `read_file` 历史结果中，仅保留最新一次为完整文本，旧有的全部用 `[Content snipped - re-read if needed]` 替换，清除多余的冗余 Token 债务。
2. **Large Output Budget 头尾裁剪**：当估算 Token 超过配置阈值的 50% 时，扫描大型 `tool_result`，若字数超限且不属于最新 read 结果，自动以 `[... truncated N characters due to token budget ...]` 保留头尾进行截断。
3. **Micro Compact 占位符压缩**：将旧的工具输出内容变为 `[Previous tool_result compacted; N chars removed]` 占位符，默认仅保留最近 2 次。
4. **Transcript Save 完整落盘**：压缩前，将含有全量完整细节的消息历史以 `transcript_*.jsonl` 物理持久化落盘，确保可回溯审计。
5. **Summary Compact 折叠**：将过于久远的消息通过 LLM 合并为单条折叠的 system 摘要。

### 7.2 自愈式状态安全重建机制（`restore_read_versions`）
在 Compaction 上下文压缩或 REPL 状态恢复（Resume）后，内存态中的 `read_versions` 缓存会被清空，导致 read-before-edit 安全防御机制触发不一致的阻断。
- **逆向扫描与哈希重建**：`restore_read_versions` 在 Agent 每次 `run_stream` 开始及压缩后自动执行。逆向递归检索传入的 `messages` 历史，反查提取 `assistant` 的 `tool_use`（解析 `read_file`/`edit_file`/`write_file` 轨迹）与 `user` 的 `tool_result`。
- **物理自愈比对**：若对应的 `read_file` 已经被 Snip 裁剪，恢复器主动读取磁盘上的当前实体文件：若实体文件的 SHA1 仍与历史未裁切时的哈希一致，说明文件未被外部篡改，则以磁盘最新 `mtime` 和 `size` 重新还原 `read_versions` 缓存指纹，实现完全自愈；若哈希冲突，则强制清空，迫使模型在编辑前必须重读。
- **写轨迹逐出**：若检测到后续存在 `write_file`/`edit_file` 的 ok 成功写入，则主动逐出对该 Path 的读指纹缓存，强一致性地要求后续改动前必须重读。

### 7.3 Watchdog 执行断流规避（`agent_runtime/structured.py`）
- **重复签名拦截**：`_check_repeated_tool_watchdog` 机制以单步为粒度，在主循环里捕获模型工具调用的唯一签名：`signature = f"{tool_name}:{stringify_tool_input(input)}"`。如果与上一次签名完全一致，则计数自增。一旦超过限制（默认 3），主循环强制拉闸阻断（`stopped_by_watchdog=True`），杜绝完全重复的物理死循环。
- **语义空转熔断 (Semantic Idle Failsafe) 与模式绑定**：针对参数微调（如微调 grep query 或 read offset 导致签名不一致但语义等价）的隐性死循环，后续将升级引入“无实质产出熔断”——当连续 4 步无任何成功的文件写入或 bash 指令执行产出时触发熔断。为了防止误杀 legitimate 的只读探索任务，该熔断将**与 `execution_modes.py` 的三态架构天然配套绑定**：仅在 `Act Mode` 下严格执行，在 `Plan Mode` 与 `Review Mode` 下自动放宽或禁用，消除冗余状态。

---

## 八、 上下文管理

### 8.1 压缩分层
利用 LayeredCompactor，由 `compact_threshold` 触发分层压缩，对老旧日志做 Micro compact 及 Large output 截断，保留 transcript 物理备份供回查。

### 8.2 上下文隔离
父 Agent 调用 `submit_subagent` 时，子 Agent 在 fresh messages 里运行。通过 `ManagedSubagentRunner` 控制并发、超时与递归深度，阻断父子 Agent 上下文相互污染。

### 8.3 Contextual Retrieval
滑窗自动记录最近访问的 8 个文件和最近 6 条工具结果摘要（每条截断 240 字符），强制作为 contextual-retrieval 注入下一轮 System Prompt。

### 8.4 自愈式状态安全重建机制（`restore_read_versions`）
在大模型 Compact 压缩折叠或会话 Resume 重载后，`restore_read_versions` 逆向分析消息历史。若发现 `read_file` 结果已被 snip 裁剪，则对磁盘物理文件拉取校验：哈希一致则静默复原 `mtime` 与 `size` 等缓存指纹屏障，保证 read-before-edit 防护不坍塌；若冲突则彻底逐出，以强一致状态确保自愈。

---

## 九、 人机协同会话流转与 Plan Exit (Plan-to-Act 状态衔接)

在 Plan Mode 漫游积累庞大 Token 债务后，通过 `/act --clear` 执行 **Plan Exit**：
- 逆向抓取最后一次 Plan 并持久化物理落盘为 `.local/session_artifacts/plan-{parent_id}.md`。
- 调用 `SessionStore.fork_clean_into("isolate", ...)` 新建物理干净 `.jsonl` 分叉会话，彻底清除 Token 历史债务。
- 缓存 Plan 并在新隔离会话首轮以前置 `<approved-plan>` 形式无痕注入 System Prompt 头部，且该注入**绝不写入新会话 transcript 中**，确保注意力 100% 精简化聚焦。

---

## 十、 Harness 工业级 Gap 分析与长期进化缺陷

Xcode 作为一个轻量级 Harness 骨架，与行业一流大厂（如 Claude Code 或 OpenAI Harness 基础设施）的硬核开发实践对照，存在以下五个根本性“硬伤空缺”：

### 10.1 评测体系从固定脚本走向事件流 Pipeline
- **当前实现**：`src/xcode/evals/` 已提供 `EvalTask`、`EvalRunner`、JSONL trace 和 grader。runner 消费 `XcodeApp.aask_stream()` 的 Agent 事件流，对最终答案、工具调用、禁止工具和工具错误数进行判定；旧的 `eval_harness.py` 仍保留为核心工具 happy path 冒烟测试。
- **设计取舍**：评测入口绑定事件协议而不是工具函数，这样能够覆盖模型选择工具、工具结果回灌、最终回答这条完整 Agent Loop。`EvalRunner.arun()` 是原生异步入口，`run()` 只是 CLI/脚本兼容 wrapper。
- **剩余边界**：当前 grader 仍是确定性规则，不包含 LLM-as-judge、Pass@k 统计和外部 benchmark 数据集接入；这些应在 trace 格式稳定后继续扩展。

### 10.2 可观测性 APM 断路与结构化外部反馈闭环缺失
- **致命缺陷**：Xcode 虽然有 HookManager 和 JsonlAuditLogger，但这仅仅是**事后的单向审计**，没有构建“ Trace + 事件流 + 多路消费”的 APM 闭环，Harness 缺乏提供给模型用作自我修正的反馈回路。
- **大厂对比**：Agent 在面对报错时，能够自主调用 trace 工具查询系统的 APM Trace 拓扑。Vector 将日志、指标和 tracing 数据发送到 Victoria 存储中，Agent 可以自主运行 TraceQL、LogQL、PromQL 进行因果关联和故障定位，主动校验改动是否生效，而非流于将粗暴的控制台错误扔进上下文让模型猜测。

### 10.3 跨会话程序性/语义记忆系统的边界
- **当前能力**：Xcode 的 `src/xcode/experimental/memory.py` 已从单纯 transcript 物理落盘前进到轻量语义记忆：`MEMORY.md` 以 H2 incident 块为基本单元，强制包含 `Context/Query`、`Solution`、`Files`、`Takeaways`，写入失败时归档到 `.local/memory_archive/`，避免损坏主记忆文件。
- **检索策略**：当前实现不是“向量库”，也不是只看 BM25 分数的纯词袋系统。BM25 负责第一层低成本召回；随后根据 `Scope`、`Source`、`Confidence`、`Status` 等元数据做可解释重排，过期记忆会被降权，匹配调用场景的记忆会被提升。这样保留本地、确定、低依赖的工程特性，同时给冲突解决和溯源留下显式字段。
- **评估闭环**：`MemorySearchEvalCase` / `evaluate_search()` 提供最小 top-k 召回评测，重点验证“相关记忆是否能被召回到前列”，而不是用主观回答质量替代检索质量。后续更重要的方向是扩大评测集，覆盖措辞漂移、过期记忆降权、同题多解冲突、局部更新和来源可追溯。
- **剩余空缺**：目前 memory 仍是实验层能力，不会默认常驻注入主 Agent prompt；还缺少自动 consolidation 的质量门、跨 session 冲突合并、引用级 provenance、长期遗忘策略和可选 hybrid retrieval。只有当记忆规模或语义漂移超过 BM25 能处理的范围时，才应考虑向量召回作为第二路候选，而不是把向量库作为默认复杂度。

### 10.4 Skill 目录预算与描述质量约束
- **当前边界**：Xcode 已移除主路径里的 `SkillRouter` 词袋路由，改为向模型暴露轻量 `<skill-catalog>`，由模型根据 `description`、`use_when` 和 `dont_use_when` 决定是否调用 `load_skill`。这避免了 Harness 用 Jaccard 或 BM25 替模型做脆弱的硬选择。
- **剩余风险**：当 skill 数量持续增长时，完整目录本身也会占用上下文预算；如果 skill 描述过长、触发词后置或缺少反例，模型仍可能误判是否加载正文。
- **后续方向**：需要做的是目录预算控制与 skill 描述规范，而不是恢复代码侧路由。预算控制只负责裁剪目录长度，不能把裁剪逻辑伪装成“自动选择 skill”的决策层。

### 10.5 任务认领与可重入断点现场（`TaskProgress`）的漂移
- **致命缺陷**：Xcode 现在的 `src/xcode/experimental/tasks.py` 提供的是任务调度和目录锁保护，不是 `agent.md` 所强调的跨 Session “可重入、可断点继续”的长任务编排控制组件，模型极易在中途崩溃时彻底偏航。
- **大厂对比**：针对长耗时任务，采用 Initializer Agent 和 Coding Agent 角色协同。Initializer 仅在开始时将任务落盘为结构化的 `TaskProgress` 物理状态控制对象（如 `feature-list.json`）；后续会话由 Coding Agent 每次启动从进度文件和 `git log` 自愈恢复现场，每做完一步就持久化落盘 `save_progress`，强制同一时间仅有一个 `in_progress` 进度锚点。

---

## 十一、 可观测性与审计大盘 (APM & Observability)

Xcode 建立了无感的结构化可观测性底盘：

| 组件 | 角色 | 输出格式与格式 | 核心内容 |
|------|------|----------------|----------|
| `JsonlAuditLogger` | 持久化安全审计 | `.local/audit_*.jsonl` | `AuditRecord` 实体，含 session_id、tool、static_risk、dynamic_decision、policy_decision、final_status、approved (基于 final_status=="ok" 判定)、脱敏后的 redacted_input 及 redacted_output。 |
| `HookManager` | 运行时生命周期拦截 | 内存事件分发 | 触发 `pre_tool`、`post_tool`、`on_error`、`on_compact` 钩子，提供 HookRecord。 |
| `PermissionPolicy` | 动态规则决策 | 内存规则匹配 | 接受 allow/deny/ask 判定，按工具名和输入片段检索匹配。 |

---

## 十二、 Experimental 可选扩展特性（按需引入）

通过 `tools.enabled_groups` 进行细粒度 opt-in 激活，所有扩展工具执行前依然经过 Harness 内置的多层安全与审计屏障。

### 12.1 MCP stdio Client 接入机制（`src/xcode/experimental/mcp.py`）
`xcode.experimental.mcp` 提供了遵循 Model Context Protocol 的 stdio 客户端接入，实现了与系统内置工具的无缝桥接：
- **JSON-RPC 通信传输**：运行 stdio 的双向长连接，采用带有 `Content-Length: ...\r\n\r\n` 首部的消息帧进行高效的异步编解码。
- **配置多重加载**：在 `build_mcp_tools()` 执行时，优先在 `.local/mcp_config.json` 寻找服务配置，若不存在则降级查找根目录下 `mcp_config.json`，支持独立配置启动命令、命令行参数（args）及环境变量（env）。
- **Schema 指纹缓存与冷启动规避**：首次解析配置时，会计算命令、参数及环境的 MD5 哈希指纹。若配置无变化，则直接从 `.local/mcp_cache.json` 中秒级提取工具 Schema 缓存；若配置指纹失效或缓存不存在，才会临时冷启动服务器调用 `tools/list` 刷新本地缓存，避免了每次会话加载都要唤醒一众外部进程的冷启动开销。
- **延迟连接（Lazy Connection）**：工具注册时仅生成并绑定带有 `LazyClientRef` 的 `ToolSpec`；只有当模型判定并首次唤醒调用特定的 `mcp__<server>__<tool>` 命令时，底层进程管道才会正式建连（`start`），彻底杜绝了空闲进程对系统资源的白白侵占。
- **动态风险判定机制**：`get_mcp_tool_risk` 采用正则敏感词智能扫描。当工具名或描述中包含 `write`/`delete`/`create`/`run`/`exec` 等变动命令词时，动态评定为 `"high"` 风险，其余根据 `read`/`view`/`list` 评定为 `"low"`，未知一律降级为安全中风险（`"medium"`），并无缝集成到 `run_tool_result` 审批和审计流程中。

### 12.2 Speculation 安全界面预热机制（`src/xcode/experimental/speculation.py`）
提供安全的非侵入式“事件预测”，为宿主系统及 REPL/Web 客户端渲染线程进行极速 UI 准备：
- **状态流追踪分类**：`FinishKindTracker` 通过分类器 `classify()`，对 Agent 最近一轮步骤 of 工具名与执行状态（`status`）进行追踪归档（如将 `edit_file` 分类为 `file_edit`，将 `bash` 归为 `bash`，或提取 `error`/`denied` 等阻断状态）。
- **投机动作调度**：`SpeculationPlanner` 结合以上信息规划 `SpeculationEvent` 事件。例如当文件编辑完成时发送 `prepare_diff_view` 事件，使得界面预先获取并加载差异比对界面；当 Bash完成时调度 `prepare_terminal_buffer` 准备终端滚动条；当遭遇阻断或拒绝时分发 `prepare_recovery_hint` 渲染恢复助手。
- **绝对副作用安全**：Speculative 动作严格限制为只读的前端界面渲染加速与数据预取，**禁止且没有**任何涉及后台文件写入或指令执行的实际动作，达到了体验极致流畅与安全隔离的和谐共存。

### 12.3 Tasks 任务存储、拓扑排序与 Kanban 视图（`src/xcode/experimental/tasks.py`）

实现了一套无外部数据库依赖、具有高一致性的分布式轻量级任务队列存储系统：
- **数据落盘组织**：每个任务作为一条结构化的 `TaskRecord`，单独落盘在 `.local/tasks.json.d/{id}.json` 文件中，状态在 `pending`、`claimed`、`completed` 之间严密迁移，并且由文件 `.highwatermark` 维护单调递增的全局任务 ID。
- **并发锁屏障（FileLock）**：采用 `filelock.FileLock` 库级文件锁取代目录锁，保证高并发下任务状态读写的原子性与事务强一致。
- **拓扑排序依赖解析**：提供 `resolve_task_dependencies` 函数，根据任务 payload 中的 `blocked_by` 依赖关系对所有任务进行拓扑排序，能够自动识别并阻断循环依赖。
- **Kanban 看板美化渲染**：提供 `render_kanban_view` 函数，将当前的任务状态列表分类为 `[PENDING]`、`[CLAIMED]` 和 `[COMPLETED]`，并打印带子任务进度及被阻塞信息的终端看板。
- **Agent 工具集导出**：支持 `build_task_tools` 导出 `create_task`、`update_task`、`list_tasks`、`get_task` 一组 Agent 可直接唤起调用的任务图维护工具，实现任务流的纯闭环管理。

### 12.4 Worktree Git 托管隔离沙箱（`src/xcode/experimental/worktree.py`）

为了在复杂长任务的试错或开发中保障主开发区的绝对安全，引入了基于 Git 特性的隔离物理沙箱：
- **物理分身与隔离开发**：调用 `create_worktree_task` 时，`WorktreeTaskRunner` 自动利用 `git worktree add -b <branch> <path> HEAD`。在不打扰主开发区脏工作树的前提下，将当前提交克隆并在 `.xcode-worktrees/{task_id}` 目录内建立一个全新的分支开发工作树，达到极佳的沙箱环境。
- **脏文件拦截器**：为防止在尚未完成或未提交内容前误删工作，非强制模式移除（`remove_worktree_task`，`force=False`）提供了强大的拦截边界。首先调用 `git status --porcelain`，若检测到工作树中有任何未提交的脏改动（即跟踪/未跟踪 of 修改与暂存），将直接拒绝删除。
- **合并缺失审计**：如果在分身沙箱中产出了提交，移除器会尝试利用 `git rev-parse --abbrev-ref @{u}` 获取上游，运行 `git cherry` 比对；如果不存在 upstream，则智能获取主开发分支（如 `main` 或 `master`）作为基准，运行 `git cherry` 探测是否存在有差异却还未合并回主开发线的临时提交。如果发现此类“漂泊”的独立提交，将立刻阻断物理删除，杜绝因误动作或模型逻辑混乱而彻底丢失重要模型产出。

### 12.5 Plugin 动态加载与 settings.json 权限安全沙箱（`src/xcode/experimental/plugins.py`, `src/xcode/harness/observability/permissions.py`）

- **Plugin 动态扫描与按需载入**：`PluginManager` 使用标准库 `importlib` 对 `.local/plugins/` 目录下的 `.py` 自定义插件进行扫描加载，能自动提取并隔离注册其中的 `exposed_tools` 工具集、生命周期 Hooks 钩子 (`exposed_hooks`) 以及外部 SOPs 技能 (`exposed_skills`)，防止全局命名空间污染。
- **settings.json 沙箱安全门禁**：`SettingsSandboxPermissionPolicy` 动态读取 `.local/settings.json` 或 `settings.json`，提供 `allowedTools` 白名单机制、`deniedTools` 黑名单阻断以及受限敏感物理目录边界 `restrictedDirs` 的硬性安全审查拦截（回退至三态 HITL）。
- **MCP 风险持久化与显式覆盖**：MCP 服务在启动时将提取出的工具静态 risk 评级缓存至 `.local/mcp_cache.json` 中，并在加载时优先解析 `mcp_config.json` 中的 `"overrides"` 覆写配置（支持风险降级），将最终控制权完整交予用户。

---

## 十三、 源码审查总结与 NEVER 铁律

Xcode 的 Source Review 深度展现了生产级 Coding Agent Harness 的精细治理智慧。本报告提炼出以下 **Agent NEVER 铁律**，要求在后续迭代与模型交互中严格遵守：

1. **NEVER** 绕过 `restore_read_versions` 自愈机制；禁止随意清空或绕过 `read_versions` 文件指纹校验；任何编辑前必须持有读缓存。
2. **NEVER** 绕过 `run_tool_result()` 直接执行工具 handler；所有工具调度必须流经 HITL 审批决策与脱敏路径。
3. **NEVER** 随意修改 active 注册表中的 static_risk 等级；敏感变动必须通过 dynamic risk_evaluator 评定。
4. **NEVER** 绕过 Popen lifecycle 控制执行阻塞式 `subprocess.run` 指令；所有命令行操作必须持有 `CancellationToken` 并支持协作取消。
5. **NEVER** 允许子 Agent 破坏 depth 深度屏障；子 Agent 必须仅持有 Tooling/Workspace/Runtime 最小提示，严禁带入 Skills SOP 与 Memory。

---

## 十四、 DeepSeek 核心增强技术适配与边界处理

Xcode 在对接支持 Thinking Mode 的 DeepSeek 模型时，针对其特殊的 API 响应规则与状态回传限制，实现了全套自适应编解码与边界保护机制：

### 14.1 深度推理内容捕获与聚合（`deepseek.py`）
- **流式增量合并**：DeepSeek 在 thinking 模式下会并行或先后输出 `delta.reasoning_content` 和 `delta.content`。系统在流式解析中对这两类 delta 进行独立聚合，生成包含完整 `reasoning_content` 和 `content` 的 `AssistantMessage`，并通过强类型事件通知上层展现推理过程。
- **非流式解析**：在单次同步 `complete()` 调用中，系统直接映射 `response.choices[0].message.reasoning_content` 字段并回灌到响应状态实体中。

### 14.2 活跃 Tool Loop 状态机中的 `reasoning_content` 回传防御
DeepSeek API 强制规定：若在 thinking 模式下发生了 tool call，则该 assistant 消息的 `reasoning_content` 必须在后续 tool result 的上下文中完整回传；而在发起新的一轮 User Turn 时，又必须彻底移除历史消息中的 `reasoning_content`，否则 API 将直接返回 400 错误。
- **基于 Turn 边界的动态清理（`_clean_reasoning_content()`）**：
  - **新一轮交互（非活跃 Tool 循环）**：当消息队列中最后一条消息的角色不是 `tool` 时，表明正处于新 Turn。系统会在发起请求前，遍历所有历史 assistant 消息，彻底 strip 掉其 `reasoning_content` 字段。
  - **活跃 Tool 循环中**：当消息队列最后一条消息的角色为 `tool` 时，表明仍在当前 Turn 的 Tool Loop 中。系统将保留 **最后一条 `user` 消息之后** 的 assistant 消息的 `reasoning_content`，而将在此之下的更久远的 `reasoning_content` 全部清除，以防累积冗余或错配导致 400。

### 14.3 空响应重试（Empty-Content Retry）与 API 规范兼容（`codec.py`）
- **空 JSON 响应防御**：在开启 JSON Output Mode（`json_object`）时，DeepSeek 有概率在 thinking mode 结束后输出空的 `content` 且不包含 `tool_calls`。系统在 `complete()` 中增加了基于 Runtime 的重试守护，判定此情形时自动进行最多 1 次静默重试，规避非确定性空包。
- **`content: None` 的严密映射**：针对当模型调用工具时 `content` 为 null 的情况，系统在序列化回传时，保持映射为 JSON 的 `null`（Python 中的 `None`），从而保持与 DeepSeek/OpenAI 官方 Schema 验证的 100% 兼容。
- **Strict Tools Schema 支持**：在 `codec.py` 中实现了对 tool definitions 参数的严格格式转换，支持为工具 schema 自动添加 `"additionalProperties": false` 和必填字段列表，并显式指定 `"strict": true` 以提升大模型工具调用的准确率。
