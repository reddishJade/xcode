# Xcode TODO

本文记录下一阶段设计和实现的方向。TODO 中仅保留未完成的计划，按优先级排序。

## 当前边界

- 默认路径继续保持：REPL/CLI -> StructuredAgent -> core tools -> permission/risk/audit -> final answer。
- 默认工具组继续保持 `tools.enabled_groups=["core"]`。
- 新能力必须先证明真实使用场景，默认以 opt-in group 或 `experimental.*` 进入。
- 不做默认启用的 MCP 全量工具注入（防 MCP 接口挤爆 Prompt Cache）。
- 不做不可观测的自动 swarm（多 Agent 必须受控于邮箱总线与物理沙箱）。
- 不做绕过权限系统的外部工具直连。
- 不做面向企业平台的 RBAC、Grafana、Phoenix、RAGAS 等集成。

---

## 待实现

---

### 1. Session 并发安全（借鉴 Pi-mono Phase 锁）

- **当前**：`SessionStore` 无任何并发保护，多 CLI/daemon 同时操作同一 session 会产生竞态。
- **做法**：引入轻量 `session.lock()` 上下文管理器（`filelock`），在 `fork_into`、`compact_current_session`、`rewind_turns` 等写操作前获取锁。不引入 pi-mono 完整 phase 状态机，只解决实际的 session 竞态。

### 2. 分支摘要自动压缩

- **当前**：有 `BranchSummaryMessage` 类型和 session fork，但无自动压缩非活跃分支的机制。分支上下文只会在被显式引用时加入，不会自动摘要后替换原始消息来释放 token。
- **做法**：在 `LayeredCompactor` 中增加分支摘要层。当上下文紧张时（超过 token 阈值），定位不活跃分支，调用 LLM 生成摘要，以 `BranchSummaryMessage` 替换原始分支内容。

### 3. Turn Snapshot 隔离

- **当前**：turn 执行中修改 model/config/skills 可能影响进行中的 turn，产生不一致行为。
- **做法**：`StructuredAgent.execute_turn()` 在 turn 开始时冻结当前 config/tools/skills 快照，turn 内使用快照而非全局引用。pi-mono 称此为 "Turn Snapshot"。

---

### 有场景再动（依赖实际需求驱动）

#### 0. 队列模式（REPL 异步化前置）

- **当前能力**：引导中断模式已实现。Ctrl+C 中断 LLM 回复后保存 partial response 并允许注入新消息。
- **缺口**：用户不能在 LLM 回复时直接打字输入（sync `for event in _ask_stream(...)` 阻塞了主线程）。队列模式需要输入事件和流事件双路复用，当前 sync REPL 架构不支持。
- **触发条件**：REPL 主体改为 asyncio 双路架构后再实现。队列模式本身逻辑简单——`agent.follow_up()` 已就绪，只差让 REPL 在流式期间接受输入。

#### 1. Tasks + Progress 编排能力增强

- **当前能力**：`tasks` 提供依赖排序和 Kanban 视图，`progress` 提供 checklist 保存/恢复。能表达任务图和进度。
- **缺口**：可重入长任务中断恢复（除最基础 `resume_task_progress`）、任务超时/重试、子任务自动分发。
- **触发条件**：有用户需要跨 session 长任务编排或嵌套子任务时再实现。保持轻量，不引入外部编排引擎。

#### 2. Daemon 生命周期管理完善

- **当前能力**：`HeartbeatDaemon` 由 `build_app()` 构造，但启动/停止需调用方在 `main.py` 中手动调用。
- **缺口**：缺少健康检查、自动重启、生命周期回调注册机制。
- **触发条件**：daemon 在真实场景中长期运行暴露出稳定性问题时再优化。

#### 3. 模型模式解析（`model:thinking_level` 三段式）

- **当前**：CLI 模型切换只支持 `--thinking` flag，不能将 thinking level 编码到模型名中。无 fuzzy match，无用户自定义模型注册表。
- **做法**：在 `repl_settings.py` 中增加 `sonnet:high`、`anthropic/claude-sonnet-4-5` 模式解析，分离 provider / model ID / thinking level 三段。
- **触发条件**：用户反馈当前模型切换方式不方便时再做。

#### 4. 类型化事件流

- **当前**：`HookManager` 用字符串回调名分派事件，类型约束弱，扩展者需知道回调名。
- **做法**：定义统一 `HarnessEvent` union 类型（`MessageStart | ToolCall | QueueUpdate | ResourcesUpdate`），提供 `subscribe()` 强类型订阅。保留 `HookManager` 做内部桥接。
- **触发条件**：TUI/Web UI 需要消费 harness 事件流时再做。

#### 5. ExecutionEnv 抽象（跨平台 FS/Shell）

- **当前**：tool 实现直接调用 `subprocess` / `pathlib`，非本地场景（web sandbox）难复用。
- **做法**：定义 `ExecutionEnv` protocol（`read_file`/`write_file`/`exec_shell`），实际 tool 通过它操作。默认实现调 `subprocess`，web sandbox 可注入 mock。
- **触发条件**：有 web sandbox 或远程执行场景需求时再动。

---

### 当前不做（无使用场景证明）

#### 3. MCP SSE / WebSocket 传输协议支持

- **现状**：stdio MCP 已正常工作，项目内所有 MCP server 都是本地进程。
- **不做原因**：远程 MCP Server 的使用场景在项目内未出现。SSE 和 WS 传输层会显著增加客户端复杂度（重连、认证、消息分帧），在有真实需求前不值得投入。
- **标记**：`wontfix` 直到有用户请求跨主机 MCP。

#### 4. OAuth 2.0 (PKCE) 认证与系统 Keyring 凭据存储

- **现状**：API key 管理通过环境变量和 `.env` 文件，满足当前所有 provider 配置需求。
- **不做原因**：OAuth 管线 + OS keychain 集成是企业级安全增强，项目当前无企业部署场景。凭据管理的复杂度远高于收益。
- **标记**：`wontfix` 直到出现多用户共享主机的部署需求。

#### 5. SpeculationPlanner 消费端实现

- **现状**：`SpeculationPlanner` 已能生成 UI 预热事件，但无宿主 UI 消费。属于有生产者无消费者。
- **不做原因**：消费端属于宿主 UI 的职责范围，不在 agent harness 内实现。如果将来 xcode 附带自己的 web UI，届时再对接。
- **标记**：`external`（依赖外部项目）。

#### 6. Session 索引大数量性能优化

- **现状**：`.local/session_index.json` 在数百条级别工作正常，未验证数千条场景。
- **不做原因**：预优化。等真实用户反馈索引加载变慢时再切换为按需加载或分片存储。

#### 7. 配置迁移测试

- **现状**：无正式测试覆盖 `xcode.config.json` 结构变更时的迁移逻辑。
- **不做原因**：预优化。当前配置结构稳定，等下次 breaking change 时同步补充迁移测试。
