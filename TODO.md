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

### 有场景再动（依赖实际需求驱动）

#### 0. 队列模式（REPL 异步化前置）

- **当前能力**：引导中断模式已实现。Ctrl+C 中断 LLM 回复后保存 partial response 并允许注入新消息。
- **缺口**：用户不能在 LLM 回复时直接打字输入（sync `for event in _ask_stream(...)` 阻塞了主线程）。队列模式需要输入事件和流事件双路复用，当前 sync REPL 架构不支持。
- **触发条件**：REPL 主体改为 asyncio 双路架构后再实现。队列模式本身逻辑简单——`agent.follow_up()` 已就绪，只差让 REPL 在流式期间接受输入。

### 有场景再动（依赖实际需求驱动）

#### 1. Tasks + Progress 编排能力增强

- **当前能力**：`tasks` 提供依赖排序和 Kanban 视图，`progress` 提供 checklist 保存/恢复。能表达任务图和进度。
- **缺口**：可重入长任务中断恢复（除最基础 `resume_task_progress`）、任务超时/重试、子任务自动分发。
- **触发条件**：有用户需要跨 session 长任务编排或嵌套子任务时再实现。保持轻量，不引入外部编排引擎。

#### 2. Daemon 生命周期管理完善

- **当前能力**：`HeartbeatDaemon` 由 `build_app()` 构造，但启动/停止需调用方在 `main.py` 中手动调用。
- **缺口**：缺少健康检查、自动重启、生命周期回调注册机制。
- **触发条件**：daemon 在真实场景中长期运行暴露出稳定性问题时再优化。

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
