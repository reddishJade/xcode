# 运行时架构与一次回合

Xcode 将编码 Agent 组织为分层运行时。每层拥有自己的状态、协议和失败边界，外层通过装配把能力组合成一个可运行的 app。

## 1. 分层

```text
CLI / TUI / Web
      │
      ▼
   XcodeApp
      │
      ▼
coding_agent product
  tools · modes · skills · memory · assembly
      │
      ▼
 generic AgentHarness
  lifecycle · inbox · hooks · gate · results
      │
      ▼
      Agent
  loop · context · request · tool scheduling
      │
      ▼
   AI provider layer
  protocol · codec · streaming · usage · fallback
```

- **AI**：`ModelProvider`、provider transport、模型元数据、费用、thinking、流式事件和 API 错误。
- **Agent**：消息、内容块、核心循环、工具执行、请求组装、压缩和终止原因。
- **Harness**：active run、取消、durable inbox、事件翻译、权限 gate、审计、session 连接和结果。
- **Coding product**：编码工具、Plan/Build/Act、技能、长期记忆、todo、Goal、MCP 与应用装配。
- **Host**：CLI、TUI、Web 的输入、审批和事件展示。

依赖由外向内指向稳定协议。前端消费事件和 app API，工具通过注册表进入 Agent，provider 通过统一协议进入循环。

## 2. 一次回合

```text
用户输入
  → SessionInbox inserted
  → active run claim
  → context prefix + session surface
  → ContextBlock / world state collection
  → RequestAssembly
  → provider request envelope
  → provider stream
  → assistant message / tool calls
  → ToolGate + PermissionEngine
  → tool handler
  → tool result / hooks / audit
  → session events
  → next model step or final result
```

`XcodeApp` 负责装配共享的 provider、工具注册表、session recorder、inbox、context rollover、skills、memory、MCP 和安全配置。每个 run 捕获一个 `AgentComposition` generation；provider、工具、配置、静态 gate 策略和请求组装器保持同一代视图。

## 3. 三种状态表达

### 运行状态

`AgentContext` 保存本次循环的 system prompt、request prefix、messages、tools、context state、project root、cwd 和 request budget。`ContextManager` 统一管理 history、world state、token usage、换窗计数和 prompt/cache fingerprint。

### 事件状态

Agent 核心事件包括 turn、message、thinking、tool execution 和 context-window reset。Harness 翻译为结构化 `AgentHarnessEvent`，前端消费 text delta、reasoning delta、tool use、tool update、tool result、context-window reset 和 final。

### 持久状态

Session recorder 保存稳定语义事件。流式碎片用于实时展示；JSONL 账本保存可重建模型历史、工具配对、换窗 replacement、provider request、Goal 和子代理谱系。

## 4. 请求组装

`RequestAssembly` 是模型请求的完整快照，包含：

- 修复后的 typed messages。
- provider wire messages。
- 工具定义与 JSON Schema。
- context trace、token 数和剩余预算。
- request hygiene 状态。
- 当前 step 与 StreamOptions。

provider 和 `before_provider_request` hook 消费同一份 assembly。请求卫生裁剪 provider 投影，session surface 保留原始结构。

## 5. 失败与停止

运行时具备多层出口：

- provider 临时错误：ProviderRuntime 重试并分类 HTTP 错误。
- Agent provider error：指数退避并限制 step retry。
- `max_tokens`：自动续写，低产出连续达到阈值后结束。
- 工具异常、超时和取消：转换为结构化 ToolResult。
- 重复工具和连续错误结果：watchdog 停止。
- Goal judge：对完成声明进行独立验收。
- `max_steps`：显式步骤上限。

最终结果包含 answer、messages、tool calls、steps、termination reason、metrics、provider failure 和 run state。
