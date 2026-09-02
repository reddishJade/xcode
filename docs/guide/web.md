# 浏览器工作台

Xcode Web workbench 由 FastAPI 服务、WebSocket 事件通道和单页前端组成。服务端持有一个 `XcodeApp` 和一个 `WebRunHub`，浏览器负责输入、展示、取消和审批。

## 1. 启动

```bash
xcode web
xcode web --host 127.0.0.1 --port 8787 --open
xcode web --project-root /path/to/project
```

参数：

- `--host`：监听地址，默认 `127.0.0.1`。
- `--port`：监听端口，默认 `8787`。
- `--open`：服务启动后自动打开浏览器。
- `--project-root`：工作区根目录。

服务默认绑定 loopback。将 host 改为外部可访问地址前，先配置合适的网络、访问控制和运行权限。

## 2. 页面区域

- 顶栏：工作区、Git branch、模型、effort、session 和 plan/build/act。
- 侧栏：新建工作台、最近工作区和 session 列表。
- 主区：用户消息、step rail、thinking、assistant Markdown、tool cards、context-window reset 和 final metrics。
- 底栏：usage、context、模型与 effort。
- 审批面板：工具名、参数、决策原因、相关 transcript 和 once/session/permanent 选择。

实时思考和工具输出可以折叠；工具卡片保留参数、状态、输出和自动授权说明。

## 3. REST API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/info` | 当前 project、session、model、MCP 和 Git branch |
| GET | `/api/stats` | usage、context、model、effort、provider |
| GET/POST | `/api/model` | 读取或切换 main profile 模型与 effort |
| GET/POST | `/api/git/branches` | 读取或切换本地/远端 branch |
| GET/POST | `/api/workspaces` | 读取最近工作区或重新装配工作区 |
| GET | `/api/sessions` | 列出 session |
| POST | `/api/sessions` | 新建 session |
| POST | `/api/sessions/resume` | 恢复指定 session |
| GET | `/api/sessions/{session_id}` | 读取 session transcript 展示数据 |

运行中切换 workspace 或 Git branch 返回冲突状态；模型替换和 session 操作也遵循 active run 生命周期。

## 4. WebSocket `/ws`

连接成功后收到 `hello`。客户端消息：

```text
{"type": "submit", "text": "检查当前修改", "mode": "build"}
{"type": "cancel"}
{"type": "approval", "id": "...", "decision": "allow", "scope": "once"}
{"type": "ping"}
```

服务端消息包括：

- `run_started`、`run_idle`、`run_cancelled`、`run_error`。
- `user_message`。
- `event`：结构化 AgentHarnessEvent。
- `approval_request`：工具预览、允许 scope、原因和有界 transcript。
- `session_reset`、`session_switched`、`workspace_switched`。
- `pong`。

`event` 内的 Agent 事件包含 `message_start`、`text_delta`、`reasoning_delta`、`assistant`、`tool_use`、`tool_update`、`tool_result`、`context_window_reset` 和 `final`。

## 5. 运行与广播

`WebRunHub` 维护单一活动 run。所有 WebSocket 客户端观察同一事件流：

1. 浏览器提交任务。
2. hub 广播 `run_started` 和 user message。
3. 独立执行线程消费 `XcodeApp.ask_stream`。
4. 事件通过 asyncio loop 安全广播。
5. run 结束后广播 `run_idle`，刷新 session 与统计。

工具线程中的同步审批请求通过 hub 转换为浏览器 approval panel。审批等待最多 300 秒，超时按 deny 返回。

## 6. 历史与工作区

点击 session 可以读取 transcript；恢复后当前工作台切换到该 session，后续提交继续写入该账本。页面实时视图可以清空而不删除服务端 session。

工作区切换会由 server factory 为目标目录创建新的 XcodeApp，迁移 hub 的当前 app，关闭旧 app，并刷新模型、session、MCP 和统计状态。最近工作区列表保存于用户级 `.xcode` 数据目录。

## 7. 事件展示边界

Web 事件序列化器递归处理 dataclass、Pydantic model、enum、Path 和 bytes。浏览器接收的长 transcript 与 final run state 具有长度上限；context-window reset 事件不在 WebSocket 中传输完整 replacement。完整事实仍保留在 session JSONL 账本中。
