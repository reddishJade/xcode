# Web 工作台（浏览器前端）

## 定位

`xcode web` 是 Xcode 的浏览器工作台：单进程 FastAPI 服务 + 零构建的
vanilla JS 单页应用。它与 CLI/TUI 共享同一套装配、会话账本和运行语义，
差异只在于交互投影层——把 harness 的类型化事件流实时渲染成浏览器里的
步骤脊柱、thinking、工具卡片与审批弹窗。

```sh
xcode web              # 默认 http://127.0.0.1:8787
xcode web --open       # 启动后自动打开浏览器
xcode web --port 9000
```

## 进程模型

```text
uvicorn (asyncio 事件循环)
├── REST API       /api/*（会话、工作区、模型、effort、git 分支、统计）
├── WebSocket /ws  唯一的实时通道：事件流 + 控制 + 审批桥接
├── StaticFiles    static/ 三件套（index.html / styles.css / app.js），no-store
└── WebRunHub      持有唯一 XcodeApp
        └── 回合在 run_in_executor 工作线程中消费（ask_stream 同步迭代）
```

关键点：

- **单实例广播**：进程内只有一个 `XcodeApp`（与 TUI 单进程模型一致），
  多个浏览器标签页共享同一事件流。不存在多会话并发。
- **回合不阻塞事件循环**：agent 的 `ask_stream` 是同步迭代器，放到
  `run_in_executor` 中消费，避免预热模型等阻塞 WebSocket 心跳。
  取消经 `CancellationToken` 传入。
- **审批桥接**：工具线程里的 `user_approval_callback` 是同步回调，通过
  `threading.Event` 阻塞等待浏览器端决策，超时 300s 按拒绝处理。

## 后端模块

| 模块 | 职责 |
|---|---|
| `server/serve.py` | `xcode web` 子命令入口：解析 `--port/--open`，装配 `create_app`，启动 uvicorn |
| `server/api.py` | `create_app()` 装配 REST + WebSocket 路由与静态资源；模型发现缓存（10s 超时 / 5min TTL） |
| `server/runner.py` | `WebRunHub`：提交流程、事件广播、会话/工作区切换、审批桥接、`broadcast()` |
| `server/serialize.py` | 事件编码：递归处理 pydantic / dataclass / enum / UUID → JSON-safe dict，超长文本截断 |

`app_factory`（`build_app`）可在服务器运行中重建，用于工作区切换
（`hub.set_app` 替换整个 `XcodeApp`）。

### 复用而非新造

Web 不复制运行语义，全部走 CLI 既有路径：

- 会话恢复：`store.resume()` + `app.restore_session()`（与 TUI `/resume` 一致）；
  `POST /api/sessions` 新建 = `store.clear` + `restore_session`（语义同 TUI `/clear`）。
- 状态栏：复用 REPL 的 `_compute_context_summary` 与 `format_usage_stats`，
  与终端口径逐字节一致（`↑in ↓out R cache CH hit% $cost context% model • effort`）。
- model / effort 选项：`reasoning_effort_levels_for_transport` 与模型网关发现
  （custom transport 跳过发现，只列当前模型 + 前端自定义入口）。
- git 分支：`git for-each-ref refs/heads refs/remotes/origin/` 发现，
  `git switch` 切换（远端缺本地同名分支时自动 `--track`；回合运行中拒绝 409）。

## WebSocket 协议（/ws，JSON 文本帧）

浏览器 → 服务端：

```text
submit   {text, mode}        提交用户消息（approval 等待中时自动先恢复会话）
cancel                       取消当前回合
approval {id, decision, scope} 审批决策
ping                          心跳
```

服务端 → 浏览器：

```text
hello | user_message | run_started | event | run_idle | run_error
run_cancelled | approval_request | session_reset | session_switched
workspace_switched | pong
```

`event` 即 harness 的 `AgentHarnessEvent`（`text_delta` / `reasoning_delta` /
`tool_use` / `tool_result` / `assistant` / `compaction` / `final` 等），按
`step` 号组织渲染，与 TUI 看到的语义同构。

## REST API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/info` | GET | 项目路径、session id、当前分支、模型 / effort / 可用模型 / thinking |
| `/api/stats` | GET | 用量统计与上下文摘要（补丁状态栏） |
| `/api/model` | GET/POST | 读取或切换模型（`{model, effort?}`），运行中拒绝 |
| `/api/sessions` | GET/POST | 列历史会话 / 新建会话 |
| `/api/sessions/resume` | POST | 恢复历史会话并继续（`{id}`） |
| `/api/sessions/{id}` | GET | 单会话回放（`_SESSION_TRANSCRIPT_LIMIT` 截断） |
| `/api/workspaces` | GET/POST | 列工作区 / 切换 `{path}`（近期列表持久化到 `~/.xcode/web_workspaces.json`） |
| `/api/git/branches` | GET/POST | 列本地 + 远端分支 / 切换 `{name}` |
| `/ws` | WS | 实时通道 |

## 前端

`static/` 三件套、无构建步骤、无框架。一个 `app.js`（IIFE）管理全部状态
与 DOM：

- **增量渲染**：文本走 `requestAnimationFrame` 节流；步骤 / 工具卡片增量建
  节点，不做全量重渲。
- **markdown**：内部 60 行迷你渲染器（转义 → 代码块 → 行内 code → 加粗），
  无第三方解析依赖。
- **主题化下拉**：模型 / effort / git 分支 / 工作区四个控件共用 `.drop`
  组件（按钮 + fixed 定位弹出菜单、点击外部关闭、宽度 = 触发按钮、超长省略）。
- **会话视图**：空态（"● 新对话"）与历史列表切换；历史会话点开即恢复继续，
  不是只读预览。
- **主题**：品牌色板映射为 CSS 变量（`--ink/--panel/--amber/--olive/...`），
  全等宽字体，滚动条已主题化；<960px 响应式与 `prefers-reduced-motion` 已处理。
- **缓存**：静态资源 `no-store` 发布（防止浏览器缓存旧 `app.js` 引用已
  删除的 DOM 节点）。

## 开发与验证

```sh
uv run xcode web --port 8791                    # 本地起服务
uv run pytest src/xcode/tests/test_server_web.py -q   # 序列化/API 契约测试
uv run ruff check src/                          # lint
uv run pyright src/xcode/server/                # 类型检查（路由未引用告警可忽略）
node --check src/xcode/server/static/app.js     # JS 语法检查
```

静态前端改动不需要构建步骤：修改 `static/` 后刷新浏览器即可。服务端改动
需要重启（Windows 上找占用端口进程后 taskkill）。

## 已知边界

- 单会话单工作区：服务器持有唯一 `XcodeApp`，不能同时跑两个工作区。
- 工作区切换以目标目录自己的 `xcode.config.json` 为准（CLI `--config`
  不继承）。
- custom transport 下切换模型后旧模型从列表消失，只能通过"＋"入口回来。
- 审批弹窗依赖真实 Act 模式工具调用的 `approval_request` 事件，端到端
  验证需要一次真实触发。
