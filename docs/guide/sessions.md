# 会话、恢复与上下文换窗

Xcode 将 session 组织为可追加的 JSONL 事实账本，并从当前 branch 投影出模型历史、界面历史和运行状态。

## 1. Session 文件

默认目录：

```text
.xcode/
├── sessions/
│   └── session-<timestamp>.jsonl
├── session_index.json
├── snapshots/<session-id>/
└── approval_grants.json
```

每个 entry 包含 `id`、`parent_id`、`type`、`content` 和 `created_at`。`session_index.json` 保存 title、summary、project path、updated time 和 `head_id`。追加操作由 file lock 保护。

## 2. 稳定事件

运行时主要记录：

- `inbox/inserted`、`inbox/claimed`、`inbox/discarded`：输入生命周期。
- `provider_request`：实际 wire messages、工具定义、provider、options 和 context trace。
- `assistant`、`tool_use`、`tool_result`：模型与工具语义事件。
- `context_window_reset`：新窗口 ID、触发原因、replacement、generation、源 entry id 和 surface digest。
- `final`：回答、终止原因、metrics 和 run state。
- `goal_state`：Goal 的条件、暂停状态和重入计数。
- `subagent/descriptor`、`subagent/activation`、`subagent_run`：子代理身份与运行谱系。

实时 text delta、reasoning delta 和 tool update 用于流式展示；稳定事件足以恢复语义历史。

## 3. Surface 与恢复

`SessionSurface` 沿当前 `head_id` 回溯 branch，再应用 inbox claim、assistant、tool use、tool result 和 context-window replacement，生成模型消息。

消息使用显式 `kind`/`payload` 标签编码。恢复时校验消息结构与工具配对：每个 tool call 对应一个 result，orphan result、重复 id 和未闭合调用形成恢复错误。

`replay_session` 同时恢复：

- Agent message history。
- 当前执行模式、Goal 和 todo。
- 已激活技能。
- active files 与工具结果上下文。
- session id、history session id 和恢复提示。

## 4. 输入 lane 与运行控制

`SessionInbox` 使用两条输入 lane：

| lane | 消费时机 |
| --- | --- |
| `NEXT_STEP` | 当前 active run 的下一次模型请求前 |
| `NEXT_TURN` | 当前 run 完成后启动新的 run |

`SessionRunController` 为每个 session 维护一个 active run。普通输入、`/steer`、follow-up、runtime reminder 和 interrupt 都先写入 inbox，再由 run 在定义好的边界 claim。

`ActiveRunHandle` 的状态为 running、cancelling、finishing、finished。生成结束前会关闭 step input，再 claim 最后一批输入，保证输入和结束事件顺序稳定。

## 5. 上下文换窗

换窗触发来源：

- provider usage 的 prompt token 达到当前模型窗口预算。
- provider 未返回 usage 时的本地 token 估算。
- 配置的 message count 或绝对 token threshold。
- 模型调用 `new_context`，或用户执行 `/new-context`。

窗口大小优先取 provider profile 的 `context_window` 覆盖；未覆盖时读取当前模型注册值。默认触发线为窗口的 95%，并且不得高于“窗口 - `reserve_tokens`”。

`ContextWindowRollover` 不生成摘要。它重新注入启动上下文，保留已激活 skill 和当前 user 回合，并仅在新工作 surface 中裁剪过期文件读取与大工具输出。项目根 `NOTE.md` 保存执行前沿；模型主动换窗前必须先写入非空 `NOTE.md`。

## 6. 换窗的持久化语义

换窗结果作为新的 `context_window_reset` event 追加，原始账本不改写。replacement 保存完整当前 surface、generation、source entry ids 和 SHA-256 digest；恢复时加载最新 replacement，再沿账本继续构建。

`history` 工具可列出窗口边界、搜索当前 branch、分页读取某条原始记录，或查看其邻近记录。因此模型的当前 context 是可丢弃工作集，session transcript 才是可检索的无损事实源。

## 7. 分支与回退

```text
/fork       从当前 branch 的用户消息选择起点并创建新 session
/clone      复制当前 session
/tree       浏览 entry tree 并把 head 移到选中 entry
/rewind 3   将 head 回退 3 个用户 turn
/continue   恢复当前项目最近的有意义 session
/resume     选择历史 session
```

原 session 的 entry 保持不变；新分支通过新的文件或新的 head 投影工作历史。`/tree` 和恢复后会重新加载 Agent 与界面状态。

## 8. 文件快照与 `/undo`

Git 工程的每个用户 turn 可以建立 pre/post snapshot。快照使用 `.xcode/snapshots/<session-id>` 下的隐藏 Git tree，记录修改、创建和删除文件，并排除环境密钥、生成目录和大型文件。

```text
/undo --list   查看快照记录
/undo          回退最近一个可撤销 turn
/undo 3        回退最近三个可撤销 turn
```

回退时校验路径、turn changed files、post snapshot 冲突和 PermissionEngine。冲突文件进入 skipped，已被用户继续修改的内容得到保留。文件恢复与 session `/rewind` 是两条独立操作。

## 9. 历史检索

`history` 工具在当前 branch 的原始 JSONL entry 上执行关键词 search，或读取指定 message id 附近的记录。它用于恢复后找回已经被 surface replacement 压缩的精确细节。
