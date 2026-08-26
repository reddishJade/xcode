# 会话、恢复与上下文压缩

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
- `compaction`：压缩 replacement、generation、源 entry id 和 surface digest。
- `final`：回答、终止原因、metrics 和 run state。
- `goal_state`：Goal 的条件、暂停状态和重入计数。
- `subagent/descriptor`、`subagent/activation`、`subagent_run`：子代理身份与运行谱系。

实时 text delta、reasoning delta 和 tool update 用于流式展示；稳定事件足以恢复语义历史。

## 3. Surface 与恢复

`SessionSurface` 沿当前 `head_id` 回溯 branch，再应用 inbox claim、assistant、tool use、tool result 和 compaction replacement，生成模型消息。

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

## 5. 自动压缩

压缩触发来源：

- provider usage 的 prompt token 达到模型窗口减去 reserve token 的阈值。
- 配置的 message count 或 token threshold。
- 用户执行 `/compact`。

模型上下文窗口的默认比例为 `compact_trigger_ratio=0.7`；`reserve_tokens` 默认 16384。模型 profile 的 `context_window` 可以覆盖注册表窗口。

`LayeredCompactor` 依次执行：

1. 旧 `read_file` 结果按文件路径保留最近一次。
2. 大型工具结果按 token 压力保留头尾。
3. 较早的工具结果压缩为短提示；最新文件读取和技能激活结果受保护。
4. 非活动 branch 可以转换为 branch summary。
5. 依据 `keep_recent_tokens` 与 `max_recent_messages` 保留近期消息。
6. 切分点对齐 user/assistant turn，工具调用与结果保持同一闭合关系。
7. 使用 LLM 生成结构化摘要；生成失败时使用结构化文本 fallback。
8. 累积读取文件和修改文件，写入摘要的 Critical Context。

摘要包含 Goal、Constraints & Preferences、Progress、Key Decisions、Next Steps 和 Critical Context。摘要中的 frozen identifiers 可以被标记并保留原文。

## 6. 压缩的持久化语义

压缩结果作为新的 `compaction` event 追加，原始账本持续存在。replacement 保存完整当前 surface、generation、source entry ids 和 SHA-256 digest；恢复时从 replacement 加载窗口，再沿账本继续构建。

这让“模型窗口变小”和“运行事实持续累积”同时成立。`ContextManager` 在压缩后增加 context window id、重置动态 context baseline 和实测 prompt token。

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
