# TODO

## Worktree — 磁盘泄露与能力缺失

### 1.1 添加 `list_worktrees` 工具
- `WorktreeTaskRunner` 新增 `list()` 方法，扫描 `.local/worktrees/` 目录，用 `git worktree list` 验证有效性
- 注册为只读 ToolSpec `list_worktrees`，返回 id / path / branch / dirty 状态

### 1.2 持久化 worktree 状态
- 将 `self.tasks` 从内存 dict 改为 `.local/worktrees/index.json` 文件
- 进程重启后可恢复已知 worktree 列表，`remove_worktree_task` 能正常工作
- 恢复后运行 `git worktree list` 验证并清理已不存在的条目

### 1.3 安全移除增强
- `_get_cherry_output` 兜底候选列表从硬编码 `("main", "master")` 改为读取 `git remote show origin | grep "HEAD branch"` 或 `git symbolic-ref refs/remotes/origin/HEAD`
- 若无 upstream 且无远程默认分支，返回明确错误而非静默通过

### 1.4 添加 prune 能力
- `remove_worktree_task` 新增 `prune: bool` 参数，清理 worktree 目录和 git 元数据
- 新增 `prune_stale_worktrees` 工具，扫描 index 中标记为已移除但目录残留的条目
- 在 daemon 中注册 `check_stale_worktrees` 定时任务，自动检测并报告

---

## Tasks — 并发安全与工具补全

### 2.1 注册 `claim_task` 工具
- 暴露 `TaskStore.claim()` 为 ToolSpec，使 agent 能原子性地领取任务
- schema: `{"task_id": int}`, `"claimant": string`，返回 claimed 后的 TaskRecord

### 2.2 乐观锁冲突检测
- `TaskStore.update()` 读取时记录 `TaskRecord.version`，写入时检查版本一致
- 版本冲突时抛出 `ConcurrentModificationError`，拒绝覆盖
- `advance_task()` 同样加入版本检查

### 2.3 状态枚举约束
- 将 `UPDATE_TASK_SCHEMA` 的 `additionalProperties: True` 改为 `False`
- `status` 字段添加 `enum: ["pending", "claimed", "completed"]`
- `kanban_view` 对未知 status 单独归类为 `[unknown]` 而非静默丢进 `pending`

### 2.4 `blocked_by` 语义清理
- `_create_task` 和 `_update_task` 统一只接受 `blocked_by`，移除 `dependencies` 别名
- `update_task` 中 `blocked_by` 的处理方式与 `_create_task` 一致（当前 `update_task` 忽略 `blocked_by` 但在 handler 里处理——保持 handler 行为，文档一致即可）

---

## Mailbox — 消息生命周期管理

### 3.1 消息过期与清理
- `LocalFileMailboxTransport` 新增 `retention_days: int = 30` 参数
- `read_unread_messages()` 自动跳过超过保留期的消息
- 新增 `cleanup_expired_messages()` 方法，重写 JSONL 文件剔除过期条目和已 ACK 条目
- 在 daemon 的 `check_mailbox` 任务中定期调用 cleanup

### 3.2 ACK 分离存储
- 将 ACK 从主 JSONL 中分离到 `.local/team/inbox/{agent_id}.ack` 文件（JSONL 格式）
- 减少主 mailbox 文件的膨胀速度
- `acknowledge_message()` 写入 ack 文件；`read_unread_messages()` 合并两个文件计算未读

### 3.3 消息元数据扩展
- 消息 schema 新增可选字段：`thread_id: str`、`priority: str`、`expires_at: str`
- `read_unread_messages()` 支持 `sort_by` 和 `filter_type` 参数
- 不影响现有协议，向后兼容

---

## Progress — 解耦与模型清理

### 4.1 移除硬编码 `claude-progress.txt`
- 将 `save_progress()` 中的路径改为可配置项：`TaskStore` 或新的 `ProgressConfig` 持有 `summary_path`
- 默认值改为 `.local/progress_summary.md`（项目级本地路径）
- 若调用方需要特定路径（如 Claude Code 约定），通过 `build_app()` 传入

### 4.2 分离 orchestration state 与 task payload
- `start_run()` 创建的 `orchestration` 字典保持独立，不混入 `task.payload`
- 在 `TaskStore` 中新增 `get_orchestration(task_id)` / `set_orchestration(task_id, state)` 方法
- 或使用单独的 `.local/orchestration/{task_id}.json` 文件

### 4.3 容错改进
- `resume_run()` 在缺少字段时记录 warning 而非静默使用默认值
- `expire_stale_runs()` 添加 `.local/orchestration/.lease_index` 按过期时间索引，避免全表扫描

---

## Daemon — 实例统一与回声防护

### 5.1 共享 mailbox / task_store 实例
- 移除 `HeartbeatDaemon.__init__()` 中硬编码的 `AgentMailbox(project_root)` 和 `TaskStore(project_root)`
- 改为通过构造函数注入：`__init__(self, ..., mailbox: AgentMailbox, task_store: TaskStore)`
- `build_app()` 在装配阶段将同一实例传递给 daemon

### 5.2 回声防护加固
- `check_mailbox` 的消息过滤从简单的 sender 检查改为 sender + type 组合过滤
- Daemon 自身产生的 `daemon_task_error`/`mailbox_summary`/`git_dirty_alert`/`tasks_summary` 事件添加 `"source": "heartbeat_daemon"` 元数据
- `read_unread_messages()` 支持 `exclude_senders` 和 `exclude_types` 过滤参数

### 5.3 自愈恢复完整性
- `ensure_healthy()` 重启后重新注册 `__init__` 中的 3 个默认任务
- 自定义 `register_task()` 支持持久化注册（写入 `.local/daemon_tasks.json`），重启后自动恢复
- 新增 `list_daemon_tasks()` 方法返回当前注册的任务清单

---

## 跨功能改进

### C.1 共享实例统一注入
- `assembly.py` 的 `build_tool_registry()` 和 `load_opt_in_services()` 创建的所有 `TaskStore`、`AgentMailbox` 实例统一为单例
- 确保 daemon、tasks tools、progress tools、mailbox tools 都使用同一个 `TaskStore` / `AgentMailbox` 实例

### C.2 测试覆盖
- worktree：list / 重启恢复 / prune / 多分支兜底
- tasks：claim 工具 / 乐观锁冲突 / 非法 status 拒绝
- mailbox：过期清理 / ACK 分离 / 大文件性能
- progress：orchestration 分离 / 全表扫描性能
- daemon：实例注入 / 自愈注册恢复 / 回声过滤
