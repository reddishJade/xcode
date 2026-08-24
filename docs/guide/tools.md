# 核心工具箱与并发调度

Xcode 内置了一套专为编程场景调优的工具集，并实现了**副作用感知的工具并发分区调度机制**。

---

## 1. 内置核心工具概览

| 工具名称 | 权限属性 | 功能说明 |
|---|---|---|
| `read_file` | 只读 / 并发安全 | 读取指定文件内容，支持行号范围切片 |
| `write_file` | 写入 / 串行屏障 | 创建或覆盖写入文件内容 |
| `edit_file` | 编辑 / 串行屏障 | 基于 SHA256 指纹匹配替换文件代码块（Read-before-edit） |
| `apply_patch` | 编辑 / 串行屏障 | 应用标准 Unified Diff 格式的 Patch |
| `glob_files` | 只读 / 并发安全 | 快速通过 Glob 模式搜索匹配的文件名 |
| `grep_search` | 只读 / 并发安全 | 基于 ripgrep 极速进行代码文本与正则搜索 |
| `list_dir` | 只读 / 并发安全 | 列出目录下的文件和子目录层级结构 |
| `find_files` | 只读 / 并发安全 | 根据名称和文件类型递归搜索文件 |
| `bash` | 命令 / 串行屏障 | 在沙箱或本地环境中执行 Shell 命令 |
| `todowrite` | 会话 / 串行屏障 | 维护会话内的 Todo 待办事项列表 |
| `search_memory`| 只读 / 并发安全 | 基于 BM25 检索项目与用户长期记忆 |
| `subagent` | 委派 / 独立会话 | 启动独立的子代理执行复杂子任务 |
| `websearch` | 网络 / 并发安全 | 通过网络搜索引擎检索外部最新文档 |
| `webfetch` | 网络 / 并发安全 | 抓取网页并转换为 Clean Markdown 格式 |
| `question` | 交互 / 暂停等待 | 向用户发起交互式单选/多选确认 |
| `history` | 只读 / 并发安全 | 检索当前会话的历史原始消息邻域 |

---

## 2. 副作用感知与工具并发分区调度

当大模型单次返回多个工具调用指令时（例如同时读取 5 个依赖文件），Xcode 的调度器会根据工具的副作用属性进行智能分区：

```
Model Tool Calls Batch
  ├── [read_file A] ──┐
  ├── [read_file B] ──┼─► [Concurrent Worker Pool] ──► 并行执行，延迟降低 60%+
  ├── [grep_search] ──┘
  │
  ├── [edit_file C] ────► [Serial Execution Barrier] ──► 串行保证写入原子性
  └── [bash pytest] ────► [Serial Execution Barrier] ──► 串行验证结果
```

* **并行调度**：只读且并发安全的工具会被分发至内部 Worker 池并行执行，大幅缩短多文件探索与上下文收集阶段的耗时。
* **串行屏障**：涉及文件写入或 Shell 命令的工具保持严格串行执行，确保前序步骤的副作用对后序步骤完全可见。

---

## 3. Read-before-edit SHA256 指纹校验

为了防止大模型基于过时的文件内容进行盲目编辑或并发导致代码被破坏，`edit_file` 强制执行指纹校验：

1. Agent 在编辑文件前必须先通过 `read_file` 读取文件；
2. 调度器在内存中记录该文件读取时的 SHA256 校验和；
3. 执行 `edit_file` 时，Xcode 会重新校验磁盘文件的当前 SHA256；若文件在读取后被用户或外部进程修改，则直接拒绝写入（Fail-safe），防止覆盖最新代码。

---

## 4. 会话 Todo 清单管理 (todowrite)

Agent 通过 `todowrite` 维护结构化的任务分解：

```json
{
  "todos": [
    {"id": "1", "content": "设计数据模型", "status": "completed", "priority": "high"},
    {"id": "2", "content": "编写数据库迁移脚本", "status": "in_progress", "priority": "high"},
    {"id": "3", "content": "添加单元测试", "status": "pending", "priority": "medium"}
  ]
}
```

* **状态持久化**：Todo 列表会自动渲染至 TUI 侧边栏与动态上下文中，即便发生上下文压缩也不会丢失当前任务进度。

---

← **上一篇**：[会话账本与分层压缩 (sessions.md)](sessions.md) | **下一篇**：[权限、安全与 Linux 沙箱 (security.md)](security.md) →

