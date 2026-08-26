# 长期记忆

Xcode 的长期记忆保存跨 session 可复用的规则、架构决策、验证事实和解决方案。当前任务连续性由 session surface 与账本承担。

## 1. 两个记忆层

| 层 | 默认文件 | 适合保存 |
| --- | --- | --- |
| project | `<project>/MEMORY.md` | 项目架构约定、技术选择、团队规则 |
| user | `~/.xcode/memory/MEMORY.md` | 个人偏好、跨项目习惯、通用工作方式 |

每个 Markdown H2 section 形成一条 `MemoryRecord`。记录包含 title、body、layer 和由 layer/title 生成的稳定 id。旧格式的 metadata 行会被解析并从检索正文中剥离；退休状态记录不会进入结果。

## 2. 按需检索

Agent 拥有 `search_memory` 工具：

```json
{
  "query": "provider timeout",
  "scope": "providers",
  "layer": "all",
  "limit": 3
}
```

检索使用确定性 BM25：

- 英文、数字、代码、路径和中文 token 参与匹配。
- 中文文本额外生成字符和双字 token。
- exact match 与 token overlap 提升排序分数。
- `project` 在相同分数条件下排在 `user` 前面。
- 单次结果最多 10 条。

索引按记忆文件的 path、inode、mtime 和 size 建立签名；文件变化后自动重建。

普通 turn 只接收长期记忆使用协议，模型在需要时调用 `search_memory`。恢复旧 session 时可以读取最多 6000 token 的 memory overview，并将其作为背景上下文。

## 3. 写入与维护

REPL 命令：

```text
/memory list
/memory list project
/memory search provider timeout
/memory add project Retry policy | Provider requests retry transient failures.
/memory update project Retry policy | Retry transient provider failures twice.
/memory delete project Retry policy
```

`/memory add` 也支持 `title: body` 或两个空格分隔字段的简写。项目和用户层通过命令中的可选前缀选择。

写入规则：

- 输入需要一个 H2 标题和至少三字符正文。
- 标题重复或正文重复时拒绝写入。
- update 按标题替换并保留其他 section。
- delete 按标题删除并保留文件其他内容。
- 文件锁保护并发修改。
- 临时文件、flush、fsync 和 replace 提供原子更新。
- 成功写入后清除内存检索索引。

## 4. 记忆与上下文的关系

记忆检索结果通过普通工具结果进入当前 session，可以参与后续压缩和恢复。记忆文件本身作为可读文件拥有来源路径；memory overview 以 `[project memory · memory_id]` 或 `[user memory · memory_id]` 标识层和记录身份。

长期记忆适合稳定事实，todo、当前 diff、临时错误和当前回合进度属于 session context。保存记忆时使用具体、可复用、可验证的描述，并保留项目层与用户层的边界。
