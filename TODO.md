# TODO

## MCP — Coding Agent 必要能力

当前基线：使用官方 Python SDK 连接本地 stdio server，支持初始化协商、
`tools/list`、`tools/call`、分页、`tools/listChanged`、schema cache、延迟加载、
超时取消和有限重连。后续只补 coding agent 的实际运行缺口，不追求完整 MCP
协议覆盖。

### M.1 暴露 workspace roots
- 通过 SDK 的 roots callback 向 server 暴露当前项目根目录
- 仅包含宿主权限边界允许读取的 workspace；不得把用户主目录或任意磁盘根目录
  默认暴露给 server
- workspace 发生明确切换时发送 roots changed notification
- 添加 roots 请求、空 roots 和越界目录过滤的协议测试

### M.2 MCP 运行时状态与手动重载
- 增加只读状态接口，展示 server 的 configured / deferred / connected /
  failed 状态、协商协议版本、server identity、工具数量和脱敏后的最近错误
- 支持手动重新读取 `.local/mcp_config.json` 并重建对应动态工具；配置变化不要求
  自动文件监听
- 重载必须关闭被删除或被替换的 stdio session，清理失效 schema cache，且不得
  影响未变化 server
- 为 REPL 增加最小 `/mcp status` 和 `/mcp reload` 入口，不扩展为通用 MCP
  管理控制台

### M.3 长工具调用的进度与取消
- 为 `tools/call` 传递 progress token，并消费 SDK progress notification
- 将进度映射为现有结构化事件或诊断状态，避免长时间调用只能等待最终结果
- 用户取消、agent run 取消和 timeout 使用同一取消路径；确认 server 不响应取消
  时仍能回收子进程
- 不为 MCP 单独建立第二套事件总线或后台任务系统

### M.4 工具元数据覆盖
- 实现 `.local/mcp_config.json` 中当前被跳过的 `overrides`
- 仅允许宿主侧覆盖 `risk`、`read_only`、`concurrency_safe`、启用状态和展示说明
- server annotations 只作为提示；权限和并发属性最终由宿主配置与保守默认值决定
- 配置按 `server + tool` 精确匹配，未知工具名给出诊断，不静默忽略

### M.5 真实 server 兼容回归
- 固定一个官方 `modelcontextprotocol/servers` 示例作为 stdio conformance smoke
  test，覆盖发现、调用、structured content、list changed 和关闭生命周期
- 单元测试继续围绕 SDK adapter 的可观察行为，不断言 SDK 私有实现细节
- 外部 server 测试与默认离线 pytest 分离，避免网络或 npm 环境成为本地测试前提
- 记录启动失败、协议不兼容、schema 非法、调用超时和异常退出的稳定诊断格式

### MCP 暂不实现
- Streamable HTTP / SSE、OAuth 和远程 server 凭据管理：出现必须使用的真实
  coding server 后再做
- resources / prompts：当前工具调用已覆盖 coding agent 主路径；先不增加第二套
  发现、注入和权限语义
- sampling / elicitation：会形成 server 反向驱动模型或用户交互的新信任边界
- 通用连接池、自动故障转移、周期健康轮询和 MCP marketplace：当前规模没有收益
- 为追求协议覆盖而透传所有 notification：只消费会影响工具目录、进度、取消和
  workspace 边界的事件

---

## Tasks — 剩余一致性问题

### T.1 `advance_task` 乐观锁
- 为 `advance_task` schema 增加 `expected_version`
- 完成主任务前检查当前版本，冲突时返回与 `update_task` 一致的可重试诊断
- 下游依赖解除仍在同一目录锁内完成，不为每个下游任务要求调用方提供版本

### T.2 `blocked_by` 显式清空
- `update_task(blocked_by=[])` 应清除已有依赖；当前 truthy 判断无法区分“未传入”
  与“显式传空数组”
- handler 使用键存在性判断，并保留 schema 的 `additionalProperties: false`

---

## Progress — 默认输出路径

### P.1 移除函数级 `claude-progress.txt` 回退
- `paths.progress_summary` 默认设为 `.local/progress_summary.md`，并补充到
  `CONFIG.md`
- `save_progress()` 不再自行选择 Claude Code 专用文件名；路径由装配层传入
- 直接调用底层函数时若没有路径，使用项目级 `.local/progress_summary.md`

---

## Daemon — 持久自定义任务语义

### D.1 明确恢复边界
- 当前 `.local/daemon_tasks.json` 只能恢复自定义任务名称，不能恢复进程内
  callable；文档和状态接口必须明确显示 `callable pending`
- 只有出现真实跨进程自定义任务需求时，才设计可序列化的任务类型注册表；不持久化
  Python callable，也不引入动态代码加载

---

## Config — 合并后强类型校验

### C.1 使用 Pydantic 校验最终配置
- 保留当前全局、项目、本地配置的 raw dict 深度合并语义，在所有来源合并及环境
  变量覆盖后执行一次统一模型校验
- 使用 Pydantic 模型替代 `_config_from_dict()` 的手写递归转换，拒绝未知字段、
  错误标量类型、非法 enum 和错误的嵌套结构
- 错误信息必须包含完整字段路径和配置来源提示，避免只暴露底层 validation error
- 不引入 `pydantic-settings` 或第二套配置发现机制；Pydantic 只负责最终数据模型
  与验证
- 迁移时保持现有显式键覆盖、profile 继承、hook source 标注和环境变量优先级

---

## Experimental State — SQLite 迁移评估门槛

### S.1 达到规模或并发瓶颈后再统一存储
- 当前继续使用 JSON / JSONL + `filelock`；不立即迁移
- 当出现以下任一真实证据时，评估使用标准库 `sqlite3` 统一 tasks、mailbox 和
  orchestration：
  - mailbox 或 task 扫描、清理、索引维护产生可测量性能瓶颈
  - 多进程写入冲突或原子更新逻辑继续扩张
  - ACK、lease、版本和查询索引需要跨多个旁路文件保持事务一致性
  - 状态迁移与损坏恢复成本明显高于单文件存储的可读性收益
- 评估时先设计最小 schema、事务边界、WAL/锁策略和现有文件的一次性迁移；
  session transcript 与 MCP schema cache 不在首轮迁移范围
- 没有 benchmark、故障案例或真实并发需求时，不为“统一技术栈”引入数据库
