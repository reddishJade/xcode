# Xcode Harness — 应用运行时

Harness 是 Xcode 最外层的应用装配与运行时层，把 Agent 循环、LLM provider、工具系统、安全策略和观测基础设施组合为可运行的应用。

## 目录职责

| 目录/文件 | 职责 |
|---|---|
| `app.py` / `assembly/` | 读取配置并装配组件 |
| `agent_runtime/` | `AgentHarness` / `CodingAgentHarness`、压缩、tool gate、prompt |
| `config.py` | 运行时配置模型 |
| `session/` / `session_todo.py` | 会话树与 TODO |
| `skills/` / `skill_activation.py` | skill 发现、注册、激活 |
| `mcp/` | MCP 客户端与动态工具 |
| `memory/` | MEMORY.md 检索与写入 |
| `execution_env/` | 文件系统与子进程执行环境 |
| `snapshot.py` | undo 快照 |
| `security/` | 权限模型、PermissionEngine、规则匹配、shell 语义分析 |
| `observability/` | 审计、关联 ID、hooks / external hooks |

`CodingAgentHarness` 位于 `agent_runtime/`，是 harness 对 agent loop 的编码领域封装——执行模式（plan/build/act）、上下文压缩、重复调用检测、subagent 等。基类 `AgentHarness` 提供领域无关运行时核心。本层依赖 `agent` 与 `ai`，但不被它们反向依赖。

## 导入约定

```python
# 权限 / 安全
from xcode.harness.security import PermissionEngine, PermissionPolicy, FileGrantStore
from xcode.harness.security.permission_model import Rule, GrantStore

# 审计 / hooks
from xcode.harness.observability import HookManager, AuditLogger, redact_text
```

约束：`security` 不依赖 `observability` 实现细节；`observability` 不 re-export 权限类型。
