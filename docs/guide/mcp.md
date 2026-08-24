# MCP (Model Context Protocol) 扩展

Xcode 深度集成了开放的 **Model Context Protocol (MCP)** 标准，允许开发者无缝将本地数据库、Git 仓库、Issue 追踪器或私有 API 工具接入 Agent 的上下文与能力集。

---

## 1. 配置 MCP 服务器

在项目根目录下创建 `.xcode/mcp_config.json`（或在全局 `~/.xcode/mcp_config.json` 中定义）：

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./test.db"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

---

## 2. 动态工具发现与命名空间

当 Xcode 启动时，`McpManager` 会自动通过 stdio 启动配置的 MCP 服务器并进行能力握手（Handshake）：

1. **命名空间隔离**：MCP 工具会自动注册为 `mcp__{server}__{tool}` 格式，例如：
   * `mcp__sqlite__read_query`
   * `mcp__sqlite__describe_table`
   * `mcp__github__create_issue`
2. **Schema 缓存与校验**：Xcode 将协商好的 Tool Schemas 缓存于 `.xcode/mcp_cache.json` 中，仅在配置文件发生变化时重新协商，启动速度毫秒级；
3. **权限与审计覆盖**：所有 MCP 工具均受 Xcode `PermissionEngine` 与模式规则引擎统一管辖。

---

## 3. 在会话中查看与调用 MCP 工具

在 REPL / TUI 中：

```text
# 查看当前所有已连接的 MCP 服务器与工具状态
/mcp status

# 热重载 MCP 配置文件与服务器连接
/mcp reload

# 显式查看已注册的特定 MCP 工具定义
/tool list mcp__sqlite
```

Agent 在理解任务时会自动识别并调用相关的 MCP 工具。例如直接向 Agent 提问：
> “查询 SQLite 中用户表的 Schema，并分析与新迁移文件的差异”

Agent 会自动调用 `mcp__sqlite__describe_table` 获取真实数据库元数据并进行分析。

---

← **上一篇**：[配置系统与设置浏览器 (configuration.md)](configuration.md) | **下一篇**：[Skills 技能系统 (skills.md)](skills.md) →

