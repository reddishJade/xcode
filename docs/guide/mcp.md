# MCP 服务与工具扩展

Xcode 通过官方 Python MCP SDK 接入 stdio server。MCP 工具注册后与内置工具使用同一个 Agent、ToolGate、参数校验、事件和 session 记录流程。

## 1. 配置

默认配置文件：项目根目录 `.xcode/mcp_config.json`。

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./test.db"],
      "timeout": 30,
      "enabled": true
    },
    "internal-api": {
      "command": "python",
      "args": ["tools/mcp_server.py"],
      "env": {"SERVICE_MODE": "read-only"},
      "defer_loading": true
    }
  }
}
```

每个 server 至少需要非空 `command`。支持 `args`、`env`、`enabled`、`timeout` 和 `defer_loading`；额外配置字段不会进入 server runtime model。

## 2. 启动与发现

普通 server 在构建工具注册表时建立 client、完成 initialize、协商 protocol version 和 capabilities，再读取 tools/list。server 返回的工具变为：

```text
mcp__<sanitized-server-name>__<sanitized-tool-name>
```

例如：`mcp__sqlite__read_query`。

`defer_loading=true` 时优先注册 bootstrap fetch tool：

```text
mcp__internal-api__fetch_tools
```

模型先调用 fetch，再使用 `mcp_tool_search` 查询名称、描述和完整 input schema。有效缓存存在时可以直接使用缓存工具目录。

## 3. 缓存

缓存文件：`.xcode/mcp_cache.json`。缓存 entry 保存：

- server 配置 hash。
- MCP protocol version。
- server name/version。
- tools 列表和 schema。

配置、协议版本或 server identity 变化时，缓存重新获取。无效或损坏的缓存进入重新发现路径。

## 4. Client 生命周期

每个 MCP client 使用独立 async worker。长期 owner task 在同一生命周期内管理 stdio transport、`ClientSession`、initialize、list、call 和 close，避免跨 task 操作 cancel scope。

运行时支持：

- tools/list 分页，最多 100 页。
- `roots` 能力协商后暴露当前 workspace roots。
- tools/list_changed 通知触发工具刷新。
- call progress 转换为 tool update。
- 请求 timeout 与 cancel event。
- 连接失败后的有限重连。
- 关闭时退出 server、transport 和 worker。

## 5. MCP 结果

MCP result 会校验 `isError`、content、structuredContent 和 outputSchema，并转换为 Xcode ToolOutput：

- text → 文本。
- image → ImageContent，保留 MIME 与数据。
- audio → FileContent。
- resource_link、embedded resource → FileContent。
- structuredContent → JSON 摘要和校验状态。
- 未支持 block → 脱敏后的协议错误诊断。

MCP 结果 metadata 保存协议内容、schema、validation 和 protocolErrors；常见凭据在返回宿主前脱敏。

## 6. 命名与冲突

server 名和 tool 名经过安全字符清理。多个 server 产生相同 host tool id 时，冲突涉及的工具整体停用，并写入诊断日志。

工具的原始名称、server 名称、slug 和 host id 由 `McpToolMetadata` 保持关联。

## 7. 查看与重载

```text
/mcp status
/mcp reload
/tool list
```

`/mcp status` 展示 server state、enabled、deferred、tool count、protocol version、server identity 和最近错误。`/mcp reload` 重新读取 `.xcode/mcp_config.json`，更新 runtime registry 并发布新的 MCP 工具快照；重新装配 app 可以让新的工具表完整进入后续 Agent run。

MCP server 的工具调用仍然经过当前模式、静态规则、路径/命令边界和审批策略。涉及网络或外部数据时，同时检查 [security.md](security.md)。
