# Xcode TODO

本文记录下一阶段设计和实现的方向。TODO 中仅保留未完成的计划，按优先级排序。

## 当前边界

- 默认路径继续保持：REPL/CLI -> StructuredAgent -> core tools -> permission/risk/audit -> final answer。
- 默认工具组继续保持 `tools.enabled_groups=["core"]`。
- 新能力必须先证明真实使用场景，默认以 opt-in group 或 `experimental.*` 进入。
- 不做默认启用的 MCP 全量工具注入（防 MCP 接口挤爆 Prompt Cache）。
- 不做不可观测的自动 swarm（多 Agent 必须受控于邮箱总线与物理沙箱）。
- 不做绕过权限系统的外部工具直连。
- 不做面向企业平台的 RBAC、Grafana、Phoenix、RAGAS 等集成。

---

## 待实现（按优先级排序）

### P5：MCP 进阶连接与安全凭据管理 (Advanced MCP & OAuth)

#### P5-1：SSE & WebSocket 通信支持
- **设计**：扩充 `src/xcode/experimental/mcp.py` 底层客户端以支持网络 SSE 与 WS 传输协议，使得 Agent 能与远程/跨主机的 MCP Server 通信。

#### P5-2：OAuth 2.0 (PKCE) 认证与系统 Keyring 密钥存储
- **设计**：对企业级高安全插件集成 OAuth 认证管线，凭据加密存入 OS keychain (Windows 凭据管理器/macOS Keychain)。
