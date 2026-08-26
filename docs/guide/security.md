# 权限、安全与 Linux Sandbox

Xcode 将语义权限、用户审批、自动 reviewer、路径边界、Shell 分析和 OS sandbox 组合为多层执行边界。工具调用在实际 handler 运行前完成决策。

## 1. 决策链

```text
Tool call
  → ActionExtractor
  → restricted_dirs / path boundary / dangerous command
  → mode + static policy + shell policy + hook constraints
  → RuleMatcher
  → PermissionResolver
  → saved grants
  → approval policy
  → user or auto reviewer
  → ToolSpecAdapter
  → handler
```

`Action` 描述工具、capability、operation、targets 和 unresolved effects。目标可以是 path、command、domain、mcp、skill 或 subagent；`apply_patch` 可以从 patch 内容提取全部文件目标。

`PermissionResolver` 以 deny、ask、allow 顺序选取最严格 constraint。每次结果包含 decision、blocked、reason、reason code、remediation、matched rule、source、metadata、approval result 和 action。

## 2. 默认模式策略

- **Plan**：规则 fallback 为 deny；写入范围限定为 `.xcode/plans/*.md`。
- **Build**：项目结构化读写默认 allow；Shell 与规则覆盖范围外动作默认 ask，由 auto reviewer 处理。
- **Act**：读取默认 allow；写入和 Shell 默认 ask，由用户处理。

静态 `security.permissions` 与 `security.tools` 会进入同一 PermissionEngine。具体工具规则追加在权限名展开的规则之后。

## 3. 路径边界

项目 root 通过 `resolve()` 和 `relative_to()` 校验；符号链接路径逐段检查。以下目标进入内置拒绝策略：

- `.git` metadata。
- `.venv`、`__pycache__`。
- `.env`、`.env.*`。
- `.aws`、`.azure`、`.docker`、`.gnupg`、`.kube`、`.netrc`、`.npmrc`、`.pypirc`、`.ssh`。
- `id_rsa`、`id_ed25519`、`id_ecdsa`、`id_dsa` 等密钥文件。
- `restricted_dirs` 中的路径，以及无法安全提取目标路径的受限动作。

项目外路径需要 `external_directories` 中的目录覆盖和对应 access：`read`、`write` 或 `read_write`。`non_workspace_access=false` 时，用户外部目录配置停止生效；Xcode 自身的 `~/.xcode` 和 `~/.agents` 保留只读基础设施访问。

`.env.example` 可以读取，写入仍进入敏感路径策略。`sensitive_path_overrides` 只接受精确环境文件路径，凭据目录和密钥文件持续拒绝。

## 4. Shell 分析

Shell analyzer 对 POSIX、PowerShell 和 cmd 提供分类器。分类对象包括：

- 只读命令和纯观察命令。
- 写入、复制、移动和删除命令。
- literal 文件路径及其 access。
- 管道、重定向、变量、glob、命令替换和组合语法。
- 解析错误、未知命令和待确认 wrapper。

以下命令直接拒绝：`rm -rf /`、主机级关机/重启、`sudo`/`su` 权限提升、`git reset --hard`、强制 `git clean` 等。未知和动态效果按照当前 mode 的 unresolved policy 进入 ask 或 deny。

## 5. 审批与 Grant

审批回调收到工具、参数、原因、工作目录、turn id、transcript 和实际允许的 scope：

| scope | 作用 |
| --- | --- |
| `once` | 当前一次执行 |
| `session` | 当前 session 的匹配目标 |
| `permanent` | 项目级持久授权 |

多 target 动作只允许 `once`。auto reviewer 只允许 `once`，不能建立 session 或 permanent grant。session grant 存在内存 session store；permanent grant 写入 `.xcode/approval_grants.json`，使用文件锁和原子替换。

`approval_policy`：

- `on-request`：ask 进入配置的 reviewer。
- `never`：显式 grant 仍可命中，其余 ask 形成确定性 deny。

`approval_router`：`mode` 让 Build 使用 auto reviewer、Act 使用 user；`user` 固定人工；`auto` 固定 reviewer。

## 6. 自动 reviewer

Build reviewer 使用独立 provider 会话，输入包含有界 transcript 和精确 action。system/user 内容作为授权证据；assistant、tool call、tool result、approval reason 和 planned arguments作为待审查证据。

reviewer 返回单个 JSON assessment：outcome、risk_level、user_authorization、rationale。低/中风险的有界动作可以 allow；高风险需要足够授权；critical 风险 deny。provider failure、超时和非法 JSON 进入 failed-closed 结果。

## 7. Linux bubblewrap

默认 Linux sandbox：

```json
{
  "security": {
    "sandbox": {
      "mode": "workspace-write",
      "network_access": "deny"
    }
  }
}
```

模式：

- `read-only`：root filesystem 以只读方式提供。
- `workspace-write`：项目 root、临时目录和获准可写外部目录可写。
- `danger-full-access`：使用当前用户的完整文件访问；network allow 时使用本地 subprocess shell。

bubblewrap 设置 user、PID、IPC、UTS namespace，network deny 时设置 network namespace，进程 drop 全部 capability。`.git`、`.agents`、`.xcode` 以只读路径保护；凭据和环境文件通过 `/dev/null` 或 tmpfs 遮蔽。项目 root 之外的 cwd 直接拒绝。

Linux 之外使用本地 SubprocessShell，语义 PermissionEngine 继续保护工具调用。

## 8. 审计与脱敏

配置 `observability.audit_path` 后，工具审计记录包含工具、动态决策、policy decision、最终状态、脱敏输入输出、审批范围、grant id、reviewer、风险、授权等级、rationale 和 correlation。

常见 `sk-`、API key、secret、token、password 形式在工具、MCP、外部 hook 和审计边界执行脱敏。详细 hook 生命周期位于 [hooks.md](hooks.md)。
