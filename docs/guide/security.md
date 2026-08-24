# 权限、安全与 Linux 沙箱

Xcode 构建了一套多层次的纵深防御安全体系（Defense-in-depth），兼顾自动化开发的高效体验与系统环境的安全性。

---

## 1. 纵深防御架构

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 策略判定层：PermissionEngine & Execution Modes           │
│    - 路径白名单与禁访目录 (restricted_dirs)                  │
│    - findLast 规则引擎精确匹配工具、命令与参数                │
├─────────────────────────────────────────────────────────────┤
│ 2. 审批路由层：Approval Router                              │
│    - Build 模式：独立 Reviewer 模型自动化语义风控             │
│    - Act 模式：用户人工审批 (HITL)                           │
├─────────────────────────────────────────────────────────────┤
│ 3. 进程隔离层：Linux Bubblewrap OS 沙箱                     │
│    - 宿主系统只读挂载，项目根与 /tmp 可写                    │
│    - .git / .xcode / 密钥文件 / 敏感凭据全面遮蔽             │
│    - 独立 Network Namespace 隔离网络访问                    │
├─────────────────────────────────────────────────────────────┤
│ 4. 审计脱敏层：JsonlAuditLogger                             │
│    - 全量记录操作日志并对敏感 Key 进行脱敏                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Linux Bubblewrap OS 沙箱

在 Linux 平台上，Agent 调用的 `bash` 命令默认被封装在基于 [bubblewrap](https://github.com/containers/bubblewrap)（`bwrap`）的独立隔离环境中运行。

### 沙箱隔离特性
1. **文件系统只读保护**：宿主系统的根目录（`/usr`, `/bin`, `/lib`, `/etc` 等）挂载为只读，阻止任何对系统全局文件和包管理器的非授权篡改；
2. **工作区可写边界**：仅当前项目根目录以及 `/tmp` 挂载为可写；
3. **版本控制与运行时保护**：项目内的 `.git`、`.xcode`、`.agents` 目录自动重新挂载为只读，禁止 Agent 擅自修改 Git 提交历史或篡改 Xcode 运行时数据；
4. **敏感凭据遮蔽 (Masking)**：`~/.ssh`、`~/.aws`、`~/.kube`、`~/.gnupg` 等已知密钥目录以及项目内的 `.env*` 敏感环境变量文件会被自动遮蔽为空，防止凭据泄露；
5. **网络命名空间隔离**：默认启用独立网络命名空间（`network_access: deny`），阻止恶意或意外的网络请求。

### 沙箱模式配置
在 `xcode.config.json` 中配置：

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

* `mode` 可选值：
  * `workspace-write`（默认）：项目与 `/tmp` 可写，宿主其余路径只读；
  * `read-only`：全部文件系统只读；
  * `danger-full-access`：允许以当前用户权限读写宿主路径。

---

## 3. 自动审批与 Reviewer 模型

在 Build 模式下，当 Agent 触发 Shell 命令时，Xcode 会将上下文与命令参数提交给独立的 `reviewer` Profile（轻量模型）：

* **评估维度**：
  * 操作的风险等级（Low / Medium / High / Critical）；
  * 用户意图的授权充分性；
  * 操作的可逆性与影响范围。
* **判定逻辑**：
  * 运行本地单元测试、读取依赖等低风险操作：秒级自动放行（Auto-approved）；
  * 删除核心文件、跨工作区写入或高风险命令：安全拒绝并转交用户人工确认。

---

## 4. 外部目录白名单 (external_directories)

若需要允许 Agent 读取或写入项目外部的公共目录（如共享模板或外部文档），可配置白名单：

```json
{
  "security": {
    "external_directories": [
      {"path": "/shared/docs", "access": "read"},
      {"path": "/tmp/build-cache", "access": "read_write"}
    ]
  }
}
```

---

## 5. 审计日志 (Audit Logging)

所有工具调用的决策、参数、执行耗时与脱敏后的输出均会写入审计日志文件（通过 `observability.audit_path` 配置）：

```json
{"timestamp": "2026-08-24T15:30:00Z", "action": "bash", "command": "pytest", "decision": "allow", "router": "auto_reviewer", "sandbox": true}
```

---

← **上一篇**：[核心工具箱与并发调度 (tools.md)](tools.md) | **下一篇**：[配置系统与设置浏览器 (configuration.md)](configuration.md) →

