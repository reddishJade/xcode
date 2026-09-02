# 工具与执行调度

工具以 `ToolSpec` 注册，以 JSON Schema 描述输入，以 `AgentTool` 协议进入 Agent 循环。工具结果同时可以携带文本、结构化 metadata 和终端、差异、位置或子代理 render intent。

## 1. 工具总览

| 工具 | 用途 |
| --- | --- |
| `read_file` | 读取文件、目录、图片和带行号的文本片段 |
| `write_file` | 创建文件或执行整文件替换 |
| `edit_file` | 对已有文本执行精确替换 |
| `apply_patch` | 以结构化 patch 批量新增、修改、删除、移动文件 |
| `glob_files` / `find_files` | 按 glob 或文件名发现文件 |
| `grep_search` | 使用正则或 literal 搜索文本 |
| `list_dir` | 列出目录内容 |
| `bash` | 执行平台 Shell 命令 |
| `question` | 向用户收集单选、多选或自由文本 |
| `todowrite` | 替换当前 session 的结构化 todo 清单 |
| `history` | 搜索当前 session 的完整账本并读取邻域 |
| `search_memory` | 检索项目和用户长期记忆 |
| `load_skill` | 加载技能正文或指定 reference |
| `webfetch` / `websearch` | 获取外部网页和搜索结果 |
| `subagent` | 创建独立 session-backed child agent |
| `mcp__*` | 调用 MCP server 提供的工具 |

## 2. 调度模型

Agent 默认使用并行工具模式，`tool_workers` 默认 4。工具可以声明 `sequential` 或 `parallel` 执行模式；当前内置 `ToolSpecAdapter` 默认使用并行调度，写入同一文件时由 file mutation queue 按规范化路径加锁。

调度步骤：

1. 根据模型返回的调用列表分组。
2. 逐调用发出 `tool_execution_start`。
3. 按 JSON Schema 校验参数。
4. 经过 ToolGate 和 PermissionEngine。
5. 启动 handler，接收 progress update。
6. 对超时、取消和异常生成 ToolResult。
7. 按原始 tool call 顺序整理并发结果。
8. 发出 `tool_execution_end`，把结果写入当前上下文。

## 3. 文件读取与编辑

### `read_file`

```json
{"path": "src/xcode/main.py", "offset": 1, "limit": 80}
```

支持文件与目录。文本结果带 `<path>`、`<type>`、`<content>`、1-based 行号和继续读取提示；默认最多 2000 行、50 KB，单行最多 2000 字符。二进制文件返回明确错误，图片按 magic bytes 检测并缩放到最大边 2000 像素，数据保存在 metadata。

### `write_file`

```json
{"path": "notes/plan.md", "content": "# Plan\n..."}
```

适合新文件和有意的整文件替换。返回 unified diff，默认写入上限 1 MB；已有文件的 BOM 和 UTF-8 内容得到保留，Python 文件写入后尝试 Ruff format。

### `edit_file`

```json
{"path": "src/app.py", "old_text": "old", "new_text": "new"}
```

`old_text` 需要精确匹配空白和换行。默认要求唯一匹配；`replace_all=true` 只在单个编辑项时启用。编辑前会保留原始 BOM 和换行风格，并返回差异与首个变化行。

### `apply_patch`

使用 Xcode 结构化格式：

```text
*** Begin Patch
*** Update File: src/app.py
@@
-old line
+new line
*** End Patch
```

支持 `*** Add File`、`*** Update File`、`*** Delete File`、`*** Move to`。系统先解析所有 hunk、路径和上下文，再创建变更计划；上下文无法匹配时返回 verification error。

## 4. 搜索

ripgrep 可用时优先使用，Python fallback 提供同一工具接口。搜索层处理 `.gitignore`、隐藏目录、`.git`、`.venv`、`__pycache__`、环境文件和二进制路径；结果具有数量、长行和总字节限制。

- `glob_files`：模式匹配当前目录下的项目文件。
- `find_files`：basename 模式自动递归。
- `list_dir`：按名称列出目录，目录带 `/` 后缀。
- `grep_search`：支持 regex、literal、ignore_case、glob 和 context。

## 5. Bash 与输出

`bash` 的输入包含 `command`、`workdir` 和 `timeout_ms`。命令自身默认超时 30 秒，最大 300 秒；Agent 等待工具结果的默认上限为 120 秒。进程以 argv 启动，stdout/stderr 独立排空，POSIX 使用进程组，Windows 使用隐藏窗口。

OutputAccumulator 保留行数、字节数和最近预览。输出超过限制时写入临时 `.log`，模型收到尾部摘要和完整输出路径；`on_progress` 将实时输出转为 tool update。Shell 的安全分类和 OS sandbox 位于 [security.md](security.md)。

## 6. 结果与上下文

工具结果的 `is_error` 驱动 watchdog 和前端状态。文件工具可以提供 `LocationRenderIntent`，写入和 patch 可以提供 `DiffRenderIntent`，Shell 提供 `TerminalRenderIntent`，子代理提供 `SubagentRenderIntent`。

工具调用和结果进入 session surface；大型结果在 provider 请求前按 request hygiene 和当前工作窗口预算生成较小投影。原始 session 账本保持可恢复结构。
