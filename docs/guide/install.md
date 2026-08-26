# 安装与环境准备

Xcode 需要 Python 3.12 或更高版本。运行时依赖 OpenAI-compatible provider、终端交互、文件处理、MCP、FastAPI Web 服务和 Linux bubblewrap sandbox。

## 1. 前置条件

- **Python**：3.12+。
- **包管理器**：推荐 [uv](https://docs.astral.sh/uv/)，pip 也可用。
- **Git**：项目内文件快照、`/undo` 和 Git 上下文需要 Git 工程。
- **Linux sandbox**：Linux 默认使用 `bwrap`。需要时安装：

  ```bash
  # Debian / Ubuntu
  sudo apt-get update && sudo apt-get install -y bubblewrap

  # Fedora / RHEL
  sudo dnf install -y bubblewrap

  # Arch Linux
  sudo pacman -S bubblewrap
  ```

  Linux 中使用默认 `workspace-write` sandbox 时，缺少 `bwrap` 会让 sandbox 初始化失败。可以安装 bubblewrap；在明确理解风险后，也可以同时配置 `danger-full-access` 与 `network_access: allow`，让运行时使用本地 subprocess shell。

Linux 以外的环境使用本地 `SubprocessShell`；工具权限和路径策略仍然生效。

## 2. 开发模式安装

```bash
git clone https://github.com/reddishJade/xcode.git
cd xcode

uv venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

uv pip install -e .
```

开发依赖：

```bash
uv pip install -e ".[dev]"
```

也可以直接使用：

```bash
uv run xcode --help
```

## 3. 首次配置

交互式向导：

```bash
xcode setup
```

向导会收集 provider、API key、base URL、模型、thinking 和可用的 reasoning effort，并把配置写入项目根目录的 `xcode.config.json`。取消保存时，当前进程可以使用临时配置继续运行。

也可以使用环境变量。常用 key 包括：

```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export MIMO_API_KEY="..."
export CHATGLM_API_KEY="..."
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY = "..."
```

API key 也可以写入 provider profile。敏感配置适合放在个人配置或环境变量中，并结合 [security.md](security.md) 的路径和审计策略使用。

## 4. 验证安装

```bash
xcode --help
xcode setup
xcode -p "输出一句安装成功"
```

若执行 `xcode` 没有指定子命令，程序启动终端 TUI。标准 REPL 使用 `xcode cli`，浏览器工作台使用 `xcode web`。

## 5. 运行目录

启动时默认以当前目录作为项目根目录。可以显式指定：

```bash
xcode --project-root /path/to/project
xcode --project-root D:\\work\\project cli
```

会话、快照、MCP、技能和项目记忆都会依据这个项目根目录建立各自的运行边界。
