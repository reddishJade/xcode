# 安装与环境准备

Xcode 是一个用 Python 3.12+ 构建的 Coding Agent 运行时骨架。本文档介绍 Xcode 的系统要求、安装方式以及运行环境配置。

---

## 1. 前置条件

* **Python 版本**：Python **3.12** 或更高版本；
* **包管理器**：推荐使用 [uv](https://docs.astral.sh/uv/)（也可使用 pip）；
* **Linux 沙箱依赖**（推荐）：Linux 环境下 Agent 的 `bash` 默认运行在 bubblewrap 沙箱中，需要系统中安装 `bwrap` 命令：
  ```bash
  # Ubuntu / Debian
  sudo apt-get update && sudo apt-get install -y bubblewrap

  # Fedora / RHEL / CentOS
  sudo dnf install -y bubblewrap

  # Arch Linux
  sudo pacman -S bubblewrap
  ```
  > 若系统未安装 `bwrap`，Linux 下启动沙箱 shell 会以 Fail-closed 策略拒绝执行并提示安装。

---

## 2. 安装方式

### 方式一：开发模式安装（推荐）
在本地克隆代码库并以 editable 模式安装，修改源码即时生效：

```bash
git clone https://github.com/reddishJade/xcode.git
cd xcode

# 创建并激活虚拟环境（使用 uv）
uv venv
source .venv/bin/activate  # Linux / macOS
# 或 .venv\Scripts\activate  # Windows

# 安装核心依赖
uv pip install -e .

# 安装开发依赖（包含 ruff, pyright, pytest 等）
uv pip install -e ".[dev]"
```

### 方式二：全局工具安装 (uv tool)
如果你希望在系统的任意目录下直接调用 `xcode` 命令：

```bash
uv tool install --python 3.12 <path-to-xcode>
```

更新已安装的全局版本：
```bash
uv tool upgrade xcode --no-cache
```

---

## 3. 验证安装

安装完成后，在终端运行以下命令验证：

```bash
# 查看帮助与子命令
xcode --help

# 运行交互式首次配置向导
xcode setup
```

---

**下一篇**：[模型与 Provider 配置 (providers.md)](providers.md) →

