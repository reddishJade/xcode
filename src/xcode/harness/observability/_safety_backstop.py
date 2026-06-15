"""Shell 命令安全分级（三桶分类）与复合命令拆分。

本模块从 permission_model.py 拆分而来，仅处理 shell 命令安全分类：
- SafetyBackstopPolicyEvaluator（Bucket A/B/C 分类）
- shell 辅助谓词（_is_root_recursive_deletion, _split_compound_command, etc.）
- 桶常量（BUCKET_A_CREDENTIAL_SUBSTRINGS, BUCKET_B_ASK_COMMANDS, etc.）

模块边界规则：
- 从 permission_model.py 导入 Action/Constraint 等共享模型类型（单向依赖）
- 不允许反向导入 permission_model 的其他分组
- 不包含 grant store、resolver、ActionExtractor 或其他 evaluator
"""

from __future__ import annotations

import shlex

from .permission_model import Action, Constraint


class SafetyBackstopPolicyEvaluator:
    """对 shell 命令进行三桶分类的不可绕过安全策略。

    按 section 10.2 定义的三桶模式对 shell 命令逐段评估：
    - Bucket A: non-bypassable deny
    - Bucket B: ask
    - Bucket C: allow
    - 未匹配: ask（安全默认）

    Compound 命令按 section 10.4 拆分后逐段独立评估，
    整体 verdict 取最严格约束。
    """

    BUCKET_A_CREDENTIAL_SUBSTRINGS: tuple[str, ...] = (
        ".ssh/",
        ".ssh",
        ".gnupg/",
        ".gnupg",
        ".aws/",
        ".aws",
        ".gcloud/",
        ".gcloud",
        ".config/git/credentials",
        ".netrc",
    )

    BUCKET_A_DEVICE_PATTERNS: tuple[str, ...] = (
        "mkfs.",
        "> /dev/sda",
        "> /dev/sdb",
        "> /dev/nvme",
    )

    BUCKET_B_ASK_COMMANDS: set[str] = {
        "rm",
        "mv",
        "curl",
        "wget",
        "kill",
        "pkill",
        "killall",
        "nohup",
        "disown",
        "tmux",
        "screen",
        "sudo",
    }

    BUCKET_C_ALLOW_COMMANDS: set[str] = {
        "ls",
        "dir",
        "find",
        "rg",
        "grep",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "pwd",
        "which",
        "type",
        "realpath",
        "echo",
        "printf",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "diff",
        "cmp",
        "comm",
        "du",
        "df",
        "ruff",
        "pyright",
        "mypy",
        "flake8",
        "eslint",
        "pytest",
        "unittest",
        "make",
        "uname",
        "env",
        "export",
        "cd",
        "npx",
        "prettier",
    }

    BUCKET_B_GIT_PREFIXES: tuple[str, ...] = (
        "git reset --hard",
        "git clean -f",
        "git push --force",
        "git push -f",
        "git config --global",
    )

    BUCKET_C_GIT_PREFIXES: tuple[str, ...] = (
        "git status",
        "git diff",
        "git log",
        "git show",
        "git branch",
        "git stash list",
        "git stash show",
        "git remote -v",
        "git tag",
        "git config --list",
        "git config --get",
    )

    BUCKET_A_GIT_CREDENTIAL_PATTERNS: tuple[str, ...] = (
        "git credential",
        "git config --global credential",
    )

    BUCKET_A_PACKAGE_PATTERNS: tuple[str, ...] = (
        "apt install",
        "apt-get install",
        "yum install",
        "dnf install",
        "brew install",
    )

    BUCKET_A_SERVICE_PATTERNS: tuple[str, ...] = (
        "systemctl start",
        "systemctl stop",
        "systemctl enable",
        "systemctl disable",
        "systemctl restart",
        "systemctl reload",
        "service ",
        "initctl ",
    )

    BUCKET_B_DOCKER_PREFIXES: tuple[str, ...] = (
        "docker build",
        "docker run",
        "docker push",
        "docker commit",
    )

    BUCKET_C_CHECK_PREFIXES: tuple[str, ...] = (
        "ruff format --check",
        "prettier --check",
        "ruff check",
        "tsc --noEmit",
        "python --version",
        "node --version",
        "go version",
        "rustc --version",
        "cargo --version",
        "pip list",
        "pip show",
        "pip freeze",
        "npm list",
        "npm outdated",
        "npx --help",
        "cargo test",
        "cargo build",
        "cargo check",
        "go test",
        "go build",
        "go vet",
        "go fmt",
    )

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        if action.capability != "shell":
            return ()
        command = self._get_command(action)
        if not command:
            return ()
        segments = _split_compound_command(command)
        constraints: list[Constraint] = []
        for segment in segments:
            constraint = self._evaluate_segment(segment)
            if constraint is not None:
                constraints.append(constraint)
        return tuple(constraints)

    def _get_command(self, action: Action) -> str:
        for target in action.targets:
            if target.kind == "command":
                return target.value
        return ""

    def _evaluate_segment(self, segment: str) -> Constraint | None:
        stripped = segment.strip()
        if not stripped:
            return None

        bucket_a = self._check_bucket_a(stripped)
        if bucket_a is not None:
            return bucket_a

        bucket_b = self._check_bucket_b(stripped)
        if bucket_b is not None:
            return bucket_b

        bucket_c = self._check_bucket_c(stripped)
        if bucket_c is not None:
            return bucket_c
        return Constraint(
            decision="ask",
            source="safety_backstop",
            reason=f"未识别的命令，安全默认 ask: {stripped[:200]}",
        )

    def _check_bucket_a(self, command: str) -> Constraint | None:
        """Bucket A: 必须 non-bypassable deny 的模式。"""
        normalized = command.strip().lower()

        if _is_root_recursive_deletion(command):
            return self._deny_constraint("根目录递归删除操作", non_bypassable=True)

        if _is_system_path_recursive_mutation(command):
            return self._deny_constraint("系统关键路径递归破坏", non_bypassable=True)

        if _is_dd_device_write(command):
            return self._deny_constraint("dd 直接设备写入", non_bypassable=True)

        for pattern in self.BUCKET_A_DEVICE_PATTERNS:
            if pattern in normalized:
                return self._deny_constraint(
                    f"裸设备写入: {pattern}", non_bypassable=True
                )

        if _is_root_recursive_permission_change(command):
            return self._deny_constraint("根目录递归权限修改", non_bypassable=True)

        for pattern in self.BUCKET_A_CREDENTIAL_SUBSTRINGS:
            if pattern in command:
                return self._deny_constraint(
                    f"凭据路径访问: {pattern}", non_bypassable=True
                )

        for pattern in self.BUCKET_A_GIT_CREDENTIAL_PATTERNS:
            if pattern in normalized:
                return self._deny_constraint(
                    f"git 凭据 helper 调用: {pattern}", non_bypassable=True
                )

        for pattern in self.BUCKET_A_PACKAGE_PATTERNS:
            if normalized.startswith(pattern) or f" {pattern}" in normalized:
                return self._deny_constraint(
                    f"系统包管理器安装: {pattern}", non_bypassable=True
                )

        for pattern in self.BUCKET_A_SERVICE_PATTERNS:
            if normalized.startswith(pattern) or f" {pattern}" in normalized:
                return self._deny_constraint(
                    f"系统服务控制: {pattern}", non_bypassable=True
                )

        return None

    def _check_bucket_b(self, command: str) -> Constraint | None:
        """Bucket B: 必须 ask 的模式。"""
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None

        first = tokens[0].lower()

        if first in self.BUCKET_B_ASK_COMMANDS:
            return self._ask_constraint(f"高风险命令: {first}")

        if first == "git" and len(tokens) > 1:
            git_cmd = " ".join(tokens).lower()
            for prefix in self.BUCKET_B_GIT_PREFIXES:
                if git_cmd.startswith(prefix):
                    return self._ask_constraint(f"强制 git 操作: {prefix}")

        for opname in ("chmod", "chown"):
            if first == opname and _has_recursive_flag(tokens[1:]):
                return self._ask_constraint(f"递归权限修改: {opname}")

        if first == "docker":
            docker_cmd = " ".join(tokens).lower()
            for prefix in self.BUCKET_B_DOCKER_PREFIXES:
                if docker_cmd.startswith(prefix):
                    return self._ask_constraint(f"Docker 变异操作: {prefix}")

        return None

    def _check_bucket_c(self, command: str) -> Constraint | None:
        """Bucket C: 已知安全可 allow 的模式。"""
        if _segment_has_redirect_or_substitution(command):
            return None

        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None

        first = tokens[0].lower()

        if first == "git" and len(tokens) > 1:
            git_cmd = " ".join(tokens).lower()
            for prefix in self.BUCKET_C_GIT_PREFIXES:
                if git_cmd.startswith(prefix):
                    return self._allow_constraint(f"git 只读操作: {prefix}")

        check_cmd = command.strip().lower()
        for prefix in self.BUCKET_C_CHECK_PREFIXES:
            if check_cmd.startswith(prefix):
                return self._allow_constraint(f"已知安全命令: {prefix}")

        if first in self.BUCKET_C_ALLOW_COMMANDS:
            return self._allow_constraint(f"已知安全命令: {first}")

        if first == "command" and len(tokens) > 1 and tokens[1] == "-v":
            return self._allow_constraint("已知安全命令: command -v")

        if first == "git" and len(tokens) == 1:
            return self._allow_constraint("已知安全命令: git")

        return None

    def _deny_constraint(
        self, reason: str, *, non_bypassable: bool = False
    ) -> Constraint:
        return Constraint(
            decision="deny",
            source="safety_backstop",
            reason=reason,
            non_bypassable=non_bypassable,
        )

    def _ask_constraint(self, reason: str) -> Constraint:
        return Constraint(
            decision="ask",
            source="safety_backstop",
            reason=reason,
        )

    def _allow_constraint(self, reason: str) -> Constraint:
        return Constraint(
            decision="allow",
            source="safety_backstop",
            reason=reason,
        )


def _split_compound_command(command: str) -> list[str]:
    """按顶层操作符拆分 shell 复合命令，忽略引号和 $()/反引号内操作符。

    操作符: && || ; | \\n
    $() 和反引号不拆分，视为单段 opaque。
    """
    command = _normalize_backslash_continuation(command)
    segments: list[str] = []
    current: list[str] = []
    i = 0
    n = len(command)
    in_single = False
    in_double = False
    in_backtick = False
    paren_depth = 0

    while i < n:
        ch = command[i]

        if in_single:
            current.append(ch)
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            current.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue

        if in_backtick:
            current.append(ch)
            if ch == "`":
                in_backtick = False
            i += 1
            continue

        if paren_depth > 0:
            current.append(ch)
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
            i += 1
            continue

        if ch == "'":
            in_single = True
            current.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double = True
            current.append(ch)
            i += 1
            continue

        if ch == "`":
            in_backtick = True
            current.append(ch)
            i += 1
            continue

        if ch == "$" and i + 1 < n and command[i + 1] == "(":
            paren_depth = 1
            current.append(ch)
            current.append("(")
            i += 2
            continue

        sep_len = 0
        if ch == ";":
            sep_len = 1
        elif ch == "|" and i + 1 < n and command[i + 1] == "|":
            sep_len = 2
        elif ch == "&" and i + 1 < n and command[i + 1] == "&":
            sep_len = 2
        elif ch == "|":
            sep_len = 1
        elif ch == "\n":
            sep_len = 1

        if sep_len:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            i += sep_len
            continue

        current.append(ch)
        i += 1

    segment = "".join(current).strip()
    if segment:
        segments.append(segment)

    return segments


def _normalize_backslash_continuation(command: str) -> str:
    """将反斜杠换行续行（\\\\n）替换为空格。"""
    return command.replace("\\\n", " ")


def _is_dd_device_write(command: str) -> bool:
    """检查 dd 命令是否写入 /dev/ 设备。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or tokens[0] != "dd":
        return False
    return any(token.startswith("of=/dev/") for token in tokens[1:])


def _is_root_recursive_deletion(command: str) -> bool:
    """检查是否为 rm -rf / 类根目录递归删除（双 pass，不依赖 flag 顺序）。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or tokens[0] != "rm":
        return False
    has_recursive = any(
        _is_short_flag_with_r(t) for t in tokens[1:] if t.startswith("-") and t != "--"
    )
    if not has_recursive:
        return False
    for token in tokens[1:]:
        if token == "--":
            continue
        if token.startswith("-"):
            continue
        cleaned = token.rstrip("/")
        if cleaned in {"", "/", "/*"}:
            return True
    return False


def _is_system_path_recursive_mutation(command: str) -> bool:
    """检查是否为系统关键路径递归破坏。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if len(tokens) < 3 or tokens[0] not in {"rm", "mv", "chmod", "chown"}:
        return False
    if not _has_recursive_flag(tokens[1:]):
        return False
    return any(_is_protected_system_target(token) for token in tokens[1:])


def _is_protected_system_target(token: str) -> bool:
    protected = {
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/sys",
        "/usr",
        "/var",
    }
    cleaned = token.rstrip("/")
    if cleaned in {"", "/", "/*"}:
        return True
    return cleaned in protected


def _is_root_recursive_permission_change(command: str) -> bool:
    """检查是否为 chmod -R 777 / 等根目录递归权限修改。"""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or tokens[0] not in {"chmod", "chown"}:
        return False
    if not _has_recursive_flag(tokens[1:]):
        return False
    for token in tokens[1:]:
        if token == "--":
            continue
        if token.startswith("-"):
            continue
        cleaned = token.rstrip("/")
        if cleaned in {"", "/", "/*"}:
            return True
    return False


def _is_short_flag_with_r(token: str) -> bool:
    """检查 token 是否为含 r/R 的短 flag 组合（-rf, -Rf, -r 等）。

    要求：
    - 单破折号开头（-- 开头的不算）；
    - 去掉前导 - 后全部为字母，长度 <= 5，且至少含一个 r/R。
    长度限制排除 -version、-format 等单破折号长 flag。
    """
    if not token.startswith("-") or token.startswith("--"):
        return False
    stripped = token.lstrip("-")
    if not stripped or not stripped.isalpha() or len(stripped) > 5:
        return False
    return "r" in stripped.lower()


def _has_recursive_flag(tokens: list[str]) -> bool:
    """检查 token 列表中是否有递归短 flag（-r 或 -R）。"""
    for token in tokens:
        if token == "--":
            return False
        if _is_short_flag_with_r(token):
            return True
    return False


def _segment_has_redirect_or_substitution(segment: str) -> bool:
    """检查命令段是否包含重定向或命令替换。

    含重定向或命令替换的段不能进入 Bucket C。
    """
    if "$(" in segment:
        return True
    if "`" in segment:
        return True
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return True
    for token in tokens:
        if token in (">", ">>", ">&", "<>", "<", "<<", "<<<", "<("):
            return True
        if token == "tee":
            return True
    return False
