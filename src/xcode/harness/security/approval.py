"""审批策略与 reviewer 的稳定契约。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from xcode.agent.types import ApprovalRequest

ApprovalPolicy = Literal["on-request", "never"]
ApprovalsReviewer = Literal["user", "auto_review"]
ReviewRisk = Literal["low", "medium", "high", "critical"]
ReviewAuthorization = Literal["unknown", "low", "medium", "high"]
ReviewStatus = Literal["completed", "timed_out", "failed"]
HITLDecision = Literal["allow", "deny"]
HITLScope = Literal["once", "session", "permanent"]


@dataclass(frozen=True)
class HITLResult:
    """一次审批的封闭结果及其审计来源。"""

    decision: HITLDecision
    scope: HITLScope
    suggestion: str = ""
    status: ReviewStatus = "completed"
    rationale: str = ""
    risk: ReviewRisk | None = None
    authorization: ReviewAuthorization | None = None


PermissionApprovalCallback = Callable[[ApprovalRequest], HITLResult]
