from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Decision(str, Enum):
    APPROVE = "Approve"
    PARTIAL = "Partially Approve"
    REJECT = "Reject"
    MANUAL = "Manual Review"


class Deduction(BaseModel):
    category: str
    claimed: float
    approved: float
    reason: str


class ReimbursementDecision(BaseModel):
    claim_id: str
    decision: Decision
    total_claimed: float
    approved_amount: float
    rejected_amount: float
    deductions: List[Deduction]
    missing_documents: List[str]
    policy_references: List[str]
    required_approver: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    manual_review_reason: Optional[str]
    audit_trail: List[str]
    model_provider_used: str
    model_name_used: Optional[str]
