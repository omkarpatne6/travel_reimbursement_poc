from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    category: str
    amount: float
    receipt: bool
    vendor: str
    date: str  # ISO format YYYY-MM-DD


class Trip(BaseModel):
    destination: str
    trip_type: Literal["domestic", "international"]
    start_date: str
    end_date: str
    purpose: str
    days: int = Field(ge=1)


class ClaimRequest(BaseModel):
    # LLM routing -- chosen by caller
    model_provider: Literal["openai", "azure_openai", "gemini", "anthropic"] = "openai"
    model_name: Optional[str] = None  # Overrides env default if provided

    # Claim data
    claim_id: str
    employee_id: str
    employee_role: str
    submission_date: str
    trip: Trip
    line_items: List[LineItem]
    total_claimed: float
    prior_claims_this_month: List[LineItem] = []


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class NodeTrace(BaseModel):
    node: str
    type: Literal["llm", "deterministic"]
    tools_called: List[str] = []
    provider: Optional[str] = None
    model: Optional[str] = None
    started_at: str
    duration_ms: float
    token_usage: TokenUsage
    # Set only when the node's LLM call failed or returned an unparseable
    # response and it fell back to the deterministic tool result instead.
    fallback_reason: Optional[str] = None


class Observability(BaseModel):
    node_trace: List[NodeTrace]
    total_tokens: TokenUsage


class EvaluationResponse(BaseModel):
    claim_id: str
    decision: str
    total_claimed: float
    approved_amount: float
    rejected_amount: float
    deductions: list
    missing_documents: List[str]
    policy_references: List[str]
    required_approver: str
    confidence: float
    explanation: str
    manual_review_reason: Optional[str]
    audit_trail: List[str]
    model_provider_used: str
    model_name_used: Optional[str]
    observability: Observability
