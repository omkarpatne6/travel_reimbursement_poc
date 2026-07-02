import operator
from typing import TypedDict, List, Optional, Any, Annotated


class AgentState(TypedDict):
    # --- Input ---
    claim: dict                          # Raw claim payload
    model_provider: str                  # "openai" | "azure_openai" | "gemini" | "anthropic"
    model_name: Optional[str]            # Override; falls back to env default

    # --- Node outputs (accumulated) ---
    policy_sections: List[str]           # Retrieved policy chunks per category
    limit_results: List[dict]            # Per-line-item limit check results
    receipt_issues: List[dict]           # Missing/incomplete receipt items
    duplicate_items: List[dict]          # Detected duplicate line items
    approval_info: dict                  # Required approver + auto-eligible flag

    # --- Decision ---
    decision: Optional[str]              # Approve | Partially Approve | Reject | Manual Review
    approved_amount: float
    rejected_amount: float
    deductions: List[dict]
    missing_documents: List[str]
    policy_references: List[str]
    confidence: float                    # 0.0 - 1.0
    explanation: str
    manual_review_reason: Optional[str]

    # --- Meta ---
    # `operator.add` reducer: limit_check / receipt_check / duplicate_check run
    # concurrently and each append to this key, so LangGraph needs a merge
    # strategy instead of last-write-wins. Every node returns only its *new*
    # audit lines (not the full accumulated list) -- the reducer does the
    # concatenation.
    audit_trail: Annotated[List[str], operator.add]
    errors: List[str]                    # Non-fatal errors encountered during processing

    # Observability: same concurrent-write situation as audit_trail, so it
    # gets the same `operator.add` reducer. One entry per node execution --
    # see agent/observability.py.
    node_trace: Annotated[List[dict], operator.add]
