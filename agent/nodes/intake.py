from datetime import datetime, timezone

from agent.observability import trace_node
from agent.state import AgentState


def run(state: AgentState) -> dict:
    claim = state["claim"]
    errors = []

    with trace_node("intake") as trace:
        required = ["claim_id", "employee_id", "trip", "line_items", "total_claimed"]
        for field in required:
            if field not in claim:
                errors.append(f"Missing required field: {field}")

        computed = sum(i["amount"] for i in claim.get("line_items", []))
        if abs(computed - claim.get("total_claimed", 0)) > 0.01:
            errors.append(
                f"Total claimed ${claim['total_claimed']} does not match "
                f"line item sum ${computed:.2f}"
            )

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] intake: claim {claim.get('claim_id')} parsed, "
        f"{len(claim.get('line_items', []))} line items, errors={errors}"
    ]

    return {
        "audit_trail": audit,
        "node_trace": [trace],
        "errors": state.get("errors", []) + errors,
        "approved_amount": 0.0,
        "rejected_amount": 0.0,
        "deductions": [],
        "missing_documents": [],
        "policy_references": [],
        "duplicate_items": [],
        "receipt_issues": [],
        "limit_results": [],
        "policy_sections": [],
        "approval_info": {},
        "confidence": 1.0,
        "decision": None,
        "explanation": "",
        "manual_review_reason": None,
    }
