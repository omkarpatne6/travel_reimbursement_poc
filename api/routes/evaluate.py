from fastapi import APIRouter, HTTPException, Request

from agent.graph import GRAPH
from agent.observability import summarize_tokens
from agent.state import AgentState
from api.models import ClaimRequest, EvaluationResponse

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate a reimbursement claim")
async def evaluate_claim(request: Request, payload: ClaimRequest):
    available = request.app.state.available_providers
    if payload.model_provider not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Provider '{payload.model_provider}' is not configured. "
            f"Available providers: {available}",
        )

    initial_state: AgentState = {
        "claim": payload.model_dump(exclude={"model_provider", "model_name"}),
        "model_provider": payload.model_provider,
        "model_name": payload.model_name,
        "audit_trail": [],
        "errors": [],
        "node_trace": [],
        "policy_sections": [],
        "limit_results": [],
        "receipt_issues": [],
        "duplicate_items": [],
        "missing_documents": [],
        "approval_info": {},
        "decision": None,
        "approved_amount": 0.0,
        "rejected_amount": 0.0,
        "deductions": [],
        "policy_references": [],
        "confidence": 1.0,
        "explanation": "",
        "manual_review_reason": None,
    }

    try:
        final_state = await GRAPH.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    node_trace = final_state["node_trace"]

    return EvaluationResponse(
        claim_id=payload.claim_id,
        decision=final_state["decision"],
        total_claimed=payload.total_claimed,
        approved_amount=final_state["approved_amount"],
        rejected_amount=final_state["rejected_amount"],
        deductions=final_state["deductions"],
        missing_documents=final_state["missing_documents"],
        policy_references=final_state["policy_references"],
        required_approver=final_state["approval_info"].get("required_approver", "Unknown"),
        confidence=final_state["confidence"],
        explanation=final_state["explanation"],
        manual_review_reason=final_state.get("manual_review_reason"),
        audit_trail=final_state["audit_trail"],
        model_provider_used=payload.model_provider,
        model_name_used=payload.model_name,
        observability={
            "node_trace": node_trace,
            "total_tokens": summarize_tokens(node_trace),
        },
    )
