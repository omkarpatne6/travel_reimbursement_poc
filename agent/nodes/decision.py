import json
from datetime import datetime, timezone

from agent import llm_factory
from agent.llm_utils import invoke_json
from agent.observability import extract_token_usage, trace_node
from agent.prompts import DECISION_PROMPT
from agent.state import AgentState


def run(state: AgentState) -> dict:
    provider = state["model_provider"]
    model = llm_factory.resolve_model_name(provider, state.get("model_name"))

    with trace_node("decision", node_type="llm", tools_called=[]) as trace:
        trace["provider"] = provider
        trace["model"] = model

        llm = llm_factory.get_llm(provider=provider, model_name=state.get("model_name"), temperature=0)

        context = {
            "claim_id": state["claim"]["claim_id"],
            "total_claimed": state["claim"]["total_claimed"],
            "trip": state["claim"]["trip"],
            "line_items": state["claim"]["line_items"],
            "policy_sections": state["policy_sections"],
            "limit_results": state["limit_results"],
            "receipt_issues": state["receipt_issues"],
            "duplicate_items": state["duplicate_items"],
            "approval_info": state["approval_info"],
            "missing_docs": state["missing_documents"],
            "errors": state["errors"],
        }

        prompt = DECISION_PROMPT.format(context=json.dumps(context, indent=2))

        try:
            parsed, response = invoke_json(llm, prompt)
            trace["token_usage"] = extract_token_usage(response)
        except json.JSONDecodeError as e:
            audit = [
                f"[{datetime.now(timezone.utc).isoformat()}] decision: "
                f"provider={provider}, FAILED to parse LLM output: {e}"
            ]
            return {
                "decision": "Manual Review",
                "confidence": 0.0,
                "explanation": "The decision model returned an unparseable response.",
                "manual_review_reason": f"LLM output parse error: {e}",
                "errors": state["errors"] + [f"decision node: JSON parse error: {e}"],
                "audit_trail": audit,
                "node_trace": [trace],
            }

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] decision: "
        f"provider={provider}, model={model}, "
        f"decision={parsed.get('decision')}, "
        f"confidence={parsed.get('confidence')}, "
        f"tokens={trace['token_usage']['total_tokens']}"
    ]

    return {
        "decision": parsed["decision"],
        "approved_amount": parsed["approved_amount"],
        "rejected_amount": parsed["rejected_amount"],
        "deductions": parsed.get("deductions", []),
        "policy_references": parsed.get("policy_references", []),
        "confidence": parsed["confidence"],
        "explanation": parsed["explanation"],
        "manual_review_reason": parsed.get("manual_review_reason"),
        "audit_trail": audit,
        "node_trace": [trace],
    }
