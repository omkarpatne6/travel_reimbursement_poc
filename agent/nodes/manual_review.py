import json
from datetime import datetime, timezone

from agent import llm_factory
from agent.llm_utils import invoke_json
from agent.observability import extract_token_usage, trace_node
from agent.prompts import MANUAL_REVIEW_PROMPT
from agent.state import AgentState


def _deterministic_reason(state: AgentState) -> str:
    reasons = []
    if state["confidence"] < 0.75:
        reasons.append(f"Low confidence score: {state['confidence']:.2f}")
    if state["duplicate_items"]:
        reasons.append(f"Duplicate items detected: {len(state['duplicate_items'])} item(s)")
    if state["missing_documents"]:
        reasons.append(f"Missing documents: {'; '.join(state['missing_documents'])}")
    if not state["approval_info"].get("auto_eligible"):
        reasons.append(
            f"Requires manual sign-off from {state['approval_info'].get('required_approver')}"
        )
    if state.get("manual_review_reason"):
        reasons.append(state["manual_review_reason"])
    return " | ".join(dict.fromkeys(reasons))


def run(state: AgentState) -> dict:
    provider = state["model_provider"]
    model = llm_factory.resolve_model_name(provider, state.get("model_name"))
    fallback_reason = _deterministic_reason(state)

    with trace_node("manual_review", node_type="llm", tools_called=[]) as trace:
        trace["provider"] = provider
        trace["model"] = model

        llm = llm_factory.get_llm(provider=provider, model_name=state.get("model_name"), temperature=0)
        context = {
            "confidence": state["confidence"],
            "duplicate_items": state["duplicate_items"],
            "missing_documents": state["missing_documents"],
            "approval_info": state["approval_info"],
            "prior_manual_review_reason": state.get("manual_review_reason"),
        }
        prompt = MANUAL_REVIEW_PROMPT.format(context=json.dumps(context, indent=2))

        try:
            parsed, response = invoke_json(llm, prompt)
            trace["token_usage"] = extract_token_usage(response)
            reason = parsed["manual_review_reason"]
        except Exception as e:
            reason = fallback_reason
            trace["fallback_reason"] = f"LLM summary unavailable, using deterministic reason: {e}"

    audit = [f"[{datetime.now(timezone.utc).isoformat()}] manual_review: routed, reason={reason!r}"]

    return {
        "decision": "Manual Review",
        "manual_review_reason": reason,
        "audit_trail": audit,
        "node_trace": [trace],
    }
