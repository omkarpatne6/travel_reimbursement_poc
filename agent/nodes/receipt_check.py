import json
from datetime import datetime, timezone

from agent import llm_factory
from agent.llm_utils import invoke_json
from agent.observability import extract_token_usage, trace_node
from agent.prompts import RECEIPT_CHECK_PROMPT
from agent.state import AgentState
from agent.tools.receipt_validator import validate_receipts


def run(state: AgentState) -> dict:
    claim = state["claim"]
    provider = state["model_provider"]
    model = llm_factory.resolve_model_name(provider, state.get("model_name"))

    # Ground truth: exact "receipt present + amount > $25" check, computed in
    # code. The LLM's job is to confirm these and additionally flag receipts
    # that are present but too vague to satisfy the policy (vendor/date/amount/
    # itemised description) -- a judgment call a boolean flag can't make.
    deterministic_issues = validate_receipts(claim)

    with trace_node(
        "receipt_check", node_type="llm", tools_called=["receipt_validator.validate_receipts"]
    ) as trace:
        trace["provider"] = provider
        trace["model"] = model

        llm = llm_factory.get_llm(provider=provider, model_name=state.get("model_name"), temperature=0)
        context = {
            "line_items": claim["line_items"],
            "policy_sections": state["policy_sections"],
            "deterministic_receipt_issues": deterministic_issues,
        }
        prompt = RECEIPT_CHECK_PROMPT.format(context=json.dumps(context, indent=2))

        try:
            parsed, response = invoke_json(llm, prompt)
            trace["token_usage"] = extract_token_usage(response)
            issues = parsed["receipt_issues"]
        except Exception as e:
            issues = deterministic_issues
            trace["fallback_reason"] = f"LLM verdict unavailable, using deterministic result: {e}"

    missing_docs = [
        f"Receipt missing: {i['category']} ${i['amount']:.2f} ({i['vendor']})" for i in issues
    ]

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] receipt_check: {len(issues)} issue(s) found"
    ]

    return {
        "receipt_issues": issues,
        "missing_documents": state["missing_documents"] + missing_docs,
        "audit_trail": audit,
        "node_trace": [trace],
    }
