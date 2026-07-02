import json
from datetime import datetime, timezone

from agent import llm_factory
from agent.llm_utils import invoke_json
from agent.observability import extract_token_usage, trace_node
from agent.prompts import APPROVAL_CHECK_PROMPT
from agent.state import AgentState
from agent.tools.approval_threshold import determine_approval


def run(state: AgentState) -> dict:
    claim = state["claim"]
    provider = state["model_provider"]
    model = llm_factory.resolve_model_name(provider, state.get("model_name"))

    # Ground truth: the approver tier is a strict threshold lookup, computed
    # in code. The LLM's job is to confirm it and double-check the
    # non-reimbursable category flags against the actual policy text.
    deterministic_info = determine_approval(claim)

    with trace_node(
        "approval_check", node_type="llm", tools_called=["approval_threshold.determine_approval"]
    ) as trace:
        trace["provider"] = provider
        trace["model"] = model

        llm = llm_factory.get_llm(provider=provider, model_name=state.get("model_name"), temperature=0)
        context = {
            "line_items": claim["line_items"],
            "policy_sections": state["policy_sections"],
            "deterministic_approval_info": deterministic_info,
        }
        prompt = APPROVAL_CHECK_PROMPT.format(context=json.dumps(context, indent=2))

        try:
            parsed, response = invoke_json(llm, prompt)
            trace["token_usage"] = extract_token_usage(response)
            result = parsed["approval_info"]
        except Exception as e:
            result = deterministic_info
            trace["fallback_reason"] = f"LLM verdict unavailable, using deterministic result: {e}"

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] approval_check: "
        f"requires {result.get('required_approver')}, "
        f"auto_eligible={result.get('auto_eligible')}, "
        f"non_reimbursable={result.get('non_reimbursable_items')}"
    ]

    return {
        "approval_info": result,
        "audit_trail": audit,
        "node_trace": [trace],
    }
