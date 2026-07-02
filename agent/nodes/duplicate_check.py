import json
from datetime import datetime, timezone

from agent import llm_factory
from agent.llm_utils import invoke_json
from agent.observability import extract_token_usage, trace_node
from agent.prompts import DUPLICATE_CHECK_PROMPT
from agent.state import AgentState
from agent.tools.duplicate_detector import detect_duplicates


def run(state: AgentState) -> dict:
    claim = state["claim"]
    provider = state["model_provider"]
    model = llm_factory.resolve_model_name(provider, state.get("model_name"))

    # Ground truth: exact (date, category, amount) matches against prior
    # claims, computed in code. The LLM's job is to also catch *near*
    # duplicates (vendor spelling variants, off-by-a-day dates) that exact
    # tuple matching would miss.
    deterministic_duplicates = detect_duplicates(claim)

    with trace_node(
        "duplicate_check", node_type="llm", tools_called=["duplicate_detector.detect_duplicates"]
    ) as trace:
        trace["provider"] = provider
        trace["model"] = model

        llm = llm_factory.get_llm(provider=provider, model_name=state.get("model_name"), temperature=0)
        context = {
            "line_items": claim["line_items"],
            "prior_claims_this_month": claim.get("prior_claims_this_month", []),
            "deterministic_duplicates": deterministic_duplicates,
        }
        prompt = DUPLICATE_CHECK_PROMPT.format(context=json.dumps(context, indent=2))

        try:
            parsed, response = invoke_json(llm, prompt)
            trace["token_usage"] = extract_token_usage(response)
            duplicates = parsed["duplicate_items"]
        except Exception as e:
            duplicates = deterministic_duplicates
            trace["fallback_reason"] = f"LLM verdict unavailable, using deterministic result: {e}"

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] duplicate_check: {len(duplicates)} duplicate(s) detected"
    ]

    return {
        "duplicate_items": duplicates,
        "audit_trail": audit,
        "node_trace": [trace],
    }
