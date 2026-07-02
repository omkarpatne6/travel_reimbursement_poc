import json
from datetime import datetime, timezone

from agent import llm_factory
from agent.llm_utils import invoke_json
from agent.observability import extract_token_usage, trace_node
from agent.prompts import LIMIT_CHECK_PROMPT
from agent.state import AgentState
from agent.tools.limit_checker import check_limits


def run(state: AgentState) -> dict:
    claim = state["claim"]
    provider = state["model_provider"]
    model = llm_factory.resolve_model_name(provider, state.get("model_name"))

    # Ground truth: exact per-diem arithmetic, computed in code because LLMs
    # cannot be trusted to do precise financial math. The LLM's job below is
    # to confirm/annotate these numbers with policy-grounded reasoning, not
    # to recompute them.
    deterministic_results = check_limits(claim)

    with trace_node("limit_check", node_type="llm", tools_called=["limit_checker.check_limits"]) as trace:
        trace["provider"] = provider
        trace["model"] = model

        llm = llm_factory.get_llm(provider=provider, model_name=state.get("model_name"), temperature=0)
        context = {
            "trip": claim["trip"],
            "line_items": claim["line_items"],
            "policy_sections": state["policy_sections"],
            "deterministic_limit_check": deterministic_results,
        }
        prompt = LIMIT_CHECK_PROMPT.format(context=json.dumps(context, indent=2))

        try:
            parsed, response = invoke_json(llm, prompt)
            trace["token_usage"] = extract_token_usage(response)
            results = parsed["limit_results"]
        except Exception as e:
            results = deterministic_results
            trace["fallback_reason"] = f"LLM verdict unavailable, using deterministic result: {e}"

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] limit_check: "
        f"{sum(1 for r in results if not r['within_limit'])} item(s) over limit"
    ]

    return {
        "limit_results": results,
        "audit_trail": audit,
        "node_trace": [trace],
    }
