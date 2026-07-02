import json
from datetime import datetime, timezone

from agent import llm_factory
from agent.llm_utils import invoke_json
from agent.observability import extract_token_usage, trace_node
from agent.prompts import POLICY_QUERY_PROMPT
from agent.state import AgentState
from agent.tools.policy_lookup import lookup_policy


def _baseline_queries(claim: dict) -> list[str]:
    categories = sorted({item["category"] for item in claim["line_items"]})
    return [f"reimbursement policy for {cat} expenses" for cat in categories] + [
        "receipt requirements",
        "approval thresholds",
    ]


def run(state: AgentState) -> dict:
    claim = state["claim"]
    provider = state["model_provider"]
    model = llm_factory.resolve_model_name(provider, state.get("model_name"))
    baseline = _baseline_queries(claim)

    with trace_node(
        "policy_retrieval", node_type="llm", tools_called=["policy_lookup.lookup_policy"]
    ) as trace:
        trace["provider"] = provider
        trace["model"] = model

        # The LLM decides which *additional* policy topics matter for this specific
        # claim (e.g. business-class exceptions, a non-reimbursable item it spots) --
        # the per-category lookups above are always run as a deterministic baseline so
        # retrieval never silently comes up empty if the LLM call fails.
        llm = llm_factory.get_llm(provider=provider, model_name=state.get("model_name"), temperature=0)
        context = {
            "trip": claim["trip"],
            "line_items": claim["line_items"],
            "total_claimed": claim["total_claimed"],
        }
        prompt = POLICY_QUERY_PROMPT.format(context=json.dumps(context, indent=2))

        extra_queries = []
        try:
            parsed, response = invoke_json(llm, prompt)
            trace["token_usage"] = extract_token_usage(response)
            if isinstance(parsed, list):
                extra_queries = [q for q in parsed if isinstance(q, str) and q.strip()]
        except Exception as e:
            trace["fallback_reason"] = f"query generation failed, using baseline only: {e}"

        queries = list(dict.fromkeys(baseline + extra_queries))  # de-duplicate, keep order
        sections = [lookup_policy(q, k=2) for q in queries]
        trace["queries"] = queries

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] policy_retrieval: "
        f"fetched {len(sections)} sections ({len(baseline)} baseline + {len(extra_queries)} agent-proposed)"
    ]

    return {
        "policy_sections": sections,
        "audit_trail": audit,
        "node_trace": [trace],
    }
