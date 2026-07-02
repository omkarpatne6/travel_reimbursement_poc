"""Per-node execution tracing: timing, tool calls, and normalized LLM token usage.

Every graph node wraps its work in `trace_node(...)` and returns the yielded
record under the `node_trace` state key (merged via an `operator.add`
reducer in `AgentState` since parallel branches all write to it). The API
layer aggregates these into a single `observability` block in the response.
"""

import time
from contextlib import contextmanager
from datetime import datetime, timezone


def extract_token_usage(response) -> dict:
    """Normalize token usage across LangChain chat model providers.

    Newer langchain-core versions expose a standard `usage_metadata` dict on
    the response for OpenAI, Azure OpenAI, Anthropic, and Gemini alike. Fall
    back to provider-specific `response_metadata` shapes if that's absent.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return {
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }

    meta = getattr(response, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
    total_tokens = token_usage.get("total_tokens") or (input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


@contextmanager
def trace_node(node_name: str, node_type: str = "deterministic", tools_called: list[str] | None = None):
    """Yield a mutable trace record. Nodes may mutate `token_usage`, `provider`,
    `model`, or append to `tools_called` before the block exits.
    """
    record = {
        "node": node_name,
        "type": node_type,  # "llm" | "deterministic"
        "tools_called": list(tools_called or []),
        "provider": None,
        "model": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)


def summarize_tokens(node_trace: list[dict]) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for entry in node_trace:
        usage = entry.get("token_usage", {})
        total["input_tokens"] += usage.get("input_tokens", 0)
        total["output_tokens"] += usage.get("output_tokens", 0)
        total["total_tokens"] += usage.get("total_tokens", 0)
    return total
