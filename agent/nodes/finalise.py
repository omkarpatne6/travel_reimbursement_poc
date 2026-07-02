from datetime import datetime, timezone

from agent.observability import trace_node
from agent.state import AgentState


def run(state: AgentState) -> dict:
    with trace_node("finalise") as trace:
        pass

    audit = [
        f"[{datetime.now(timezone.utc).isoformat()}] finalise: decision={state['decision']} finalised"
    ]
    return {"audit_trail": audit, "node_trace": [trace]}
