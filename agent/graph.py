from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    intake,
    policy_retrieval,
    limit_check,
    receipt_check,
    duplicate_check,
    approval_check,
    decision,
    manual_review,
    finalise,
)


def needs_manual_review(state: AgentState) -> str:
    # NOTE: missing_documents is intentionally *not* a standalone trigger here.
    # A missing receipt is a "Partially Approve" case (deduct that line item) --
    # policy exceptions serious enough for human review are signalled by the
    # LLM itself (decision == "Manual Review"), by low confidence, or by a
    # detected duplicate submission.
    if (
        state["confidence"] < 0.75
        or len(state["duplicate_items"]) > 0
        or state["decision"] == "Manual Review"
    ):
        return "manual_review"
    return "finalise"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake.run)
    g.add_node("policy_retrieval", policy_retrieval.run)
    g.add_node("limit_check", limit_check.run)
    g.add_node("receipt_check", receipt_check.run)
    g.add_node("duplicate_check", duplicate_check.run)
    g.add_node("approval_check", approval_check.run)
    g.add_node("decision", decision.run)
    g.add_node("manual_review", manual_review.run)
    g.add_node("finalise", finalise.run)

    g.set_entry_point("intake")
    g.add_edge("intake", "policy_retrieval")

    # Fan-out: policy_retrieval -> three independent, parallel checks
    g.add_edge("policy_retrieval", "limit_check")
    g.add_edge("policy_retrieval", "receipt_check")
    g.add_edge("policy_retrieval", "duplicate_check")

    # Fan-in: all three must complete before approval_check runs
    g.add_edge("limit_check", "approval_check")
    g.add_edge("receipt_check", "approval_check")
    g.add_edge("duplicate_check", "approval_check")

    g.add_edge("approval_check", "decision")

    g.add_conditional_edges(
        "decision",
        needs_manual_review,
        {"manual_review": "manual_review", "finalise": "finalise"},
    )

    g.add_edge("manual_review", END)
    g.add_edge("finalise", END)

    return g.compile()


GRAPH = build_graph()
