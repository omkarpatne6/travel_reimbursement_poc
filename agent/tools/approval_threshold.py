"""Tool: approver tier lookup and non-reimbursable category flagging."""

import json
import os

_MATRIX_PATH = os.environ.get("APPROVAL_MATRIX_PATH", "data/policy/approval_matrix.json")

with open(_MATRIX_PATH, encoding="utf-8") as f:
    MATRIX = json.load(f)


def determine_approval(claim: dict) -> dict:
    total = claim["total_claimed"]
    result = {"total": total, "required_approver": "Board", "auto_eligible": False}

    for tier in MATRIX["thresholds"]:
        if total <= tier["max_amount"]:
            result = {
                "total": total,
                "required_approver": tier["approver"],
                "auto_eligible": tier["auto_eligible"],
            }
            break

    blocked = MATRIX.get("non_reimbursable_categories", [])
    flagged = [
        item["category"] for item in claim["line_items"] if item["category"].lower() in blocked
    ]
    result["non_reimbursable_items"] = flagged

    return result
