"""Tool: compare each claimed line item against per-diem category limits."""

import json
import os

_LIMITS_PATH = os.environ.get("PER_DIEM_PATH", "data/policy/per_diem_limits.json")

with open(_LIMITS_PATH, encoding="utf-8") as f:
    LIMITS = json.load(f)


def check_limits(claim: dict) -> list[dict]:
    trip_type = claim["trip"]["trip_type"]
    days = claim["trip"].get("days", 1)
    results = []

    for item in claim["line_items"]:
        cat = item["category"]
        amount = item["amount"]
        daily_limit = LIMITS.get(trip_type, {}).get(cat)

        if daily_limit is None:
            results.append(
                {
                    "category": cat,
                    "claimed": amount,
                    "allowed_amount": amount,
                    "excess": 0,
                    "within_limit": True,
                    "note": "No daily cap",
                }
            )
        else:
            total_limit = daily_limit * days
            excess = max(0.0, amount - total_limit)
            results.append(
                {
                    "category": cat,
                    "claimed": amount,
                    "allowed_amount": min(amount, total_limit),
                    "excess": excess,
                    "within_limit": excess == 0,
                    "daily_limit": daily_limit,
                    "days": days,
                }
            )

    return results
