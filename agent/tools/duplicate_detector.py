"""Tool: cross-claim duplicate detection against prior submissions this month."""


def detect_duplicates(claim: dict) -> list[dict]:
    prior = claim.get("prior_claims_this_month", [])
    prior_set = {(p["date"], p["category"], p["amount"]) for p in prior}

    return [
        item
        for item in claim["line_items"]
        if (item["date"], item["category"], item["amount"]) in prior_set
    ]
