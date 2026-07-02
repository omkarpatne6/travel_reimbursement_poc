"""Tool: check receipt metadata completeness for each line item."""

RECEIPT_THRESHOLD = 25.0


def validate_receipts(claim: dict) -> list[dict]:
    issues = []
    for item in claim["line_items"]:
        if item["amount"] > RECEIPT_THRESHOLD and not item.get("receipt", False):
            issues.append(
                {
                    "category": item["category"],
                    "amount": item["amount"],
                    "vendor": item.get("vendor", "Unknown"),
                    "issue": "Receipt required for amounts over $25",
                }
            )
    return issues
