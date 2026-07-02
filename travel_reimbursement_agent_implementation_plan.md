# Travel Reimbursement Approval Agent — Implementation Plan v2
### Stack: LangGraph · FastAPI · Multi-Provider LLM · Python 3.11+

---

## 1. What Changed from v1

| Area | v1 (LangChain Agent) | v2 (This Plan) |
|---|---|---|
| Orchestration | `initialize_agent` — LLM decides everything | LangGraph — explicit graph nodes + conditional edges |
| Workflow visibility | Tool calls in LLM reasoning trace | Named nodes, typed state, inspectable graph |
| API | Optional FastAPI | First-class FastAPI with versioned routes |
| LLM provider | Hardcoded OpenAI | Runtime-switchable: OpenAI / Azure OpenAI / Gemini / Anthropic |
| Model config | Hardcoded in code | Read from `.env`; chosen per-request via payload |
| Audit trail | LLM trace only | Explicit `audit_trail` list updated at every node |

---

## 2. Repository Structure

```
travel-reimbursement-agent/
│
├── data/
│   ├── policy/
│   │   ├── travel_policy.md            # Core reimbursement policy (Markdown)
│   │   ├── per_diem_limits.json        # Per diem limits by trip type & category
│   │   └── approval_matrix.json        # Approval authority thresholds
│   ├── claims/
│   │   ├── claim_001.json              # Scenario: Approve
│   │   ├── claim_002.json              # Scenario: Partially Approve
│   │   ├── claim_003.json              # Scenario: Reject
│   │   ├── claim_004.json              # Scenario: Manual Review (duplicate)
│   │   └── claim_005.json              # Scenario: Manual Review (missing docs)
│   └── receipts/
│       └── receipt_metadata.json       # Mock receipt attachment metadata
│
├── agent/
│   ├── __init__.py
│   ├── state.py                        # LangGraph AgentState (TypedDict)
│   ├── graph.py                        # LangGraph graph definition & compilation
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── intake.py                   # Node 1: Parse & validate claim input
│   │   ├── policy_retrieval.py         # Node 2: RAG policy fetch per category
│   │   ├── limit_check.py              # Node 3: Per diem / category limit check
│   │   ├── receipt_check.py            # Node 4: Receipt completeness validation
│   │   ├── duplicate_check.py          # Node 5: Duplicate submission detection
│   │   ├── approval_check.py           # Node 6: Approval authority determination
│   │   ├── decision.py                 # Node 7: LLM synthesises final decision
│   │   └── manual_review.py            # Node 8: Manual review routing & reason
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── policy_lookup.py            # Tool: semantic policy search
│   │   ├── limit_checker.py            # Tool: amount vs limit comparison
│   │   ├── receipt_validator.py        # Tool: receipt metadata check
│   │   ├── duplicate_detector.py       # Tool: cross-claim duplicate detection
│   │   └── approval_threshold.py       # Tool: approver tier lookup
│   ├── llm_factory.py                  # Multi-provider LLM builder
│   ├── vector_store.py                 # FAISS index build & retrieval
│   ├── prompts.py                      # All prompt templates
│   └── output_schema.py               # Pydantic models for I/O
│
├── api/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app, lifespan, routers
│   ├── routes/
│   │   ├── evaluate.py                 # POST /api/v1/evaluate
│   │   └── health.py                   # GET  /api/v1/health
│   └── models.py                       # Request / response Pydantic models
│
├── tests/
│   ├── test_tools.py                   # Unit tests for each tool
│   ├── test_graph_nodes.py             # Unit tests for each node
│   ├── test_api.py                     # FastAPI integration tests (httpx)
│   └── eval_runner.py                  # Batch evaluation across all 5 claims
│
├── outputs/                            # Auto-generated decision JSONs
├── .env.example                        # All supported env vars documented
├── requirements.txt
├── README.md
└── run_demo.py                         # CLI entrypoint (wraps FastAPI client)
```

---

## 3. Data Design

### 3.1 Travel Policy (`data/policy/travel_policy.md`)

Structured in clear Markdown sections so the splitter creates clean, retrievable chunks:

```markdown
# Travel Reimbursement Policy

## 1. Eligible Categories
Flights, hotel, ground transport (taxi/rideshare/rental), meals, parking, conference fees.
Alcohol, personal entertainment, and upgrades are NOT reimbursable.

## 2. Per Diem Limits
### Domestic Travel
- Meals: $75/day (receipts required over $25)
- Hotel: $200/night
- Ground Transport: $50/day

### International Travel
- Meals: $100/day
- Hotel: $300/night
- Ground Transport: $80/day

## 3. Receipt Requirements
Receipts are mandatory for any single expense over $25.
Receipts must show: vendor name, date, amount, itemised description.

## 4. Booking Requirements
Flights must be booked at least 7 calendar days before departure unless approved emergency travel.
Economy class only. Business class permitted for flights over 6 hours with VP approval.

## 5. Approval Thresholds
- Up to $999: Line Manager (auto-approve eligible)
- $1,000 – $4,999: VP or Director sign-off required
- $5,000 and above: CFO approval required

## 6. Non-Reimbursable Items
Alcohol, personal phone bills, gym fees, clothing, traffic fines, tips over 20%.
```

---

### 3.2 Per Diem Limits (`data/policy/per_diem_limits.json`)

```json
{
  "domestic": {
    "meals": 75,
    "hotel": 200,
    "ground_transport": 50,
    "flight": null,
    "parking": 40,
    "conference": null
  },
  "international": {
    "meals": 100,
    "hotel": 300,
    "ground_transport": 80,
    "flight": null,
    "parking": 60,
    "conference": null
  }
}
```

`null` means no daily cap (flight and conference fees are approved at cost, subject to pre-approval).

---

### 3.3 Approval Matrix (`data/policy/approval_matrix.json`)

```json
{
  "thresholds": [
    { "max_amount": 999,   "approver": "Line Manager",  "auto_eligible": true  },
    { "max_amount": 4999,  "approver": "VP / Director", "auto_eligible": false },
    { "max_amount": 99999, "approver": "CFO",            "auto_eligible": false }
  ],
  "non_reimbursable_categories": ["alcohol", "personal_entertainment", "clothing", "fines"]
}
```

---

### 3.4 Claim Schema (`data/claims/claim_001.json`)

```json
{
  "claim_id": "CLM-2024-001",
  "employee_id": "EMP-1042",
  "employee_role": "Senior Engineer",
  "submission_date": "2024-11-05",
  "trip": {
    "destination": "New York, NY",
    "trip_type": "domestic",
    "start_date": "2024-10-28",
    "end_date": "2024-10-31",
    "purpose": "Client onsite engagement",
    "days": 3
  },
  "line_items": [
    { "category": "flight",  "amount": 420.00, "receipt": true,  "vendor": "Delta Airlines", "date": "2024-10-28" },
    { "category": "hotel",   "amount": 189.00, "receipt": true,  "vendor": "Marriott NYC",   "date": "2024-10-28" },
    { "category": "meals",   "amount": 65.00,  "receipt": true,  "vendor": "Various",         "date": "2024-10-29" },
    { "category": "taxi",    "amount": 35.00,  "receipt": true,  "vendor": "Uber",            "date": "2024-10-28" }
  ],
  "total_claimed": 709.00,
  "prior_claims_this_month": []
}
```

---

### 3.5 Five Claim Scenarios

| File | Total | Expected | Trigger |
|---|---|---|---|
| `claim_001.json` | $709 | **Approve** | All within limits, all receipts present |
| `claim_002.json` | $890 | **Partially Approve** | Meals $95 > $75 limit; taxi missing receipt |
| `claim_003.json` | $1,200 | **Reject** | Alcohol under meals; first-class flight without auth |
| `claim_004.json` | $540 | **Manual Review** | Duplicate: same items submitted 3 days ago |
| `claim_005.json` | $3,800 | **Manual Review** | VP approval required but not documented |

---

## 4. LangGraph State

### `agent/state.py`

```python
from typing import TypedDict, List, Optional, Any

class AgentState(TypedDict):
    # --- Input ---
    claim: dict                          # Raw claim payload
    model_provider: str                  # "openai" | "azure_openai" | "gemini" | "anthropic"
    model_name: Optional[str]            # Override; falls back to env default

    # --- Node outputs (accumulated) ---
    policy_sections: List[str]           # Retrieved policy chunks per category
    limit_results: List[dict]            # Per-line-item limit check results
    receipt_issues: List[dict]           # Missing/incomplete receipt items
    duplicate_items: List[dict]          # Detected duplicate line items
    approval_info: dict                  # Required approver + auto-eligible flag

    # --- Decision ---
    decision: Optional[str]              # Approve | Partially Approve | Reject | Manual Review
    approved_amount: float
    rejected_amount: float
    deductions: List[dict]
    missing_documents: List[str]
    policy_references: List[str]
    confidence: float                    # 0.0 – 1.0
    explanation: str
    manual_review_reason: Optional[str]

    # --- Meta ---
    audit_trail: List[str]               # Timestamped log of every node + tool call
    errors: List[str]                    # Non-fatal errors encountered during processing
```

Every node reads from this state and returns a **partial update dict** — LangGraph merges them automatically.

---

## 5. LangGraph Graph Design

### `agent/graph.py`

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   intake    │  Parse claim, validate schema, initialise state
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │ policy_retrieval │  RAG: fetch relevant policy for each category
                    └──────┬──────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │   (parallel — all three run concurrently)
       ┌──────▼──┐  ┌──────▼──┐  ┌─────▼──────┐
       │  limit  │  │ receipt │  │ duplicate  │
       │  check  │  │  check  │  │   check    │
       └──────┬──┘  └──────┬──┘  └─────┬──────┘
              │            │            │
              └────────────┼────────────┘
                           │   (fan-in — all three must complete)
                    ┌──────▼──────┐
                    │  approval   │  Determine required approver tier
                    │   check     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  decision   │  LLM synthesises structured decision from state
                    └──────┬──────┘
                           │
              ┌────────────┴──────────────┐
              │  conditional_edge:        │
              │  needs_manual_review()?   │
              │                           │
       ┌──────▼──────┐          ┌─────────▼──────┐
       │manual_review│          │    finalise     │
       │   (node)    │          │   (passthrough) │
       └──────┬──────┘          └─────────┬───────┘
              │                           │
              └────────────┬──────────────┘
                           │
                         END
```

**Parallel execution** of `limit_check`, `receipt_check`, and `duplicate_check` using LangGraph's `Send` API — these are independent tool calls with no data dependency between them, so running them together cuts latency.

**Conditional routing** after `decision`: if `confidence < 0.75` OR `duplicate_items` found OR `missing_documents` non-empty → route to `manual_review` node; otherwise go to `finalise`.

---

### Graph Code Sketch

```python
# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from agent.state import AgentState
from agent.nodes import (
    intake, policy_retrieval, limit_check,
    receipt_check, duplicate_check,
    approval_check, decision, manual_review, finalise
)

def needs_manual_review(state: AgentState) -> str:
    if (
        state["confidence"] < 0.75
        or len(state["duplicate_items"]) > 0
        or len(state["missing_documents"]) > 0
        or state["decision"] == "Manual Review"
    ):
        return "manual_review"
    return "finalise"

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("intake",            intake.run)
    g.add_node("policy_retrieval",  policy_retrieval.run)
    g.add_node("limit_check",       limit_check.run)
    g.add_node("receipt_check",     receipt_check.run)
    g.add_node("duplicate_check",   duplicate_check.run)
    g.add_node("approval_check",    approval_check.run)
    g.add_node("decision",          decision.run)
    g.add_node("manual_review",     manual_review.run)
    g.add_node("finalise",          finalise.run)

    # Edges
    g.set_entry_point("intake")
    g.add_edge("intake",           "policy_retrieval")

    # Fan-out: policy_retrieval → three parallel nodes
    g.add_edge("policy_retrieval", "limit_check")
    g.add_edge("policy_retrieval", "receipt_check")
    g.add_edge("policy_retrieval", "duplicate_check")

    # Fan-in: all three → approval_check (LangGraph waits for all)
    g.add_edge("limit_check",      "approval_check")
    g.add_edge("receipt_check",    "approval_check")
    g.add_edge("duplicate_check",  "approval_check")

    g.add_edge("approval_check",   "decision")

    # Conditional after decision
    g.add_conditional_edges("decision", needs_manual_review, {
        "manual_review": "manual_review",
        "finalise":      "finalise"
    })

    g.add_edge("manual_review", END)
    g.add_edge("finalise",      END)

    return g.compile()

GRAPH = build_graph()
```

---

## 6. Multi-Provider LLM Factory

### `agent/llm_factory.py`

The factory reads all provider credentials from `.env` at startup. The caller just passes a provider name (from the request payload) and optionally a model name override.

```python
import os
from functools import lru_cache
from langchain_core.language_models import BaseChatModel

PROVIDER_DEFAULTS = {
    "openai":       "gpt-4o",
    "azure_openai": "gpt-4o",          # deployment name from env
    "gemini":       "gemini-1.5-pro",
    "anthropic":    "claude-3-5-sonnet-20241022",
}

def get_llm(provider: str, model_name: str | None = None, temperature: float = 0) -> BaseChatModel:
    """
    Build a chat LLM for the given provider.
    model_name overrides the env default if supplied.
    Raises ValueError if required env vars are missing for the provider.
    """
    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or os.environ["OPENAI_MODEL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature,
        )

    elif provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model_name or os.environ["AZURE_OPENAI_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or os.environ["GEMINI_MODEL"],
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or os.environ["ANTHROPIC_MODEL"],
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: openai, azure_openai, gemini, anthropic"
        )


def get_available_providers() -> list[str]:
    """Return list of providers whose required env vars are present."""
    checks = {
        "openai":       ["OPENAI_API_KEY"],
        "azure_openai": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"],
        "gemini":       ["GOOGLE_API_KEY"],
        "anthropic":    ["ANTHROPIC_API_KEY"],
    }
    return [p for p, keys in checks.items() if all(os.environ.get(k) for k in keys)]
```

**Key design decision:** `get_available_providers()` is called at startup to surface which providers are configured, and the API validates the incoming `model_provider` field against this list — so the API returns a clear 422 if someone requests Gemini but `GOOGLE_API_KEY` is not set.

---

## 7. Node Implementations

Each node is a plain Python function `run(state: AgentState) -> dict` — the dict is a partial state update.

---

### Node 1: Intake (`nodes/intake.py`)

```python
from datetime import datetime
from agent.state import AgentState

def run(state: AgentState) -> dict:
    claim = state["claim"]
    errors = []

    # Basic schema validation
    required = ["claim_id", "employee_id", "trip", "line_items", "total_claimed"]
    for field in required:
        if field not in claim:
            errors.append(f"Missing required field: {field}")

    # Validate total matches sum of line items
    computed = sum(i["amount"] for i in claim.get("line_items", []))
    if abs(computed - claim.get("total_claimed", 0)) > 0.01:
        errors.append(
            f"Total claimed ${claim['total_claimed']} does not match "
            f"line item sum ${computed:.2f}"
        )

    audit = [f"[{datetime.utcnow().isoformat()}] intake: claim {claim.get('claim_id')} parsed, "
             f"{len(claim.get('line_items', []))} line items, errors={errors}"]

    return {
        "audit_trail": state.get("audit_trail", []) + audit,
        "errors": state.get("errors", []) + errors,
        "approved_amount": 0.0,
        "rejected_amount": 0.0,
        "deductions": [],
        "missing_documents": [],
        "policy_references": [],
        "duplicate_items": [],
        "receipt_issues": [],
        "limit_results": [],
        "policy_sections": [],
        "approval_info": {},
        "confidence": 1.0,
        "decision": None,
        "explanation": "",
        "manual_review_reason": None,
    }
```

---

### Node 2: Policy Retrieval (`nodes/policy_retrieval.py`)

```python
from datetime import datetime
from agent.vector_store import retrieve_policy, STORE
from agent.state import AgentState

def run(state: AgentState) -> dict:
    categories = list({item["category"] for item in state["claim"]["line_items"]})
    sections = []
    for cat in categories:
        query = f"reimbursement policy for {cat} expenses"
        result = retrieve_policy(STORE, query, k=2)
        sections.append(result)

    # Also pull general receipt and approval policy
    sections.append(retrieve_policy(STORE, "receipt requirements"))
    sections.append(retrieve_policy(STORE, "approval thresholds"))

    audit = [f"[{datetime.utcnow().isoformat()}] policy_retrieval: "
             f"fetched {len(sections)} sections for categories {categories}"]

    return {
        "policy_sections": sections,
        "audit_trail": state["audit_trail"] + audit,
    }
```

---

### Node 3: Limit Check (`nodes/limit_check.py`)

```python
import json
from datetime import datetime
from agent.state import AgentState

with open("data/policy/per_diem_limits.json") as f:
    LIMITS = json.load(f)

def run(state: AgentState) -> dict:
    claim = state["claim"]
    trip_type = claim["trip"]["trip_type"]
    days = claim["trip"].get("days", 1)
    results = []

    for item in claim["line_items"]:
        cat = item["category"]
        amount = item["amount"]
        daily_limit = LIMITS.get(trip_type, {}).get(cat)

        if daily_limit is None:
            results.append({
                "category": cat, "claimed": amount,
                "allowed_amount": amount, "excess": 0,
                "within_limit": True, "note": "No daily cap"
            })
        else:
            total_limit = daily_limit * days
            excess = max(0.0, amount - total_limit)
            results.append({
                "category": cat, "claimed": amount,
                "allowed_amount": min(amount, total_limit),
                "excess": excess, "within_limit": excess == 0,
                "daily_limit": daily_limit, "days": days
            })

    audit = [f"[{datetime.utcnow().isoformat()}] limit_check: "
             f"{sum(1 for r in results if not r['within_limit'])} item(s) over limit"]

    return {
        "limit_results": results,
        "audit_trail": state["audit_trail"] + audit,
    }
```

---

### Node 4: Receipt Check (`nodes/receipt_check.py`)

```python
from datetime import datetime
from agent.state import AgentState

RECEIPT_THRESHOLD = 25.0

def run(state: AgentState) -> dict:
    issues = []
    for item in state["claim"]["line_items"]:
        if item["amount"] > RECEIPT_THRESHOLD and not item.get("receipt", False):
            issues.append({
                "category": item["category"],
                "amount": item["amount"],
                "vendor": item.get("vendor", "Unknown"),
                "issue": "Receipt required for amounts over $25"
            })

    missing_docs = [
        f"Receipt missing: {i['category']} ${i['amount']:.2f} ({i['vendor']})"
        for i in issues
    ]

    audit = [f"[{datetime.utcnow().isoformat()}] receipt_check: "
             f"{len(issues)} missing receipt(s)"]

    return {
        "receipt_issues": issues,
        "missing_documents": state["missing_documents"] + missing_docs,
        "audit_trail": state["audit_trail"] + audit,
    }
```

---

### Node 5: Duplicate Check (`nodes/duplicate_check.py`)

```python
from datetime import datetime
from agent.state import AgentState

def run(state: AgentState) -> dict:
    claim = state["claim"]
    prior = claim.get("prior_claims_this_month", [])
    prior_set = {(p["date"], p["category"], p["amount"]) for p in prior}

    duplicates = [
        item for item in claim["line_items"]
        if (item["date"], item["category"], item["amount"]) in prior_set
    ]

    audit = [f"[{datetime.utcnow().isoformat()}] duplicate_check: "
             f"{len(duplicates)} duplicate(s) detected"]

    return {
        "duplicate_items": duplicates,
        "audit_trail": state["audit_trail"] + audit,
    }
```

---

### Node 6: Approval Check (`nodes/approval_check.py`)

```python
import json
from datetime import datetime
from agent.state import AgentState

with open("data/policy/approval_matrix.json") as f:
    MATRIX = json.load(f)

def run(state: AgentState) -> dict:
    total = state["claim"]["total_claimed"]
    result = {"total": total, "required_approver": "Board", "auto_eligible": False}

    for tier in MATRIX["thresholds"]:
        if total <= tier["max_amount"]:
            result = {
                "total": total,
                "required_approver": tier["approver"],
                "auto_eligible": tier["auto_eligible"]
            }
            break

    # Check for non-reimbursable categories
    blocked = MATRIX.get("non_reimbursable_categories", [])
    flagged = [
        item["category"] for item in state["claim"]["line_items"]
        if item["category"].lower() in blocked
    ]
    result["non_reimbursable_items"] = flagged

    audit = [f"[{datetime.utcnow().isoformat()}] approval_check: "
             f"requires {result['required_approver']}, "
             f"auto_eligible={result['auto_eligible']}, "
             f"non_reimbursable={flagged}"]

    return {
        "approval_info": result,
        "audit_trail": state["audit_trail"] + audit,
    }
```

---

### Node 7: Decision (`nodes/decision.py`)

This is the only node that calls the LLM. It receives fully-prepared state and synthesises the final decision.

```python
import json
from datetime import datetime
from agent.state import AgentState
from agent.llm_factory import get_llm
from agent.prompts import DECISION_PROMPT
from agent.output_schema import ReimbursementDecision

def run(state: AgentState) -> dict:
    llm = get_llm(
        provider=state["model_provider"],
        model_name=state.get("model_name"),
        temperature=0
    )

    # Build a rich context summary for the LLM
    context = {
        "claim_id":        state["claim"]["claim_id"],
        "total_claimed":   state["claim"]["total_claimed"],
        "trip":            state["claim"]["trip"],
        "line_items":      state["claim"]["line_items"],
        "policy_sections": state["policy_sections"],
        "limit_results":   state["limit_results"],
        "receipt_issues":  state["receipt_issues"],
        "duplicate_items": state["duplicate_items"],
        "approval_info":   state["approval_info"],
        "missing_docs":    state["missing_documents"],
        "errors":          state["errors"],
    }

    prompt = DECISION_PROMPT.format(context=json.dumps(context, indent=2))
    response = llm.invoke(prompt)

    # Strip markdown fences if present
    raw = response.content.strip().lstrip("```json").rstrip("```").strip()
    parsed = json.loads(raw)

    audit = [f"[{datetime.utcnow().isoformat()}] decision: "
             f"provider={state['model_provider']}, "
             f"decision={parsed.get('decision')}, "
             f"confidence={parsed.get('confidence')}"]

    return {
        "decision":           parsed["decision"],
        "approved_amount":    parsed["approved_amount"],
        "rejected_amount":    parsed["rejected_amount"],
        "deductions":         parsed.get("deductions", []),
        "policy_references":  parsed.get("policy_references", []),
        "confidence":         parsed["confidence"],
        "explanation":        parsed["explanation"],
        "manual_review_reason": parsed.get("manual_review_reason"),
        "audit_trail":        state["audit_trail"] + audit,
    }
```

---

### Node 8: Manual Review (`nodes/manual_review.py`)

```python
from datetime import datetime
from agent.state import AgentState

def run(state: AgentState) -> dict:
    reasons = []
    if state["confidence"] < 0.75:
        reasons.append(f"Low confidence score: {state['confidence']:.2f}")
    if state["duplicate_items"]:
        reasons.append(f"Duplicate items detected: {len(state['duplicate_items'])} item(s)")
    if state["missing_documents"]:
        reasons.append(f"Missing documents: {'; '.join(state['missing_documents'])}")
    if not state["approval_info"].get("auto_eligible"):
        reasons.append(f"Requires manual sign-off from {state['approval_info'].get('required_approver')}")

    audit = [f"[{datetime.utcnow().isoformat()}] manual_review: routed, reasons={reasons}"]

    return {
        "decision": "Manual Review",
        "manual_review_reason": " | ".join(reasons),
        "audit_trail": state["audit_trail"] + audit,
    }
```

---

## 8. Prompts (`agent/prompts.py`)

```python
DECISION_PROMPT = """
You are a Travel Reimbursement Approval Agent. You have already run all policy checks.
Your job is to synthesise the results below into a structured final decision.

=== CLAIM CONTEXT ===
{context}

=== DECISION RULES ===
- APPROVE: all items within limits, all receipts present, no duplicates, no policy violations.
- PARTIALLY APPROVE: some items exceed limits or have missing receipts; approve compliant portion only.
- REJECT: non-reimbursable categories present (alcohol, personal items), or first-class without VP auth.
- MANUAL REVIEW: duplicates detected, missing critical documents, conflicting data, or policy exception needed.

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object — no markdown, no preamble — matching this schema exactly:

{{
  "decision": "Approve | Partially Approve | Reject | Manual Review",
  "approved_amount": <float>,
  "rejected_amount": <float>,
  "deductions": [
    {{ "category": "<str>", "claimed": <float>, "approved": <float>, "reason": "<str>" }}
  ],
  "policy_references": ["<policy section cited>"],
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<2-3 sentence plain-English summary>",
  "manual_review_reason": "<str or null>"
}}
"""
```

---

## 9. Output Schema (`agent/output_schema.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Decision(str, Enum):
    APPROVE  = "Approve"
    PARTIAL  = "Partially Approve"
    REJECT   = "Reject"
    MANUAL   = "Manual Review"

class Deduction(BaseModel):
    category:  str
    claimed:   float
    approved:  float
    reason:    str

class ReimbursementDecision(BaseModel):
    claim_id:             str
    decision:             Decision
    total_claimed:        float
    approved_amount:      float
    rejected_amount:      float
    deductions:           List[Deduction]
    missing_documents:    List[str]
    policy_references:    List[str]
    required_approver:    str
    confidence:           float = Field(ge=0.0, le=1.0)
    explanation:          str
    manual_review_reason: Optional[str]
    audit_trail:          List[str]
    model_provider_used:  str
    model_name_used:      Optional[str]
```

---

## 10. FastAPI — First-Class Citizen

### `api/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.vector_store import build_vector_store
from agent.llm_factory import get_available_providers
from api.routes import evaluate, health
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build FAISS index once at startup
    app.state.vector_store = build_vector_store(
        os.environ.get("POLICY_PATH", "data/policy/travel_policy.md")
    )
    app.state.available_providers = get_available_providers()
    print(f"✓ Vector store ready")
    print(f"✓ Available LLM providers: {app.state.available_providers}")
    yield
    # Cleanup (if needed) goes here

app = FastAPI(
    title="Travel Reimbursement Approval Agent",
    version="1.0.0",
    description="AI-powered travel reimbursement evaluation via LangGraph",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router, prefix="/api/v1")
app.include_router(health.router,   prefix="/api/v1")
```

---

### `api/models.py`

```python
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

class LineItem(BaseModel):
    category: str
    amount:   float
    receipt:  bool
    vendor:   str
    date:     str                       # ISO format YYYY-MM-DD

class Trip(BaseModel):
    destination: str
    trip_type:   Literal["domestic", "international"]
    start_date:  str
    end_date:    str
    purpose:     str
    days:        int = Field(ge=1)

class ClaimRequest(BaseModel):
    # LLM routing — chosen by caller
    model_provider: Literal["openai", "azure_openai", "gemini", "anthropic"] = "openai"
    model_name:     Optional[str] = None   # Overrides env default if provided

    # Claim data
    claim_id:       str
    employee_id:    str
    employee_role:  str
    submission_date: str
    trip:           Trip
    line_items:     List[LineItem]
    total_claimed:  float
    prior_claims_this_month: List[LineItem] = []

class EvaluationResponse(BaseModel):
    claim_id:             str
    decision:             str
    total_claimed:        float
    approved_amount:      float
    rejected_amount:      float
    deductions:           list
    missing_documents:    List[str]
    policy_references:    List[str]
    required_approver:    str
    confidence:           float
    explanation:          str
    manual_review_reason: Optional[str]
    audit_trail:          List[str]
    model_provider_used:  str
    model_name_used:      Optional[str]
```

---

### `api/routes/evaluate.py`

```python
from fastapi import APIRouter, HTTPException, Request
from api.models import ClaimRequest, EvaluationResponse
from agent.graph import GRAPH
from agent.state import AgentState

router = APIRouter(tags=["Evaluation"])

@router.post("/evaluate", response_model=EvaluationResponse, summary="Evaluate a reimbursement claim")
async def evaluate_claim(request: Request, payload: ClaimRequest):
    # Validate provider is available
    available = request.app.state.available_providers
    if payload.model_provider not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Provider '{payload.model_provider}' is not configured. "
                   f"Available providers: {available}"
        )

    # Build initial state
    initial_state: AgentState = {
        "claim":           payload.model_dump(exclude={"model_provider", "model_name"}),
        "model_provider":  payload.model_provider,
        "model_name":      payload.model_name,
        "audit_trail":     [],
        "errors":          [],
        "policy_sections": [],
        "limit_results":   [],
        "receipt_issues":  [],
        "duplicate_items": [],
        "missing_documents": [],
        "approval_info":   {},
        "decision":        None,
        "approved_amount": 0.0,
        "rejected_amount": 0.0,
        "deductions":      [],
        "policy_references": [],
        "confidence":      1.0,
        "explanation":     "",
        "manual_review_reason": None,
    }

    try:
        final_state = await GRAPH.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    return EvaluationResponse(
        claim_id=             payload.claim_id,
        decision=             final_state["decision"],
        total_claimed=        payload.total_claimed,
        approved_amount=      final_state["approved_amount"],
        rejected_amount=      final_state["rejected_amount"],
        deductions=           final_state["deductions"],
        missing_documents=    final_state["missing_documents"],
        policy_references=    final_state["policy_references"],
        required_approver=    final_state["approval_info"].get("required_approver", "Unknown"),
        confidence=           final_state["confidence"],
        explanation=          final_state["explanation"],
        manual_review_reason= final_state.get("manual_review_reason"),
        audit_trail=          final_state["audit_trail"],
        model_provider_used=  payload.model_provider,
        model_name_used=      payload.model_name,
    )
```

---

### `api/routes/health.py`

```python
from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])

@router.get("/health", summary="Health check and provider availability")
async def health(request: Request):
    return {
        "status":              "ok",
        "available_providers": request.app.state.available_providers,
        "vector_store":        "ready",
    }
```

---

## 11. Environment Variables (`.env.example`)

```dotenv
# ── General ─────────────────────────────────────────────────────────────
POLICY_PATH=data/policy/travel_policy.md
PER_DIEM_PATH=data/policy/per_diem_limits.json
APPROVAL_MATRIX_PATH=data/policy/approval_matrix.json

# ── OpenAI ──────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o                         # default if model_name not in payload

# ── Azure OpenAI ────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o              # deployment name (not model name)
AZURE_OPENAI_API_VERSION=2024-02-01

# ── Google Gemini ────────────────────────────────────────────────────────
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-1.5-pro

# ── Anthropic ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

Only the providers whose keys are present will be active. Requesting an unconfigured provider returns HTTP 422 with a clear message.

---

## 12. Requirements (`requirements.txt`)

```
# Core
langgraph>=0.2.0
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0

# LLM providers — install only what you need
langchain-openai>=0.1.0          # OpenAI + Azure OpenAI
langchain-google-genai>=1.0.0    # Gemini
langchain-anthropic>=0.1.0       # Anthropic Claude

# Provider SDKs
openai>=1.30.0
google-generativeai>=0.7.0
anthropic>=0.28.0

# Vector store & embeddings
faiss-cpu>=1.7.4
tiktoken>=0.7.0

# API
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0                    # for test client

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
```

---

## 13. CLI Demo (`run_demo.py`)

```python
import json
import sys
import httpx
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"

def main():
    claim_path = sys.argv[1] if len(sys.argv) > 1 else "data/claims/claim_001.json"
    provider   = sys.argv[2] if len(sys.argv) > 2 else "openai"

    with open(claim_path) as f:
        claim = json.load(f)

    claim["model_provider"] = provider

    print(f"\n{'='*60}")
    print(f" Evaluating: {claim['claim_id']}  |  Provider: {provider}")
    print(f"{'='*60}\n")

    response = httpx.post(f"{BASE_URL}/evaluate", json=claim, timeout=120)
    response.raise_for_status()
    result = response.json()

    output_path = Path("outputs") / f"{claim['claim_id']}_decision.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\n✓ Decision saved to {output_path}")

if __name__ == "__main__":
    main()
```

Usage:
```bash
python run_demo.py data/claims/claim_001.json openai
python run_demo.py data/claims/claim_003.json azure_openai
python run_demo.py data/claims/claim_005.json gemini
```

---

## 14. Batch Evaluator (`tests/eval_runner.py`)

```python
import json
import httpx
from pathlib import Path

CLAIMS = [
    ("data/claims/claim_001.json", "openai",       "Approve"),
    ("data/claims/claim_002.json", "openai",       "Partially Approve"),
    ("data/claims/claim_003.json", "openai",       "Reject"),
    ("data/claims/claim_004.json", "openai",       "Manual Review"),
    ("data/claims/claim_005.json", "openai",       "Manual Review"),
]

BASE_URL = "http://localhost:8000/api/v1"

def run():
    print(f"\n{'='*70}")
    print(f"{'Claim':<16} {'Provider':<14} {'Expected':<20} {'Got':<20} {'Pass'}")
    print(f"{'='*70}")

    for path, provider, expected in CLAIMS:
        with open(path) as f:
            claim = json.load(f)
        claim["model_provider"] = provider

        r = httpx.post(f"{BASE_URL}/evaluate", json=claim, timeout=120)
        result = r.json()
        got = result.get("decision", "ERROR")
        passed = "✓" if got == expected else "✗"
        print(f"{claim['claim_id']:<16} {provider:<14} {expected:<20} {got:<20} {passed}")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    run()
```

---

## 15. Implementation Phases

### Phase 1 — Foundation (Day 1 AM, ~3 hrs)
- Set up repo, `venv`, `.env`, install requirements
- Create all mock data files (policy, per diems, approval matrix, 5 claims)
- Build `vector_store.py` — chunk policy, embed with OpenAI embeddings, persist FAISS index
- Smoke test: run a few manual `retrieve_policy()` calls

### Phase 2 — Tools & Nodes (Day 1 PM, ~3 hrs)
- Implement all 5 tool functions with `pytest` unit tests
- Implement nodes 1–6 (intake through approval_check) — no LLM calls yet
- Test each node independently with a known state dict

### Phase 3 — LangGraph Graph (Day 2 AM, ~3 hrs)
- Wire up `graph.py`: all nodes, edges, parallel fan-out, conditional routing
- Implement `llm_factory.py` with all 4 providers
- Implement `nodes/decision.py` and `nodes/manual_review.py`
- Run graph end-to-end on claim_001 in a notebook or script

### Phase 4 — FastAPI & Multi-Provider (Day 2 PM, ~3 hrs)
- Build `api/main.py` with lifespan FAISS init
- Build `api/routes/evaluate.py` and `health.py`
- Test with `httpx` against all 5 claims
- Validate provider switching: test at least 2 different providers

### Phase 5 — Polish & Submission (Day 3, ~3 hrs)
- Run `eval_runner.py` and capture output table
- Write `README.md` with setup, run, and sample outputs
- Save 5 decision JSONs to `/outputs` for submission evidence
- Document assumptions and limitations

---

## 16. Design Choices & Trade-offs

| Choice | Rationale | Trade-off |
|---|---|---|
| LangGraph over LangChain agent | Explicit graph makes agentic workflow inspectable and auditable; conditional edges handle Manual Review routing cleanly | More boilerplate than `initialize_agent`; overkill for simpler workflows |
| Parallel nodes (limit/receipt/duplicate) | Independent checks with no data dependency — running in parallel reduces latency | LangGraph fan-in requires all parallel branches to complete before proceeding |
| LLM called only in decision node | All data gathering is deterministic (tool functions); LLM only synthesises — more reliable, cheaper, easier to test | LLM doesn't drive tool selection; workflow is fixed (acceptable for structured domain) |
| Provider factory with env-based defaults | Payload controls provider; env controls credentials and defaults — clean separation | Adding a new provider requires code change in factory |
| FAISS (local, no service) | Zero infrastructure; works offline | No persistence between restarts; re-indexes on each API startup (fast for small policy) |
| FastAPI lifespan for FAISS init | Index built once at startup, shared across requests | Stateful — in multi-worker deployments, each worker builds its own index |
| Pydantic v2 throughout | Runtime validation on all I/O; clear error messages | Stricter than v1; some LangChain internals still use v1 patterns |

---

## 17. Known Limitations & Next Steps

**Current limitations:**
- Duplicate detection is in-memory / per-claim (no persistent claim database)
- Embeddings re-computed on every startup (no FAISS index file saved to disk)
- Receipt validation checks metadata only — no actual OCR on uploaded files
- No authentication on the FastAPI endpoints
- Confidence score is LLM-estimated, not statistically calibrated

**Improvements for production:**
- Persist FAISS index to disk (`index.save()`) and reload on startup
- Add SQLite or PostgreSQL for claim history and real duplicate detection
- Add Bearer token / API key auth to FastAPI routes
- Integrate Tesseract or Google Vision for receipt image validation
- Add LangSmith tracing (one env var: `LANGCHAIN_TRACING_V2=true`)
- Introduce policy versioning with effective dates
- Add streaming support to FastAPI (`StreamingResponse`) for real-time audit trail

---

## 18. README Outline

```markdown
# Travel Reimbursement Approval Agent

## Quick Start
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env`  → fill in at least one provider's keys
4. `uvicorn api.main:app --reload`
5. Open http://localhost:8000/docs  (Swagger UI, all endpoints documented)

## Evaluate a Claim (CLI)
python run_demo.py data/claims/claim_001.json openai
python run_demo.py data/claims/claim_003.json azure_openai

## Evaluate via API
POST http://localhost:8000/api/v1/evaluate
Content-Type: application/json
{ "model_provider": "gemini", "claim_id": "CLM-001", ... }

## Run All 5 Scenarios
python tests/eval_runner.py

## Check Available Providers
GET http://localhost:8000/api/v1/health

## Key Design Choices
See Section 16 of the implementation plan.
```

---

*Estimated total effort: 2.5 days for the full prototype including LangGraph graph, FastAPI, multi-provider LLM, 5 claim scenarios, eval runner, and documentation.*