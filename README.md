# Travel Reimbursement Approval Agent

An AI-powered travel reimbursement evaluator built with **LangGraph** (explicit, inspectable
graph orchestration) and served via **FastAPI**. LLM provider is chosen per-request at
runtime: OpenAI, Azure OpenAI, Gemini, or Anthropic.

See [`travel_reimbursement_agent_implementation_plan.md`](travel_reimbursement_agent_implementation_plan.md)
for the full design document this project implements.

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate            # Windows: venv\Scripts\activate  |  macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # fill in at least one provider's keys
uvicorn api.main:app --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI, or
http://localhost:8000/ui for the web UI (see below).

## Web UI

A self-contained, dependency-free web UI is served at `/ui` (also the target of a
redirect from `/`) once `uvicorn` is running:

```
http://localhost:8000/ui
```

- Paste claim JSON directly, or upload a `.json` file.
- One-click sample buttons load all 5 scenario claims from
  [Section 3.5](#five-claim-scenarios) for quick testing.
- Choose the model provider from a dropdown -- populated from `/api/v1/health` so
  unconfigured providers are visibly flagged.
- Optional "Advanced options" let you override the model name or point the UI at a
  different API base URL (needed only if you open `index.html` directly via
  `file://` instead of through the running server -- CORS is already wide open on the
  API for this case).
- Results show the decision badge, approved/rejected amounts, confidence, deductions,
  missing documents, policy references, and the full audit trail, plus an
  **Observability** panel: total token usage and a per-node execution trace table
  (node, type, tool(s) called, provider/model, duration, token usage) -- the same data
  described below in [Observability](#observability).

The page is plain HTML/CSS/JS (`index.html`, project root) with no build step and no
external dependencies, served via a dedicated `/ui` route in `api/main.py`.

## Evaluate a Claim (CLI)

```bash
python run_demo.py data/claims/claim_001.json openai
python run_demo.py data/claims/claim_003.json azure_openai
python run_demo.py data/claims/claim_005.json gemini
```

## Evaluate via API

```
POST http://localhost:8000/api/v1/evaluate
Content-Type: application/json

{ "model_provider": "gemini", "claim_id": "CLM-001", ... }
```

## Run All 5 Scenarios

```bash
python tests/eval_runner.py
```

## Check Available Providers

```
GET http://localhost:8000/api/v1/health
```

## Run the Test Suite

```bash
pytest
```

`test_tools.py` exercises the raw deterministic tool functions directly (no LLM or API key
needed). `test_graph_nodes.py` and `test_api.py` run the LLM-backed nodes and the full
FastAPI -> LangGraph pipeline with the LLM call stubbed out (`tests/fakes.py`), so the whole
wiring -- including each node's fallback path -- can be verified offline, deterministically,
and for free.

## Architecture

```
intake -> policy_retrieval -> [limit_check | receipt_check | duplicate_check] (parallel)
       -> approval_check -> decision -> (conditional) -> manual_review | finalise -> END
```

Every node except `intake` and `finalise` is LLM-backed -- each is a small agent that calls
its own deterministic tool for ground-truth numbers, then asks the LLM to confirm/annotate
that result with policy-grounded reasoning a plain rule engine can't produce:

- **intake** (deterministic): schema validation, line-item total reconciliation. Kept
  deterministic on purpose -- this is exact arithmetic and structural validation, where an
  LLM adds risk (imprecise math) with no judgment benefit.
- **policy_retrieval** (agentic RAG): runs a deterministic per-category FAISS lookup as a
  baseline, then asks the LLM to propose *additional* search queries specific to this claim
  (e.g. "business class flight approval for long-haul travel") so retrieval adapts to what
  the claim actually needs, not just a fixed per-category template.
- **limit_check**: `limit_checker.check_limits` computes exact per-diem numbers in code
  (never trust an LLM with financial arithmetic); the LLM confirms `within_limit` and adds a
  policy-grounded note (e.g. flagging a no-cap category that's still "at cost, subject to
  pre-approval").
- **receipt_check**: `receipt_validator.validate_receipts` computes the exact "missing
  receipt over $25" boolean; the LLM additionally flags receipts that are *present* but too
  vague to satisfy policy (vendor/date/amount/itemised description) -- a judgment call a
  boolean can't make.
- **duplicate_check**: `duplicate_detector.detect_duplicates` finds exact (date, category,
  amount) matches against prior claims; the LLM additionally looks for *near* duplicates
  (vendor spelling variants, off-by-a-day dates) that exact tuple matching would miss.
- **approval_check**: `approval_threshold.determine_approval` computes the exact approver
  tier from the total; the LLM confirms it and double-checks the non-reimbursable category
  flags against the actual policy text.
- **decision**: synthesises a structured final decision (`Approve` / `Partially Approve` /
  `Reject` / `Manual Review`) from all upstream state -- the only node with no deterministic
  tool underneath it, since this is genuinely a judgment call.
- **manual_review** (agentic): reached via conditional edge when the LLM signals
  `Manual Review`, confidence is low, or a duplicate was detected. Asks the LLM to write one
  clear, professional reason from the concrete signals, instead of concatenating strings.
- **finalise** (deterministic): passthrough terminal node -- nothing to reason about.

**Resilience:** every LLM-backed check node wraps its call in a try/except and falls back to
its deterministic tool's raw result if the LLM call fails or returns unparseable JSON (see
`fallback_reason` in the observability trace). The `decision` node is the one exception --
there's no meaningful deterministic fallback for the actual decision itself, so an LLM
outage there surfaces as an HTTP 500.

**Cost/latency trade-off:** this now issues up to 7 LLM calls per claim (vs. 1 previously).
A real run against Azure OpenAI (gpt-4o) used roughly 6,000-8,000 tokens and a few seconds
per claim end-to-end. If cost or latency matters more than the added judgment, the check
nodes' LLM calls can be removed without touching the deterministic tools underneath them --
each node's fallback path is exactly what running it deterministically looks like.

Every node appends to `audit_trail` (an `Annotated[List[str], operator.add]` field on
`AgentState`) so LangGraph can merge concurrent writes from the three parallel branches
without conflict.

## Observability

Every `/api/v1/evaluate` response includes an `observability` block with a per-node
execution trace and aggregated token usage:

```json
"observability": {
  "node_trace": [
    {
      "node": "policy_retrieval",
      "type": "llm",
      "tools_called": ["policy_lookup.lookup_policy"],
      "provider": "azure_openai",
      "model": "gpt-4o",
      "started_at": "2026-07-02T10:15:03.120Z",
      "duration_ms": 812.31,
      "token_usage": { "input_tokens": 320, "output_tokens": 40, "total_tokens": 360 },
      "fallback_reason": null
    },
    {
      "node": "decision",
      "type": "llm",
      "tools_called": [],
      "provider": "azure_openai",
      "model": "gpt-4o",
      "started_at": "2026-07-02T10:15:05.980Z",
      "duration_ms": 1842.55,
      "token_usage": { "input_tokens": 1284, "output_tokens": 156, "total_tokens": 1440 },
      "fallback_reason": null
    }
  ],
  "total_tokens": { "input_tokens": 5122, "output_tokens": 644, "total_tokens": 5766 }
}
```

- `node_trace` has one entry per graph node, in the order it ran (the three parallel checks
  -- `limit_check` / `receipt_check` / `duplicate_check` -- appear together but their
  relative order among themselves isn't guaranteed, since they execute concurrently).
- `type` is `"llm"` for every node that calls a language model (all of them except `intake`
  and `finalise`) and `"deterministic"` otherwise.
- `tools_called` names the deterministic tool function each node grounds its reasoning in
  (e.g. `limit_checker.check_limits`); empty for `decision` and `manual_review`, which have
  no underlying tool.
- `token_usage` is normalized across providers (OpenAI, Azure OpenAI, Gemini, Anthropic) via
  `agent/observability.py::extract_token_usage`, reading each provider's `usage_metadata` /
  `response_metadata` shape.
- `fallback_reason` is set (non-null) when that node's LLM call failed or returned
  unparseable JSON and it fell back to the deterministic tool's raw result instead.
- `total_tokens` is the sum across all nodes in the run.

## Design Choices & Trade-offs

| Choice | Rationale | Trade-off |
|---|---|---|
| LangGraph over a LangChain agent | Explicit graph makes the workflow inspectable and auditable; conditional edges handle Manual Review routing cleanly | More boilerplate than `initialize_agent`; overkill for simpler workflows |
| Parallel nodes (limit/receipt/duplicate) | Independent checks with no data dependency -- running in parallel reduces latency | Requires an `operator.add` reducer on `audit_trail` since all three write to it concurrently |
| Missing documents are *not* an automatic Manual Review trigger | A missing receipt is a "Partially Approve" case (deduct that line item); only duplicates, low LLM confidence, or an explicit LLM "Manual Review" call route to human review | Deviates from a naive first draft of the routing predicate, which would have forced every claim with any missing receipt into Manual Review, contradicting the documented "Partially Approve" scenario |
| LLM confirms/annotates deterministic tool output, rather than replacing it | Every check node still computes exact numbers in code first (arithmetic, exact-match lookups) and asks the LLM to reason over that ground truth -- gets real policy judgment (vague vendors, near-duplicates, pre-approval nuances) without trusting an LLM to do financial math | 7 LLM calls per claim instead of 1 -- meaningfully more cost and latency; each node's fallback path is what running it deterministically would look like if that trade-off isn't worth it |
| `intake` and `finalise` stay deterministic | Schema validation and passthrough have no judgment to add value to; the exception is easy to reason about since both are explicitly load-bearing correctness checks | The system isn't "every node calls an LLM" absolutely -- two of nine remain pure code |
| Provider factory with env-based defaults | Payload controls provider; env controls credentials and defaults -- clean separation | Adding a new provider requires a code change in `llm_factory.py` |
| FAISS (local, no service) with a hashing-embedding fallback | Zero infrastructure; policy retrieval works even without an OpenAI key | Fallback embedding is keyword-overlap only, not semantic; no persistence between restarts |
| Pydantic v2 throughout | Runtime validation on all I/O; clear error messages | Stricter than v1 |

## Known Limitations & Next Steps

**Current limitations:**
- Duplicate detection is in-memory / per-claim (no persistent claim database).
- FAISS index is rebuilt on every process startup (not persisted to disk).
- Receipt validation checks metadata only -- no OCR on uploaded files.
- No authentication on the FastAPI endpoints.
- Confidence score is LLM-estimated, not statistically calibrated.

**Improvements for production:**
- Persist the FAISS index to disk and reload on startup.
- Add SQLite/PostgreSQL for claim history and real duplicate detection.
- Add Bearer token / API key auth to FastAPI routes.
- Integrate OCR (Tesseract / Google Vision) for receipt image validation.
- Add LangSmith tracing (`LANGCHAIN_TRACING_V2=true`).
- Introduce policy versioning with effective dates.
- Add streaming support (`StreamingResponse`) for real-time audit trail.

## Repository Structure

```
travel-reimbursement-agent/
- data/policy/            travel_policy.md, per_diem_limits.json, approval_matrix.json
- data/claims/             claim_001.json .. claim_005.json (5 scenarios)
- data/receipts/           receipt_metadata.json (mock)
- agent/                   LangGraph state, graph, nodes, tools, llm_factory, prompts, observability
- api/                     FastAPI app, routes, request/response models
- index.html               web UI, served at /ui
- tests/                   pytest unit + integration tests, batch eval_runner
- outputs/                 decision JSONs written by run_demo.py
- run_demo.py              CLI entrypoint (wraps the FastAPI client)
```

## Five Claim Scenarios

| File | Total | Expected | Trigger |
|---|---|---|---|
| `claim_001.json` | $709 | Approve | All within limits, all receipts present |
| `claim_002.json` | $890 | Partially Approve | Meals $95 > $75/day limit; taxi missing receipt |
| `claim_003.json` | $1,200 | Reject | Alcohol line item (non-reimbursable) |
| `claim_004.json` | $540 | Manual Review | Duplicate: hotel line item matches a prior claim this month |
| `claim_005.json` | $3,800 | Manual Review | Requires VP/Director sign-off; not documented |
