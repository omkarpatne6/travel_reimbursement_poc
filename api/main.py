import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

load_dotenv()

from agent.llm_factory import get_available_providers
from agent.vector_store import get_store
from api.routes import evaluate, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store(os.environ.get("POLICY_PATH", "data/policy/travel_policy.md"))
    app.state.available_providers = get_available_providers()
    print("Vector store ready")
    print(f"Available LLM providers: {app.state.available_providers}")
    yield


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
app.include_router(health.router, prefix="/api/v1")

@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse("index.html")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")
