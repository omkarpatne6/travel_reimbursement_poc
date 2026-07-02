from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health check and provider availability")
async def health(request: Request):
    return {
        "status": "ok",
        "available_providers": request.app.state.available_providers,
        "vector_store": "ready",
    }
