"""Health check: lets load balancers, Docker and monitoring know the app is alive."""
from fastapi import APIRouter, Request

from app.api.deps import SettingsDep
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, settings: SettingsDep) -> HealthResponse:
    state = request.app.state
    rag_ready = getattr(state, "rag", None) is not None
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        rag_ready=rag_ready,
        rag_error=None if rag_ready else getattr(state, "rag_error", None),
        iris_ready=getattr(state, "iris", None) is not None,
    )
