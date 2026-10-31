"""Dependencies: things routes NEED, provided by FastAPI's dependency injection.

A route declares `rag: RagDep` and FastAPI calls get_rag() for it on each request.
Benefits:
  - routes don't care WHERE the model comes from,
  - tests can swap it with app.dependency_overrides[get_rag] = lambda: FakeRag()
"""
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.ml.errors import MLServiceUnavailableError
from app.ml.iris import IrisClassifier
from app.ml.rag import RagPipeline


def get_rag(request: Request) -> RagPipeline:
    # Models were loaded once at startup (see lifespan in main.py) and stored on app.state
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        reason = getattr(request.app.state, "rag_error", "not loaded")
        raise MLServiceUnavailableError(f"RAG service unavailable: {reason}")
    return rag


def get_iris(request: Request) -> IrisClassifier:
    model = getattr(request.app.state, "iris", None)
    if model is None:
        raise MLServiceUnavailableError("Iris model not loaded")
    return model


# Annotated aliases keep route signatures short and readable
RagDep = Annotated[RagPipeline, Depends(get_rag)]
IrisDep = Annotated[IrisClassifier, Depends(get_iris)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
