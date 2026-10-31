"""Application entry point: creates the FastAPI app and wires everything together.

Run:  .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.router import api_router
from app.api.routes import health
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.middleware import register_middleware
from app.ml.iris import IrisClassifier
from app.ml.providers import build_chat_model, build_embeddings
from app.ml.rag import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs ONCE: the code before `yield` at startup, the code after it at shutdown.

    Loading models here (not inside routes) means they're loaded once and shared
    by all requests, instead of reloaded on every request (slow and expensive).
    """
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    app.state.iris = IrisClassifier()
    logger.info("Iris model ready")

    # If the RAG setup fails (e.g. no API key), keep the server running:
    # the other endpoints still work, and RAG endpoints answer 503 with the reason.
    try:
        api_key = {"google": settings.google_api_key,
                   "openai": settings.openai_api_key}
        app.state.rag = RagPipeline(
            llm=build_chat_model(settings.llm_provider, settings.llm_model,
                                 api_key.get(settings.llm_provider), settings.llm_timeout_seconds),
            embeddings=build_embeddings(settings.embedding_provider, settings.embedding_model,
                                        api_key.get(settings.embedding_provider)),
            chroma_dir=settings.chroma_dir,
            collection_name=settings.collection_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            k=settings.retriever_k,
        )
        logger.info("RAG pipeline ready (%s)", settings.llm_provider)
    except Exception as exc:
        app.state.rag = None
        app.state.rag_error = str(exc)
        logger.warning("RAG pipeline NOT available: %s", exc)

    yield  # <- the app serves requests while paused here

    logger.info("Shutting down")


def create_app() -> FastAPI:
    """App factory: builds a fresh app. Tests call it to get a clean app each time."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Learning project: FastAPI backend serving a RAG pipeline and a classic ML model.",
        lifespan=lifespan,
    )
    register_middleware(app, settings)
    register_exception_handlers(app)
    app.include_router(health.router)   # /health (no auth, no version prefix)
    app.include_router(api_router)      # /api/v1/...
    return app


app = create_app()
