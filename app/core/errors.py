"""Error handling: turn Python exceptions into clean JSON HTTP responses.

The ML layer raises plain Python exceptions (it knows nothing about HTTP).
Here we map each one to a status code, in ONE place, instead of
try/except in every route.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.ml.errors import DocumentNotFoundError, MLServiceUnavailableError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentNotFoundError)
    async def not_found(request: Request, exc: DocumentNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(MLServiceUnavailableError)
    async def unavailable(request: Request, exc: MLServiceUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        # Log the full error for developers, but never leak internals to clients
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
