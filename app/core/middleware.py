"""Middleware: code that runs around EVERY request (before and after the route).

Good for cross-cutting concerns: logging, timing, request IDs, CORS, compression.
"""
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings

logger = logging.getLogger("app.requests")


def register_middleware(app: FastAPI, settings: Settings) -> None:
    # CORS: browsers block calls from another website (origin) unless the API allows it
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # A request ID lets you find all log lines for one request
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        start = time.perf_counter()

        response = await call_next(request)  # <- the actual route runs here

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info("%s %s -> %s (%.1f ms) [id=%s]", request.method, request.url.path,
                    response.status_code, elapsed_ms, request_id)
        return response
