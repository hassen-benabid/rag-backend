"""Collects all versioned routes under /api/v1.

Versioning (/api/v1, later /api/v2) lets you change the API without breaking
existing clients. The API-key check is applied once here, to every v1 route.
"""
from fastapi import APIRouter, Depends

from app.api.routes import chat, documents, predict
from app.core.security import require_api_key

api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(predict.router)
