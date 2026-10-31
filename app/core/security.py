"""Simple API-key authentication, written as a FastAPI dependency.

Real apps often use OAuth2 / JWT tokens instead, but the pattern is the same:
a dependency checks the request and raises 401 if it's not allowed.
"""
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings

# Declaring it this way also adds an "Authorize" button to the /docs page
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    provided: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key is None:  # auth disabled (handy for local learning)
        return
    # compare_digest takes the same time whether the key is right or wrong,
    # so attackers can't guess it character by character from response times
    if provided is None or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing or invalid API key")
