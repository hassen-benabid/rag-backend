from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    rag_ready: bool
    rag_error: str | None = None
    iris_ready: bool
