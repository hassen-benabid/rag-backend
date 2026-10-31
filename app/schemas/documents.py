from pydantic import BaseModel


class DocumentOut(BaseModel):
    source: str
    chunks: int


class DocumentList(BaseModel):
    documents: list[DocumentOut]
    total_chunks: int


class UploadAccepted(BaseModel):
    filename: str
    status: str = "processing"
    detail: str = "Indexing runs in the background. Check GET /api/v1/documents in a few seconds."
