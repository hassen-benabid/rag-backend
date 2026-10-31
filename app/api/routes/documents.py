"""Document endpoints: upload PDFs, list them, delete them.

Shows: file uploads, background tasks, path and query parameters, status codes.
"""
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, UploadFile, status

from app.api.deps import RagDep, SettingsDep
from app.schemas.documents import DocumentList, DocumentOut, UploadAccepted

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks,
                          rag: RagDep, settings: SettingsDep) -> UploadAccepted:
    """Upload a PDF. It's indexed AFTER the response is sent (202 = accepted, not done yet)."""
    # Path(...).name strips folders, so "../../evil.pdf" can't escape the upload dir
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are accepted")

    content = await file.read()  # async: reading the upload doesn't block other requests
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"File larger than {settings.max_upload_mb} MB")
    if not content.startswith(b"%PDF"):  # don't trust the file name alone
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File is not a valid PDF")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = upload_dir / filename
    saved.write_bytes(content)

    # Embedding a big PDF can take a while: do it in the background so the client
    # gets an answer immediately. (Production apps use a job queue like Celery for this.)
    background_tasks.add_task(rag.ingest_pdf, saved)
    return UploadAccepted(filename=filename)


@router.get("", response_model=DocumentList)
def list_documents(
    rag: RagDep,
    search: str | None = Query(default=None, description="Only names containing this text"),
) -> DocumentList:
    """List indexed documents. Example of a QUERY parameter: /documents?search=hand"""
    docs = rag.list_documents()
    if search:
        docs = {name: n for name, n in docs.items() if search.lower() in name.lower()}
    return DocumentList(
        documents=[DocumentOut(source=name, chunks=n) for name, n in sorted(docs.items())],
        total_chunks=sum(docs.values()),
    )


@router.delete("/{filename}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(filename: str, rag: RagDep) -> None:
    """Example of a PATH parameter. Unknown names -> 404 (via the exception handler)."""
    rag.delete_document(filename)
