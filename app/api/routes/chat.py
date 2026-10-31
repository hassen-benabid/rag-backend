"""Chat endpoints: ask questions about the indexed PDFs.

Why `def` and not `async def`?
  The LLM/Chroma calls are BLOCKING (they wait for the network without `await`).
  FastAPI runs plain `def` routes in a thread pool, so one slow LLM call doesn't
  freeze the whole server. Use `async def` only when everything inside is awaitable.
"""
import json
from dataclasses import asdict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import RagDep
from app.schemas.chat import ChatRequest, ChatResponse, SourceOut

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, rag: RagDep) -> ChatResponse:
    """Ask a question; get the full answer and its sources in one JSON response."""
    result = rag.ask(body.question, k=body.k)
    return ChatResponse(answer=result.answer,
                        sources=[SourceOut(**asdict(s)) for s in result.sources])


@router.post("/stream")
def chat_stream(body: ChatRequest, rag: RagDep) -> StreamingResponse:
    """Same as /chat, but sends the answer word by word (like ChatGPT's typing effect).

    Format: Server-Sent Events (SSE), a simple standard browsers understand:
        event: sources  -> the retrieved chunks
        data: {"token": "..."}  (repeated)
        event: done
    """
    sources, tokens = rag.stream(body.question, k=body.k)

    def event_stream():
        yield f"event: sources\ndata: {json.dumps([asdict(s) for s in sources])}\n\n"
        for token in tokens:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
