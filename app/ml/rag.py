"""RagPipeline: the rag-project logic packaged as a reusable class.

rag-project ran each stage as a script. A backend needs the same stages as
METHODS it can call per request: ingest_pdf, ask, stream, list/delete documents.
"""
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.ml.errors import DocumentNotFoundError
from app.ml.prompts import RAG_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Source:
    source: str
    page: int
    snippet: str


@dataclass
class RagAnswer:
    answer: str
    sources: list[Source]


class RagPipeline:
    # The models are passed IN (dependency injection) instead of created here.
    # That makes it easy to swap providers, or pass fake models in tests.
    def __init__(self, llm, embeddings, chroma_dir: str, collection_name: str,
                 chunk_size: int = 400, chunk_overlap: int = 50, k: int = 3):
        self.llm = llm
        self.k = k
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        self.store = Chroma(collection_name=collection_name, embedding_function=embeddings,
                            persist_directory=chroma_dir)

    # ---------- Indexing ----------
    def ingest_pdf(self, pdf_path: str | Path) -> int:
        """Load -> split -> embed -> store one PDF. Returns the number of chunks."""
        pdf_path = Path(pdf_path)
        pages = [
            Document(page_content=text, metadata={"source": pdf_path.name, "page": n})
            for n, page in enumerate(PdfReader(pdf_path).pages, start=1)
            if (text := page.extract_text() or "").strip()
        ]
        chunks = self.splitter.split_documents(pages)
        # Re-uploading the same file replaces its old chunks instead of duplicating them
        self._delete_chunks(pdf_path.name)
        if chunks:
            self.store.add_documents(chunks)
        logger.info("Ingested %s: %d pages, %d chunks", pdf_path.name, len(pages), len(chunks))
        return len(chunks)

    def list_documents(self) -> dict[str, int]:
        """{file name: number of chunks} for every indexed document."""
        metadatas = self.store.get(include=["metadatas"])["metadatas"]
        return dict(Counter(m["source"] for m in metadatas))

    def delete_document(self, source: str) -> int:
        deleted = self._delete_chunks(source)
        if deleted == 0:
            raise DocumentNotFoundError(f"Document '{source}' is not indexed")
        return deleted

    def _delete_chunks(self, source: str) -> int:
        ids = self.store.get(where={"source": source})["ids"]
        if ids:
            self.store.delete(ids=ids)
        return len(ids)

    # ---------- Querying ----------
    def _retrieve(self, question: str, k: int | None) -> tuple[str, list[Source]]:
        docs = self.store.similarity_search(question, k=k or self.k)
        context = "\n\n".join(f"[{d.metadata['source']}, page {d.metadata['page']}]\n{d.page_content}"
                              for d in docs)
        sources = [Source(d.metadata["source"], d.metadata["page"], d.page_content[:200])
                   for d in docs]
        return context, sources

    def ask(self, question: str, k: int | None = None) -> RagAnswer:
        """Retrieve -> augment the prompt -> generate, in one call."""
        context, sources = self._retrieve(question, k)
        messages = RAG_PROMPT.format_messages(context=context, question=question)
        return RagAnswer(answer=self.llm.invoke(messages).text, sources=sources)

    def stream(self, question: str, k: int | None = None) -> tuple[list[Source], Iterator[str]]:
        """Like ask(), but yields the answer piece by piece as the model writes it."""
        context, sources = self._retrieve(question, k)
        messages = RAG_PROMPT.format_messages(context=context, question=question)
        tokens = (chunk.text for chunk in self.llm.stream(messages) if chunk.text)
        return sources, tokens
