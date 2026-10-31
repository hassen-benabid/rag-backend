"""Unit tests for the ML layer itself: no HTTP involved.

The real RagPipeline runs with fake models (deterministic embeddings and a
chat model with canned replies), so we test OUR logic without paying for AI calls.
"""
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.ml.errors import DocumentNotFoundError
from app.ml.rag import RagPipeline

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture
def rag(tmp_path):
    return RagPipeline(
        llm=FakeListChatModel(responses=["Employees get 27 vacation days."]),
        embeddings=DeterministicFakeEmbedding(size=32),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test",
    )


def test_ingest_list_and_delete(rag):
    chunks = rag.ingest_pdf(SAMPLES / "employee_handbook.pdf")
    assert chunks > 0
    assert rag.list_documents() == {"employee_handbook.pdf": chunks}

    # re-ingesting the same file replaces it instead of duplicating
    rag.ingest_pdf(SAMPLES / "employee_handbook.pdf")
    assert rag.list_documents() == {"employee_handbook.pdf": chunks}

    rag.delete_document("employee_handbook.pdf")
    assert rag.list_documents() == {}
    with pytest.raises(DocumentNotFoundError):
        rag.delete_document("employee_handbook.pdf")


def test_ask_returns_answer_with_sources(rag):
    rag.ingest_pdf(SAMPLES / "employee_handbook.pdf")
    result = rag.ask("How many vacation days?", k=2)
    assert result.answer == "Employees get 27 vacation days."
    assert len(result.sources) == 2
    assert result.sources[0].source == "employee_handbook.pdf"


def test_stream_yields_text_pieces(rag):
    rag.ingest_pdf(SAMPLES / "employee_handbook.pdf")
    sources, tokens = rag.stream("How many vacation days?")
    assert "".join(tokens) == "Employees get 27 vacation days."
    assert sources
