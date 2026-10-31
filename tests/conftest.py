"""Shared test setup ("fixtures"). pytest finds this file automatically.

Key idea: app.dependency_overrides replaces real dependencies with fakes,
so tests are fast, free, and don't need API keys or network.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_iris, get_rag
from app.core.config import Settings, get_settings
from app.main import create_app
from app.ml.errors import DocumentNotFoundError
from app.ml.iris import IrisClassifier
from app.ml.rag import RagAnswer, Source

IRIS = IrisClassifier()  # train once for all tests


class FakeRag:
    """Behaves like RagPipeline, but with canned answers and no AI calls."""

    def __init__(self):
        self.docs = {"employee_handbook.pdf": 5, "kestrel_product_guide.pdf": 4}
        self.ingested = []

    def ask(self, question, k=None):
        return RagAnswer("Employees get 27 vacation days.",
                         [Source("employee_handbook.pdf", 1, "Every employee gets 27 vacation days")])

    def stream(self, question, k=None):
        return [Source("employee_handbook.pdf", 1, "...")], iter(["Employees ", "get ", "27 days."])

    def list_documents(self):
        return dict(self.docs)

    def delete_document(self, source):
        if source not in self.docs:
            raise DocumentNotFoundError(f"Document '{source}' is not indexed")
        return self.docs.pop(source)

    def ingest_pdf(self, path):
        self.ingested.append(Path(path).name)


@pytest.fixture
def fake_rag():
    return FakeRag()


@pytest.fixture
def settings(tmp_path):
    # _env_file=None: ignore the real .env so tests behave the same on every machine
    return Settings(_env_file=None, api_key=None, upload_dir=str(tmp_path / "uploads"))


@pytest.fixture
def app(fake_rag, settings):
    app = create_app()
    app.dependency_overrides[get_rag] = lambda: fake_rag
    app.dependency_overrides[get_iris] = lambda: IRIS
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def client(app):
    # Not using "with TestClient(app)" on purpose: that would run the lifespan
    # and try to build the real RAG pipeline.
    return TestClient(app)
