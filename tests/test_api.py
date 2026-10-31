"""API tests: send real HTTP requests to the app (in memory) and check the responses."""
from fastapi.testclient import TestClient

from app.api.deps import get_rag
from app.main import create_app

PDF_BYTES = b"%PDF-1.4\n%fake pdf for tests\n"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_middleware_adds_headers(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert "X-Process-Time-Ms" in r.headers


# ---------- chat ----------
def test_chat_returns_answer_and_sources(client):
    r = client.post("/api/v1/chat", json={"question": "How many vacation days?"})
    assert r.status_code == 200
    body = r.json()
    assert "27" in body["answer"]
    assert body["sources"][0]["source"] == "employee_handbook.pdf"


def test_chat_rejects_invalid_input(client):
    # question too short and k out of range -> Pydantic rejects it with 422
    r = client.post("/api/v1/chat", json={"question": "hi", "k": 50})
    assert r.status_code == 422
    fields = {err["loc"][-1] for err in r.json()["detail"]}
    assert fields == {"question", "k"}


def test_chat_stream_sends_server_sent_events(client):
    r = client.post("/api/v1/chat/stream", json={"question": "How many vacation days?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "event: sources" in r.text
    assert '"token": "27 days."' in r.text
    assert r.text.rstrip().endswith("event: done\ndata: {}")


# ---------- documents ----------
def test_upload_pdf_is_ingested_in_background(client, fake_rag):
    r = client.post("/api/v1/documents",
                    files={"file": ("../../sneaky.pdf", PDF_BYTES, "application/pdf")})
    assert r.status_code == 202
    assert r.json()["filename"] == "sneaky.pdf"     # folder part was stripped
    assert fake_rag.ingested == ["sneaky.pdf"]      # background task ran after the response


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/v1/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_upload_rejects_fake_pdf(client):
    r = client.post("/api/v1/documents", files={"file": ("virus.pdf", b"MZ...", "application/pdf")})
    assert r.status_code == 415


def test_list_documents_with_search(client):
    r = client.get("/api/v1/documents", params={"search": "hand"})
    assert r.status_code == 200
    assert r.json() == {"documents": [{"source": "employee_handbook.pdf", "chunks": 5}],
                        "total_chunks": 5}


def test_delete_document(client):
    assert client.delete("/api/v1/documents/employee_handbook.pdf").status_code == 204
    assert client.delete("/api/v1/documents/employee_handbook.pdf").status_code == 404


# ---------- classic ML ----------
def test_predict_iris(client):
    r = client.post("/api/v1/predict/iris", json={
        "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 200
    assert r.json()["species"] == "setosa"
    assert r.json()["confidence"] > 0.9


def test_predict_iris_rejects_negative_values(client):
    r = client.post("/api/v1/predict/iris", json={
        "sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 422


# ---------- security & errors ----------
def test_api_key_is_enforced_when_configured(app, settings):
    settings.api_key = "secret"
    client = TestClient(app)
    question = {"question": "How many vacation days?"}
    assert client.post("/api/v1/chat", json=question).status_code == 401
    assert client.post("/api/v1/chat", json=question,
                       headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/v1/chat", json=question,
                       headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays public


def test_rag_unavailable_returns_503():
    app = create_app()  # no override for get_rag, and lifespan didn't run -> no model
    app.state.rag_error = "missing API key"
    r = TestClient(app).post("/api/v1/chat", json={"question": "How many vacation days?"})
    assert r.status_code == 503
    assert "missing API key" in r.json()["detail"]
    assert get_rag not in app.dependency_overrides
