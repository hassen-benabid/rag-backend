# RAG Backend: learn FastAPI by serving ML models

A FastAPI backend that turns the `rag-project` scripts into a real web API, and also
serves a classic scikit-learn model. The goal is to learn **FastAPI and backend
architecture** on a realistic ML example.

## What it does

| Method | URL | What it does |
|---|---|---|
| GET | `/health` | Is the server alive? Are the models loaded? |
| POST | `/api/v1/documents` | Upload a PDF; it's indexed into ChromaDB in the background |
| GET | `/api/v1/documents?search=...` | List indexed PDFs |
| DELETE | `/api/v1/documents/{filename}` | Remove a PDF from the index |
| POST | `/api/v1/chat` | Ask a question → answer + sources (RAG) |
| POST | `/api/v1/chat/stream` | Same, but the answer streams in word by word (SSE) |
| POST | `/api/v1/predict/iris` | Classic ML: predict a flower species from 4 numbers |

## Run it

1. Copy `.env.example` to `.env` and put your Gemini key in it.
2. Create the environment and install the packages (once):

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Start the server (`--reload` restarts it automatically when you edit code):

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

4. Open **http://127.0.0.1:8000/docs**. This page is generated automatically from the code.
   Click an endpoint → **Try it out** → **Execute**. Start by uploading the PDFs in `samples/`
   through `POST /api/v1/documents`, then ask questions with `POST /api/v1/chat`.

Run the tests (no API key needed, takes about 2 seconds):

```bash
.venv\Scripts\python.exe -m pytest -v
```

## The architecture: layers

```
HTTP request
   │
   ▼
app/main.py ............ creates the app, loads models at startup (lifespan)
app/core/middleware.py . runs around EVERY request: CORS, request ID, timing log
app/core/security.py ... API-key check (a dependency)
   │
   ▼
app/api/routes/*.py .... API LAYER: HTTP only. Validates input, calls the ML layer,
app/schemas/*.py         returns JSON. Schemas define the request/response shapes.
app/api/deps.py          Dependencies hand the routes the loaded models.
   │
   ▼
app/ml/*.py ............ ML LAYER: pure Python, no FastAPI imports.
                         RagPipeline (your rag-project as a class) + IrisClassifier.
   │
   ▼
ChromaDB, Gemini, scikit-learn
```

**Why layers?** Each layer has one job. The ML layer doesn't know about HTTP, so you could reuse
it in a CLI, a notebook, or a worker, or publish it as its own pip package (the "ML library").
The API layer doesn't know how RAG works; it just calls `rag.ask(...)`. You can change one
side without breaking the other, and test each separately.

## The files

```
app/
├── main.py             App factory + lifespan (load models once at startup)
├── core/
│   ├── config.py       Settings from .env, typed and validated (pydantic-settings)
│   ├── security.py     X-API-Key check
│   ├── middleware.py   CORS + request ID + timing logs
│   └── errors.py       Maps ML exceptions → HTTP status codes (404, 503, 500)
├── api/
│   ├── router.py       Groups routes under /api/v1 and applies the API-key check
│   ├── deps.py         Dependencies: get_rag, get_iris, get_settings
│   └── routes/         One file per topic: health, chat, documents, predict
├── schemas/            Pydantic models: what goes in and out of each endpoint
└── ml/                 The ML library: rag.py, iris.py, providers.py, prompts.py, errors.py
tests/
├── conftest.py         Test setup: a FakeRag replaces the real one (no AI calls)
├── test_api.py         HTTP tests for every endpoint
└── test_rag_pipeline.py  Tests for the ML layer alone
samples/                Two PDFs you can upload
storage/                Created at runtime: uploaded PDFs + the ChromaDB database
```

## FastAPI concepts, and where to find each one

| Concept | Where | In one sentence |
|---|---|---|
| **Path operations / routing** | `api/routes/*.py` | `@router.post("/chat")` connects a URL and HTTP method to a function |
| **APIRouter** | `api/router.py` | Splits routes into files and groups them under a prefix like `/api/v1` |
| **Pydantic schemas** | `schemas/` | Define data shapes; bad input is rejected automatically with **422** |
| **Path / query / body params** | `documents.py`, `chat.py` | `/documents/{filename}` (path), `?search=x` (query), JSON (body) |
| **Dependency injection** | `api/deps.py` | A route declares what it needs (`rag: RagDep`) and FastAPI provides it |
| **Lifespan** | `main.py` | Code that runs once at startup/shutdown: the place to load ML models |
| **app.state** | `main.py`, `deps.py` | Where the loaded models live, shared by all requests |
| **Settings** | `core/config.py` | All configuration in one typed class, read from `.env` |
| **Middleware** | `core/middleware.py` | Code that wraps every request (logging, timing, CORS) |
| **CORS** | `core/middleware.py` | Lets a browser frontend on another URL call this API |
| **Exception handlers** | `core/errors.py` | Turn Python errors into clean JSON responses in one place |
| **HTTPException + status codes** | `documents.py` | 202 Accepted, 204 No Content, 401, 404, 413, 415, 422, 503 |
| **Security dependency** | `core/security.py` | An API-key check; adds an "Authorize" button to `/docs` |
| **File upload** | `documents.py` | `UploadFile`, with checks for type, size, and path tricks |
| **Background tasks** | `documents.py` | Respond immediately, do the slow work afterwards |
| **Streaming (SSE)** | `chat.py` | Send the answer piece by piece as the LLM writes it |
| **`def` vs `async def`** | `chat.py` docstring | Blocking code → `def` (runs in a thread pool). Awaitable code → `async def` |
| **OpenAPI / Swagger** | `/docs` | Interactive documentation generated from your code |
| **Testing + dependency overrides** | `tests/` | `TestClient` sends in-memory requests; overrides swap real models for fakes |
| **App factory** | `main.py` `create_app()` | Builds a fresh app on demand, so each test gets a clean one |

## Exercises

1. Set `API_KEY=mysecret` in `.env`, restart, and call `/api/v1/chat` from `/docs`. You get 401. Click
   **Authorize**, enter the key, and try again.
2. Add an endpoint `GET /api/v1/predict/iris/species` that returns the list of species names.
3. Add a field `language: str = "en"` to `ChatRequest` and pass it into the prompt.
4. Change `chat()` from `def` to `async def` and send two questions at once from two browser tabs.
   The second waits for the first, because a blocking call inside `async def` freezes the server.
   Change it back.
5. Write a test for exercise 2 in `tests/test_api.py`.

## Interview questions (short answers)

**Why FastAPI?**
It's fast (built on Starlette and async), uses Python type hints for validation (Pydantic), and
generates OpenAPI docs automatically. It's very popular for serving ML models.

**What's ASGI, and what is uvicorn?**
ASGI is the standard interface between async Python web apps and servers. Uvicorn is the server
that receives HTTP requests and passes them to FastAPI. In production you often run several
uvicorn worker processes, for example behind Nginx or in Docker/Kubernetes.

**`def` or `async def`?**
Use `async def` when everything inside is awaitable (async DB driver, `httpx.AsyncClient`).
Use `def` for blocking code (most ML libraries, `requests`). FastAPI runs `def` routes in a
thread pool. A blocking call inside `async def` freezes every request.

**How do you serve an ML model efficiently?**
Load it once at startup (lifespan), keep it in memory, and validate inputs with schemas. For heavy
models: batching, a GPU server, or a dedicated serving tool (Triton, TorchServe, BentoML, vLLM).
Long jobs go to a queue (Celery, RQ, Kafka) instead of running inside the request.

**What is dependency injection, and why use it?**
Routes declare what they need instead of creating it. That avoids duplicated code, keeps routes
simple, and lets tests swap real services for fakes (`app.dependency_overrides`).

**How do you handle long-running tasks?**
Small ones: `BackgroundTasks`, as in our upload endpoint. Real ones: a job queue plus a status
endpoint (`POST` returns a job ID, `GET /jobs/{id}` shows its progress). BackgroundTasks run in
the same process, so they're lost if the server restarts.

**Why stream LLM responses?**
Users see the first words after about 0.5 s instead of waiting 10–20 s for the full answer.
Note: our timing middleware logs `/chat/stream` as fast, because it measures until the response
*starts*, not until the stream ends.

**How do you secure an API?**
Authentication (API keys, OAuth2/JWT), HTTPS, input validation, file-upload checks, rate
limiting, CORS restricted to known frontends, and never returning internal errors to clients.

**How would you deploy this?**
Package it in a Docker image, run it with several uvicorn workers, and put a reverse proxy or
load balancer in front. Configure it with environment variables, and use `/health` for health
checks. Store data in managed services (a hosted vector DB instead of a local folder).

## How it connects to rag-project

`app/ml/rag.py` is the same pipeline as `rag-project` (load PDF → chunk → embed → Chroma →
retrieve → prompt → LLM), reorganised as a class with methods. The scripts ran once and ended;
a backend keeps the models in memory and serves many users over HTTP.
