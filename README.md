# HH Goa 2026 — Voice-Enabled RAG (Task 2)

A production-quality, voice-enabled Retrieval-Augmented Generation system.
This repository is being built **phase-by-phase**.

## Project purpose

Build a system that can:

1. Ingest a document corpus.
2. Chunk, embed, and index it in a vector store.
3. Accept a spoken user question (microphone → STT).
4. Retrieve relevant chunks for the question.
5. Generate a grounded answer (LLM) with citations.
6. Speak the answer back (TTS).

This README will be updated as later phases are added.

## High-level architecture

```
+-------------------+        HTTP         +---------------------+
|  React + Vite UI  | <-----------------> |  FastAPI Backend    |
| (frontend/)       |   /api proxy via    | (backend/)          |
|  - voice capture  |     Vite dev proxy  |  - /health          |
|  - chat surface   |                     |  - RAG pipeline (*) |
+-------------------+                     |  - STT/TTS hooks (*)|
                                          +----------+----------+
                                                     |
                                          +----------v----------+
                                          |  Vector DB / LLMs   |
                                          |  (added in later    |
                                          |   phases)           |
                                          +---------------------+
(*) Not implemented yet — see "Current status" below.
```

## Repository layout

```
.
├── backend/          # FastAPI service (Python)
│   ├── app/          # Application package
│   ├── tests/        # pytest tests
│   └── requirements.txt
├── frontend/         # Vite + React client
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── raw/          # Original documents (gitkept)
│   └── processed/    # Cleaned / chunked outputs (gitkept)
├── scripts/          # One-off utility scripts (gitkept)
├── evaluation/       # Eval harnesses for later phases (gitkept)
├── .env.example
├── .gitignore
└── README.md
```

## Local setup

### Prerequisites

* Python 3.10+
* Node.js 18+ and npm

### Backend

```bash
cd backend
python -m venv .venv

# Windows (Git Bash / PowerShell)
source .venv/Scripts/activate       # Git Bash
# .venv\Scripts\activate            # PowerShell

pip install -r requirements.txt
```

Create a local `.env` in `backend/` (or at repo root) based on `.env.example`.
**Never commit a populated `.env`.**

### Frontend

```bash
cd frontend
npm install
```

## Running locally

### Backend (FastAPI)

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify: `curl http://localhost:8000/health` should return
`{"status":"ok","service":"rag-backend"}`.

OpenAPI docs: http://localhost:8000/docs

### Frontend (Vite)

From the `frontend/` directory:

```bash
npm run dev
```

Then visit http://localhost:5173.

The Vite dev server proxies `/api/*` requests to the FastAPI backend on
`http://localhost:8000`, so frontend and backend can talk during development
without CORS surprises.

## Running tests

From the `backend/` directory:

```bash
pytest
```

This runs the health endpoint test (and any future tests added in later
phases).

## Current project status

**Phase 1 — Project Foundation (complete)**

* Monorepo layout created.
* FastAPI backend with `GET /health` returning `{"status":"ok","service":"rag-backend"}`.
* CORS configured for local frontend development.
* Configuration loaded from environment variables (no hardcoded secrets).
* Minimal Vite + React frontend that boots cleanly.
* One backend test for `/health`.
* `.gitignore`, `.env.example`, and `README.md` in place.

**Phase 2.1 — Dataset Discovery and Analysis (complete)**

* Dataset source: `ai4bharat/MSMARCO-XI` from Hugging Face
* Lightweight remote inspection approach:
  - Uses HuggingFace Hub API to list repository files (no download)
  - Uses PyArrow HTTP range requests to read Parquet metadata
  - Samples only ~20 rows per split for inspection
  - Does NOT download the full multi-GB dataset
* Analysis modules:
  - Remote Parquet metadata inspection
  - Schema discovery (including nested fields like `passages`)
  - Text length statistics on tiny samples
  - Missing value detection in samples
  - Field role inference (query vs. document vs. metadata)
* Analysis script: `python -m scripts.analyze_dataset`
* Generated reports:
  - `data/dataset_report.json` (machine-readable)
  - `data/dataset_report.md` (human-readable)
* Test coverage for dataset helpers (see `backend/tests/test_dataset.py`)

**To run dataset analysis:**

```bash
# From repository root:
python -m scripts.analyze_dataset

# With custom options:
python -m scripts.analyze_dataset --lang hi --max-sample 20
```

This analysis completes in seconds without downloading the full dataset.

**Not yet implemented (will be added in later phases):**

* Document chunking and preprocessing pipeline.
* Embedding generation.
* Vector database (FAISS / Chroma / pgvector / etc.).
* Retrieval and reranking logic.
* LLM integration for answer generation.
* Speech-to-text (STT) and text-to-speech (TTS).
* Frontend chat UI, microphone capture, and streaming.
* Evaluation harness.
