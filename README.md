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

### LLM provider configuration

The chat endpoint uses an OpenAI-compatible LLM provider when configured
(and returns `501 LLM_PROVIDER_NOT_CONFIGURED` otherwise). Configuration
is read from environment variables (or `backend/.env`) via the settings
in `backend/app/settings.py`:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai_compatible` | Provider selector (only this value is supported) |
| `LLM_API_KEY` | (none) | Provider API key - keep it secret, never commit it |
| `LLM_BASE_URL` | OpenAI API | Any OpenAI-compatible endpoint (local servers need no key) |
| `LLM_MODEL` | `gpt-4o-mini` | Model served at the endpoint |
| `LLM_TIMEOUT_SECONDS` | `60.0` | Provider call timeout |

Examples:

```bash
# Hosted OpenAI (requires a key)
LLM_API_KEY=sk-... LLM_MODEL=gpt-4o-mini

# Local OpenAI-compatible server (no key needed)
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2
```

The provider is considered configured when `LLM_API_KEY` is set **or**
`LLM_BASE_URL` is a non-default value. API keys are never logged,
surfaced in errors, or returned in responses.

### STT provider configuration

The STT (speech-to-text) layer is implemented in `backend/app/stt/`
and is **provider-agnostic**: any concrete provider implementing the
`BaseSTT` contract can be wired in. The default and only production
provider today is the OpenAI Whisper API (or any OpenAI-compatible
Whisper endpoint).

Configuration is read from environment variables (or `backend/.env`):

| Variable | Default | Purpose |
|---|---|---|
| `STT_PROVIDER` | `openai_whisper` | Provider selector. Anything other than `openai_whisper` leaves the STT layer unconfigured. |
| `STT_API_KEY` | (none) | Provider API key. Required for the hosted OpenAI Whisper API. May be omitted for local compatible servers. Falls back to `LLM_API_KEY` when unset. |
| `STT_BASE_URL` | OpenAI API | Any OpenAI-compatible Whisper endpoint (e.g. self-hosted faster-whisper, vLLM Whisper, etc.). Falls back to `LLM_BASE_URL` when unset. Local endpoints may need no API key. |
| `STT_MODEL` | `whisper-1` | Model identifier served at the endpoint |
| `STT_LANGUAGE` | (none) | Default language hint (`en`, `hi`, ...). Empty/`None` triggers automatic language detection. |
| `STT_TIMEOUT_SECONDS` | `30.0` | Provider call timeout in seconds |
| `STT_MAX_AUDIO_SIZE_MB` | `10.0` | Maximum accepted audio upload size in MB (10 MB by default; the hosted Whisper API accepts up to 25 MB) |

Examples:

```bash
# Hosted OpenAI Whisper (requires a key)
STT_API_KEY=sk-... STT_MODEL=whisper-1

# Local Whisper server (no key needed)
STT_BASE_URL=http://localhost:9000/v1 STT_MODEL=Systran/faster-whisper-large-v3

# Force English transcription
STT_LANGUAGE=en
```

**Architecture (app/stt/)**

```
app/stt/
├── __init__.py          # public re-exports
├── base.py              # BaseSTT ABC, STTProtocol, STTError, shared validators
├── types.py             # STTAudio, STTText, STTLanguage aliases
├── config.py            # STTConfig, STTProvider enum
├── models.py            # STTRequest, STTResponse (Pydantic v2)
├── validation.py        # size, extension, MIME, magic-byte checks
├── fake.py              # FakeSTT (tests / offline dev ONLY)
└── openai_whisper.py    # OpenAIWhisperSTT (production)
```

The `FakeSTT` provider exists exclusively for tests and offline
development. It is **never** returned by `get_stt()` in
`app/api/dependencies.py`; production wiring returns either a real
`OpenAIWhisperSTT` or `None` (when unconfigured).

**Supported audio formats (validated by magic bytes + extension)**

* `.wav` / `audio/wav` (RIFF/WAVE container)
* `.mp3` / `audio/mpeg` (ID3 tag or raw frame sync)
* `.m4a` / `audio/mp4` (MP4/ISO-BMFF container with `ftyp`)
* `.aac` / `audio/aac` (ADTS sync)
* `.webm` / `audio/webm` (EBML container)
* `.ogg` / `audio/ogg` (OggS container, includes Opus-in-Ogg)

MIME aliases (`audio/x-wav`, `audio/mp3`, `audio/x-m4a`, `audio/opus`,
parameters like `;codecs=opus`) are normalized to the canonical form.

**Language support**

* `language="en"` — English (explicit hint sent to the model)
* `language="hi"` — Hindi (explicit hint sent to the model)
* `language=None` — automatic language detection (no hint sent)
* Any ISO 639-1 two-letter code is accepted.

**Upload size and security limits**

* Default max audio size: **10 MB** (`STT_MAX_AUDIO_SIZE_MB`).
* Empty audio, oversize audio, unsupported extensions, declared MIME
  mismatches, and corrupt/unrecognized audio are all rejected
  *before* any provider call.
* Audio is validated strictly:
  1. Type/size bounds
  2. Supported file extension
  3. MIME type (aliases canonicalized, parameters stripped)
  4. Extension/MIME cross-check
  5. Container signature (magic-byte) sniffing
* The API key is **never** logged, returned in responses, or echoed
  in error messages — any occurrence in an SDK exception is
  replaced with `[REDACTED]`.

**Temporary-file cleanup**

* Audio is passed to the OpenAI SDK as an in-memory
  `(filename, bytes, content_type)` tuple (`file=` kwarg).
* **No temporary files are written to disk** by the STT layer.
* Audio bytes are never persisted; they exist only for the duration
  of the request.

**Local / custom OpenAI-compatible endpoint**

Set `STT_BASE_URL` to any OpenAI-compatible Whisper endpoint. The
provider does **not** require an API key when the URL is not the
default OpenAI URL — useful for self-hosted faster-whisper, vLLM
Whisper, or Ollama-style local servers.

```bash
STT_BASE_URL=http://localhost:9000/v1
STT_MODEL=Systran/faster-whisper-large-v3
# STT_API_KEY is optional for local servers
```

**What's NOT implemented yet (STT-adjacent)**

* `POST /api/voice-query` (the user-facing voice endpoint that
  would orchestrate STT → Guardrail → Retrieval → LLM → Grounding)
  is **not** wired into the FastAPI router yet.
* End-to-end Voice → STT → Guardrail → Retrieval → LLM → Grounding
  → TTS integration — not yet integrated.

The STT package itself is provider-complete and tested: it can
be invoked directly, the OpenAI Whisper adapter produces a
`STTResponse` over real audio once credentials are configured,
and `get_stt()` resolves a real `OpenAIWhisperSTT` from settings.

### TTS provider configuration

The TTS (text-to-speech) layer is implemented in `backend/app/tts/`
and is **provider-agnostic**: any concrete provider implementing the
`BaseTTS` contract can be wired in. The default and production provider
today is the OpenAI Text-to-Speech API (or any OpenAI-compatible TTS
endpoint such as local Kokoro, Piper, or vLLM TTS servers).

Configuration is read from environment variables (or `backend/.env`):

| Variable | Default | Purpose |
|---|---|---|
| `TTS_PROVIDER` | `openai_tts` | Provider selector. Anything other than `openai_tts` leaves the TTS layer unconfigured. |
| `TTS_API_KEY` | (none) | Provider API key. Required for the hosted OpenAI TTS API. May be omitted for local compatible servers. Falls back to `LLM_API_KEY` when unset. |
| `TTS_BASE_URL` | OpenAI API | Any OpenAI-compatible TTS endpoint. Falls back to `LLM_BASE_URL` when unset. Local endpoints need no API key. |
| `TTS_MODEL` | `tts-1` | Model identifier (`tts-1`, `tts-1-hd`, or custom endpoint model). |
| `TTS_VOICE` | `alloy` | Default voice (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`). |
| `TTS_OUTPUT_FORMAT` | `mp3` | Default output format (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`). |
| `TTS_SPEED` | `1.0` | Default speech playback speed multiplier (`0.25` to `4.0`). |
| `TTS_TIMEOUT_SECONDS` | `30.0` | Provider call timeout in seconds. |
| `TTS_MAX_TEXT_LENGTH` | `4096` | Maximum allowed character length for text inputs. |
| `TTS_MAX_AUDIO_SIZE_MB` | `10.0` | Maximum accepted generated audio size in MB. |

Examples:

```bash
# Hosted OpenAI TTS (requires a key)
TTS_API_KEY=sk-... TTS_MODEL=tts-1 TTS_VOICE=nova

# Local TTS server (no key needed)
TTS_BASE_URL=http://localhost:8080/v1 TTS_MODEL=kokoro-v0_19

# High quality audio with custom speed and WAV output
TTS_MODEL=tts-1-hd TTS_VOICE=shimmer TTS_OUTPUT_FORMAT=wav TTS_SPEED=1.25
```

**Architecture (app/tts/)**

```
app/tts/
├── __init__.py          # public re-exports
├── base.py              # BaseTTS ABC, TTSProtocol, TTSError, shared validators
├── types.py             # TTSText, TTSAudio, TTSVoice, TTSModel, TTSFormat aliases
├── config.py            # TTSConfig, TTSProvider enum
├── models.py            # TTSRequest, TTSResponse (Pydantic v2)
├── validation.py        # audio output validation, format checks, container sniffing
├── fake.py              # FakeTTS (tests / offline dev ONLY)
└── openai_tts.py        # OpenAITTS (production adapter via OpenAI SDK 1.x)
```

The `FakeTTS` provider exists exclusively for tests and offline
development. It is **never** returned by `get_tts()` in
`app/api/dependencies.py`; production wiring returns either a real
`OpenAITTS` or `None` (when unconfigured).

**Supported audio formats**

* `mp3` (`audio/mpeg`) — default compressed audio format
* `opus` (`audio/opus`) — low-latency internet streaming
* `aac` (`audio/aac`) — digital audio compression
* `flac` (`audio/flac`) — lossless audio compression
* `wav` (`audio/wav`) — uncompressed PCM in RIFF container
* `pcm` (`audio/pcm`) — raw 16-bit 24kHz audio samples

**Language & Voice capabilities**

* OpenAI TTS models (`tts-1`, `tts-1-hd`) natively synthesize English,
  Hindi, and dozens of other languages based on the input text.
* 6 built-in voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`.
* Variable speed from 0.25x to 4.0x.

**Security and Memory Bounds**

* Text input bounded: max length enforced (default: 4096 characters).
* Speed bounded: must be within [0.25, 4.0].
* Audio output bounded: maximum generated audio size enforced (default: 10 MB).
* Container signature verification: generated audio is verified with
  magic-byte container sniffing.
* Transient in-memory audio: synthesized audio bytes are handled in memory
  and never written to disk or logged.
* Credential protection: API keys are never logged, exposed in `__repr__`,
  or echoed in exceptions (redacted to `[REDACTED]`).

**Standalone Subsystem Status & Remaining Work**

* The TTS subsystem is implemented as a standalone library.
* `POST /api/voice-query` (the user-facing voice query endpoint) is **not**
  implemented yet.
* Full STT → Guardrail → Retrieval → LLM → Grounding → TTS pipeline
  integration is **not** implemented yet.

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

## Voice Query Pipeline (`POST /api/voice-query`)

The backend exposes a full production-quality voice question-answering pipeline connecting:

```
Uploaded Audio (wav, mp3, ogg, etc.)
  │
  ▼
[Audio Validation] (Size, MIME type, container magic-bytes)
  │
  ▼
[STT Transcription] (Whisper / OpenAI audio transcriptions)
  │
  ▼
[Input Guardrail] (Prompt injection & safety checks)
  │  ├── OFF_TOPIC_REJECTED ──► HTTP 400 (stops immediately; skips retrieval, LLM, grounding, TTS)
  │
  ▼
[Retrieval Orchestrator] (Embed -> Vector Search -> Chunk Resolver)
  │  ├── Index Unavailable ──► HTTP 503
  │
  ▼
[LLM Generation] (System prompt enforces grounded synthesis)
  │  ├── LLM Unconfigured ──► HTTP 501
  │
  ▼
[Grounding Verification] (Token overlap & numeric fact-checking against evidence)
  │  ├── UNGROUNDED_FLAGGED ──► HTTP 422 (DOES NOT synthesize audio)
  │
  ▼
[TTS Synthesis] (OpenAI TTS speech synthesis)
  │  ├── TTS Unconfigured ──► HTTP 501
  │
  ▼
[VoiceQueryResponse] (JSON with base64 audio, transcript, answer, citations, latencies)
```

### Request

`POST /api/voice-query` (`multipart/form-data`)

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `audio` | File | Yes | Audio file binary (`wav`, `mp3`, `ogg`, `m4a`, `aac`) |
| `language` | string | No | Optional language code hint (`en`, `hi`, etc.) |
| `voice` | string | No | Optional TTS voice override (`alloy`, `nova`, `echo`, `fable`, `onyx`, `shimmer`) |
| `speed` | float | No | Optional TTS speech speed multiplier (`0.25` to `4.0`, default `1.0`) |

### Response (`VoiceQueryResponse`)

```json
{
  "transcribed_text": "tell me about goa beaches on the west coast",
  "answer": "Goa has many scenic beaches situated along the Arabian Sea on the west coast...",
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "audio_content_type": "audio/mpeg",
  "audio_format": "mp3",
  "citations": [
    {
      "chunk_id": "passage-doc-0-0",
      "document_id": "doc-0",
      "score": 0.892,
      "text": "goa has many beaches on the west coast"
    }
  ],
  "guardrail": {
    "verdict": "SAFE_AND_GROUNDED",
    "reason": "Input passed all pre-generation safety and topic checks.",
    "score": 1.0,
    "flagged_claims": []
  },
  "grounding": {
    "verdict": "SAFE_AND_GROUNDED",
    "reason": "All claims in the generated answer are supported by retrieved evidence.",
    "score": 0.94,
    "flagged_claims": []
  },
  "model": "gpt-4o-mini",
  "stt_model": "whisper-1",
  "tts_model": "tts-1",
  "latency_breakdown": {
    "stt_ms": 320.5,
    "guardrail_ms": 1.2,
    "retrieval_ms": 14.8,
    "llm_ms": 450.1,
    "grounding_ms": 2.4,
    "tts_ms": 280.9,
    "total_pipeline_ms": 1070.1,
    "total_ms": 1070.1
  }
}
```

### Safety & Error Short-Circuits

* **Invalid/Oversized Audio (HTTP 400)**: Rejects corrupt headers, invalid extensions, or payload exceeding `STT_MAX_AUDIO_SIZE_MB`.
* **Guardrail Rejection (HTTP 400)**: If STT transcript is off-topic or prompt injection, execution stops immediately. Retrieval, LLM, Grounding, and TTS are never called.
* **Unconfigured Providers (HTTP 501)**: Missing `STT_API_KEY`, `LLM_API_KEY`, or `TTS_API_KEY` returns `STT_PROVIDER_NOT_CONFIGURED`, `LLM_PROVIDER_NOT_CONFIGURED`, or `TTS_PROVIDER_NOT_CONFIGURED`.
* **Missing Index (HTTP 503)**: Returns `INDEX_NOT_AVAILABLE` if the vector database index is not loaded.
* **Ungrounded Answer (HTTP 422)**: If grounding verifier flags hallucinated claims (`UNGROUNDED_FLAGGED`), TTS synthesis is aborted to prevent speaking untrustworthy output.

---

**Not yet implemented (will be added in later phases):**

* Frontend microphone capture and browser playback streaming.
* Multi-turn conversational voice context.
* Evaluation harness.
