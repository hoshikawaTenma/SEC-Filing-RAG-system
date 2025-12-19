# Filing RAG Workspace

This is a minimal scaffold that matches the architecture you described:

- Parsing-first ingestion → parents (pages/sections) + child chunks
- Dual retrieval (vector + keyword) → parent dedupe
- Rerank parents (LLM structured output or heuristic fallback)
- Answer from parent context with strict citations

## Structure

- `services/api`: FastAPI orchestrator
- `services/ingest`: ingestion + parsing
- `packages/core`: shared pipeline, models, routing, indexes
- `apps/web`: simple UI (upload, chat, sources)

## Quick start

1) Create a virtual environment and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Set your DashScope key (optional but recommended):

```bash
set DASHSCOPE_API_KEY=your_key_here
```

3) Run the API:

```bash
python -m uvicorn services.api.main:app --reload
```

4) Open the UI:

Open `apps/web/index.html` in your browser.

## Notes

- PDF parsing uses `pdfplumber` and stores per-page parents with chunk overlap.
- Table extraction uses `camelot` (lattice + stream). On Windows you may need Ghostscript installed for lattice mode. Tables are stored only as parent text for retrieval (no CSV persistence).
- Stored chunks, parents, and embeddings are reused on subsequent ingests to skip parsing and chunking when possible.
- HTML parsing uses a simple section split. Swap with a DOM parser for real filings.
- The pipeline uses DashScope if `DASHSCOPE_API_KEY` is set. Otherwise it falls back to a local hashing embedding and heuristic reranker.
- Parents and chunks are persisted as JSONL in `data/store` per document.
 - Rerank uses weighted scoring (vector 0.3 + LLM 0.7 by default) with structured JSON outputs.

## API routes

- `POST /upload`: upload a file (PDF/HTML/TXT)
- `POST /chat`: ask a question, returns answer + citations + parent IDs
- `GET /parents/{parent_id}`: fetch parent text for “show me where”
- `GET /docs`: list ingested filings
- `GET /health`: basic health check
