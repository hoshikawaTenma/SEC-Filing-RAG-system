from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from packages.core.dashscope_client import DashScopeClient
from packages.core.models import AnswerPayload
from packages.core.pipeline import PipelineDependencies, RagPipeline
from packages.core.routers import route_documents
from packages.core.storage import InMemoryStore, KeywordIndex, VectorIndex
from services.ingest.ingest import ingest_file, load_existing_document


app = FastAPI(title="Filing RAG Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
store = InMemoryStore()
keyword_index = KeywordIndex()
vector_index = VectorIndex()
dashscope = DashScopeClient()
pipeline = RagPipeline(
    PipelineDependencies(
        store=store,
        keyword_index=keyword_index,
        vector_index=vector_index,
        dashscope=dashscope,
    )
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"
ingestion_state = {"stage": "idle", "ready": False, "message": "No filings uploaded."}


class ChatRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None


class ChatResponse(BaseModel):
    answer: AnswerPayload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/docs")
def list_docs() -> list[dict]:
    return [doc.model_dump() for doc in store.list_documents()]


@app.get("/uploads")
def list_uploads() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return [path.name for path in DATA_DIR.iterdir() if path.is_file()]


@app.get("/status")
def get_status() -> dict:
    return ingestion_state


@app.get("/parents/{parent_id}")
def get_parent(parent_id: str) -> dict:
    parent = store.get_parent(parent_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")
    return parent.model_dump()


def _run_ingestion(path: str) -> None:
    def set_stage(stage: str) -> None:
        ingestion_state.update(
            {"stage": stage, "ready": False, "message": f"{stage.capitalize()}..."}
        )

    try:
        ingest_file(
            path,
            store,
            keyword_index,
            vector_index,
            dashscope,
            status_cb=set_stage,
        )
    except Exception as exc:
        ingestion_state.update(
            {"stage": "error", "ready": False, "message": str(exc)}
        )
        return
    ingestion_state.update(
        {"stage": "ready", "ready": True, "message": "You can ask questions now."}
    )


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks, file: Annotated[UploadFile, File(...)]
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DATA_DIR / file.filename
    content = await file.read()
    target_path.write_bytes(content)
    ingestion_state.update(
        {
            "stage": "queued",
            "ready": False,
            "message": "Upload received. Starting ingestion.",
        }
    )
    background_tasks.add_task(_run_ingestion, str(target_path))
    return {"status": "queued", "filename": file.filename}


class IngestExistingRequest(BaseModel):
    filename: str


@app.post("/ingest_existing")
def ingest_existing(
    request: IngestExistingRequest, background_tasks: BackgroundTasks
) -> dict:
    target_path = DATA_DIR / request.filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    ingestion_state.update(
        {
            "stage": "loading",
            "ready": False,
            "message": "Checking stored chunks.",
        }
    )
    document = load_existing_document(
        request.filename,
        store,
        keyword_index,
        vector_index,
        dashscope,
        status_cb=lambda stage: ingestion_state.update(
            {
                "stage": stage,
                "ready": False,
                "message": f"{stage.replace('_', ' ').capitalize()}...",
            }
        ),
    )
    if document:
        ingestion_state.update(
            {"stage": "ready", "ready": True, "message": "You can ask questions now."}
        )
        return {"status": "loaded", "filename": request.filename}
    ingestion_state.update(
        {
            "stage": "queued",
            "ready": False,
            "message": "Starting ingestion from existing file.",
        }
    )
    background_tasks.add_task(_run_ingestion, str(target_path))
    return {"status": "queued", "filename": request.filename}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    docs = store.list_documents()
    if not docs:
        raise HTTPException(status_code=400, detail="Upload a filing first.")
    doc_ids = request.doc_ids or route_documents(request.question, docs)
    answer = pipeline.run(request.question, doc_ids)
    return ChatResponse(answer=answer)
