from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

from packages.core.chunking import chunk_text
from packages.core.config import settings
from packages.core.dashscope_client import DashScopeClient, local_hash_embedding
from packages.core.models import ChildChunk, Document, ParentBlock
from packages.core.storage import InMemoryStore, KeywordIndex, VectorIndex
from services.ingest.table_extract import extract_tables


def ingest_file(
    file_path: str,
    store: InMemoryStore,
    keyword_index: KeywordIndex,
    vector_index: VectorIndex,
    dashscope: DashScopeClient,
    status_cb: Callable[[str], None] | None = None,
) -> Document:
    path = Path(file_path)
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    doc_type = _detect_doc_type(path.suffix.lower())
    document = Document(
        doc_id=doc_id,
        title=path.name,
        source_path=str(path),
        doc_type=doc_type,
        metadata={},
    )
    store.add_document(document)
    _persist_document(document)
    if status_cb:
        status_cb("parsing")
    parent_blocks = _parse_document(doc_id, path, doc_type, status_cb=status_cb)
    _persist_parents(doc_id, parent_blocks)
    for parent in parent_blocks:
        store.add_parent(parent)
        if status_cb:
            status_cb("chunking")
        chunks = _chunk_parent(parent, doc_id)
        for chunk in chunks:
            store.add_chunk(chunk)
        _persist_chunks(doc_id, chunks)
        if status_cb:
            status_cb("indexing")
        keyword_index.add_chunks(chunks)
        _index_embeddings(chunks, vector_index, dashscope)
    return document


def _detect_doc_type(ext: str) -> str:
    if ext == ".pdf":
        return "pdf"
    if ext in [".html", ".htm"]:
        return "html"
    if ext in [".txt", ".md"]:
        return "txt"
    return "other"


def _parse_document(
    doc_id: str, path: Path, doc_type: str, status_cb: Callable[[str], None] | None
) -> list[ParentBlock]:
    if doc_type == "txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return _split_into_sections(doc_id, text)
    if doc_type == "html":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return _split_into_sections(doc_id, text, parent_type="section")
    if doc_type == "pdf":
        return _parse_pdf_pages(doc_id, path, status_cb=status_cb)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _split_into_sections(doc_id, text, parent_type="other")


def _parse_pdf_pages(doc_id: str, path: Path, status_cb: Callable[[str], None] | None) -> list[ParentBlock]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required for PDF parsing. Install it and retry."
        ) from exc
    parents: list[ParentBlock] = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            parents.append(
                ParentBlock(
                    parent_id=f"parent_{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    parent_type="page",
                    title=f"Page {idx}",
                    page_number=idx,
                    text=text,
                )
            )
    if status_cb:
        status_cb("extracting_tables")
    parents.extend(_extract_table_parents(doc_id, path))
    return parents


def _split_into_sections(
    doc_id: str, text: str, parent_type: str = "section"
) -> list[ParentBlock]:
    sections = [section.strip() for section in text.split("\n\n") if section.strip()]
    parents: list[ParentBlock] = []
    for idx, section in enumerate(sections, start=1):
        parents.append(
            ParentBlock(
                parent_id=f"parent_{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                parent_type=parent_type,
                title=f"Section {idx}",
                section_path=f"Section {idx}",
                text=section,
            )
        )
    return parents


def _chunk_parent(parent: ParentBlock, doc_id: str) -> list[ChildChunk]:
    chunks = chunk_text(
        parent.text,
        settings.max_chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    results: list[ChildChunk] = []
    for text, start, end in chunks:
        results.append(
            ChildChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                parent_id=parent.parent_id,
                text=text,
                offset_start=start,
                offset_end=end,
            )
        )
    return results


def _index_embeddings(
    chunks: list[ChildChunk],
    vector_index: VectorIndex,
    dashscope: DashScopeClient,
) -> None:
    if not chunks:
        return
    if dashscope.is_configured():
        embeddings = dashscope.embed([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            vector_index.add_embedding(chunk.chunk_id, embedding)
        _persist_embeddings(chunks, embeddings)
    else:
        for chunk in chunks:
            embedding = local_hash_embedding(chunk.text)
            vector_index.add_embedding(chunk.chunk_id, embedding)
            _persist_embeddings([chunk], [embedding])


def _persist_parents(doc_id: str, parents: list[ParentBlock]) -> None:
    if not parents:
        return
    store_dir = Path(__file__).resolve().parents[2] / "data" / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    target = store_dir / f"{doc_id}_parents.jsonl"
    with target.open("a", encoding="utf-8") as handle:
        for parent in parents:
            handle.write(json.dumps(parent.model_dump(), ensure_ascii=True) + "\n")


def _persist_chunks(doc_id: str, chunks: list[ChildChunk]) -> None:
    if not chunks:
        return
    store_dir = Path(__file__).resolve().parents[2] / "data" / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    target = store_dir / f"{doc_id}_chunks.jsonl"
    with target.open("a", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=True) + "\n")


def _persist_embeddings(chunks: list[ChildChunk], embeddings: list[list[float]]) -> None:
    if not chunks:
        return
    store_dir = Path(__file__).resolve().parents[2] / "data" / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    doc_id = chunks[0].doc_id
    target = store_dir / f"{doc_id}_embeddings.jsonl"
    with target.open("a", encoding="utf-8") as handle:
        for chunk, embedding in zip(chunks, embeddings):
            payload = {"chunk_id": chunk.chunk_id, "embedding": embedding}
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _persist_document(document: Document) -> None:
    store_dir = Path(__file__).resolve().parents[2] / "data" / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    target = store_dir / "documents.jsonl"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(document.model_dump(), ensure_ascii=True) + "\n")


def load_existing_document(
    filename: str,
    store: InMemoryStore,
    keyword_index: KeywordIndex,
    vector_index: VectorIndex,
    dashscope: DashScopeClient,
    status_cb: Callable[[str], None] | None = None,
) -> Document | None:
    store_dir = Path(__file__).resolve().parents[2] / "data" / "store"
    documents_path = store_dir / "documents.jsonl"
    if not documents_path.exists():
        return None
    candidates: list[Document] = []
    with documents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc = Document(**json.loads(line))
            if doc.title == filename or doc.source_path.endswith(filename):
                candidates.append(doc)
    if not candidates:
        return None
    document = candidates[-1]
    parents_path = store_dir / f"{document.doc_id}_parents.jsonl"
    chunks_path = store_dir / f"{document.doc_id}_chunks.jsonl"
    embeddings_path = store_dir / f"{document.doc_id}_embeddings.jsonl"
    if not parents_path.exists() or not chunks_path.exists():
        return None
    if status_cb:
        status_cb("loading_parents")
    with parents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parent = ParentBlock(**json.loads(line))
            store.add_parent(parent)
    if status_cb:
        status_cb("loading_chunks")
    chunks: list[ChildChunk] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = ChildChunk(**json.loads(line))
            store.add_chunk(chunk)
            chunks.append(chunk)
    keyword_index.add_chunks(chunks)
    embeddings_map: dict[str, list[float]] = {}
    if embeddings_path.exists():
        with embeddings_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                embeddings_map[data["chunk_id"]] = data["embedding"]
    if embeddings_map:
        for chunk in chunks:
            embedding = embeddings_map.get(chunk.chunk_id)
            if embedding:
                vector_index.add_embedding(chunk.chunk_id, embedding)
    else:
        _index_embeddings(chunks, vector_index, dashscope)
    store.add_document(document)
    return document


def _extract_table_parents(doc_id: str, path: Path) -> list[ParentBlock]:
    try:
        tables = extract_tables(str(path))
    except Exception:
        return []
    parents: list[ParentBlock] = []
    for table in tables:
        csv_preview = table.df.to_csv(index=False)
        text_parts = [
            f"Table {table.table_id} (page {table.page})",
            table.context_before.strip(),
            csv_preview.strip(),
            table.context_after.strip(),
        ]
        text = "\n".join([part for part in text_parts if part])
        parents.append(
            ParentBlock(
                parent_id=f"parent_{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                parent_type="table",
                title=f"Table {table.table_id}",
                page_number=table.page,
                section_path=f"Table {table.table_id}",
                text=text,
            )
        )
    return parents
