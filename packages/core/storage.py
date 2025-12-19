from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import ChildChunk, Document, ParentBlock, RetrievalChunkHit


def normalize(text: str) -> list[str]:
    clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [t for t in clean.split() if t]


class InMemoryStore:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.parents: dict[str, ParentBlock] = {}
        self.chunks: dict[str, ChildChunk] = {}

    def add_document(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc

    def add_parent(self, parent: ParentBlock) -> None:
        self.parents[parent.parent_id] = parent

    def add_chunk(self, chunk: ChildChunk) -> None:
        self.chunks[chunk.chunk_id] = chunk

    def list_documents(self) -> list[Document]:
        return list(self.documents.values())

    def get_parent(self, parent_id: str) -> ParentBlock | None:
        return self.parents.get(parent_id)

    def get_chunk(self, chunk_id: str) -> ChildChunk | None:
        return self.chunks.get(chunk_id)


class KeywordIndex:
    def __init__(self) -> None:
        self.inverted: dict[str, set[str]] = defaultdict(set)
        self.chunk_text: dict[str, str] = {}

    def add_chunks(self, chunks: Iterable[ChildChunk]) -> None:
        for chunk in chunks:
            self.chunk_text[chunk.chunk_id] = chunk.text
            for term in set(normalize(chunk.text)):
                self.inverted[term].add(chunk.chunk_id)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalChunkHit]:
        terms = normalize(query)
        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            for chunk_id in self.inverted.get(term, set()):
                scores[chunk_id] += 1.0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalChunkHit(chunk_id=chunk_id, score=score, source="keyword")
            for chunk_id, score in ranked
        ]


class VectorIndex:
    def __init__(self) -> None:
        self.embeddings: dict[str, list[float]] = {}

    def add_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        self.embeddings[chunk_id] = embedding

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[str]:
        scores: list[tuple[str, float]] = []
        for chunk_id, embedding in self.embeddings.items():
            score = cosine_similarity(query_embedding, embedding)
            scores.append((chunk_id, score))
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        return [chunk_id for chunk_id, _score in ranked]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
