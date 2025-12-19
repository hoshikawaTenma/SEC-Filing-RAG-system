from __future__ import annotations

from datetime import datetime

from .models import Document


def route_documents(question: str, documents: list[Document]) -> list[str]:
    if len(documents) <= 1:
        return [doc.doc_id for doc in documents]
    q = question.lower()
    wants_compare = any(term in q for term in ["compare", "change", "difference", "changed"])
    wants_quarter = any(term in q for term in ["quarter", "q1", "q2", "q3", "q4"])
    if wants_compare:
        return [doc.doc_id for doc in _pick_latest(documents, count=2)]
    if wants_quarter:
        return [doc.doc_id for doc in _pick_latest(documents, count=1)]
    return [doc.doc_id for doc in _pick_latest(documents, count=1)]


def route_answer_type(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("where") or "where is" in q or "locate" in q:
        return "locate"
    if any(term in q for term in ["how much", "what is revenue", "amount", "number"]):
        return "numeric"
    if any(term in q for term in ["compare", "change", "difference"]):
        return "compare"
    if q.startswith("explain") or q.startswith("define") or "what does" in q:
        return "explain"
    return "explain"


def _pick_latest(documents: list[Document], count: int) -> list[Document]:
    def score(doc: Document) -> tuple[int, str]:
        date_str = doc.metadata.get("period_end") if doc.metadata else None
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                return (int(dt.timestamp()), doc.doc_id)
            except ValueError:
                pass
        return (0, doc.doc_id)

    return sorted(documents, key=score, reverse=True)[:count]
