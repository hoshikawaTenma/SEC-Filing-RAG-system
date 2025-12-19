from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .chunking import simple_tokenize
from .config import settings
from .dashscope_client import DashScopeClient, local_hash_embedding
from .models import (
    AnswerPayload,
    AnswerSentence,
    Citation,
    ParentBlock,
    ParentCandidate,
    RetrievalChunkHit,
    RerankResult,
)
from .routers import route_answer_type
from .storage import InMemoryStore, KeywordIndex, VectorIndex, cosine_similarity


@dataclass
class PipelineDependencies:
    store: InMemoryStore
    keyword_index: KeywordIndex
    vector_index: VectorIndex
    dashscope: DashScopeClient


class RagPipeline:
    def __init__(self, deps: PipelineDependencies) -> None:
        self.deps = deps

    def run(self, question: str, doc_ids: list[str]) -> AnswerPayload:
        answer_type = route_answer_type(question)
        candidates = self.retrieve_candidates(question, doc_ids, answer_type=answer_type)
        reranked = self.rerank_parents(question, candidates, answer_type=answer_type)
        parents = self.pick_top_parents(reranked)
        return self.answer_from_parents(question, answer_type, parents)

    def retrieve_candidates(
        self,
        question: str,
        doc_ids: list[str],
        answer_type: str,
        vector_top_k: int = 30,
        keyword_top_k: int = 20,
    ) -> list[ParentCandidate]:
        if answer_type == "numeric":
            vector_top_k = max(vector_top_k, 40)
            keyword_top_k = max(keyword_top_k, 30)
        keyword_hits = self.deps.keyword_index.search(question, top_k=keyword_top_k)
        vector_hits = self._vector_search(question, top_k=vector_top_k)
        parent_map: dict[str, ParentCandidate] = {}
        for hit in keyword_hits + vector_hits:
            chunk = self.deps.store.get_chunk(hit.chunk_id)
            if not chunk or chunk.doc_id not in doc_ids:
                continue
            candidate = parent_map.get(chunk.parent_id)
            if not candidate:
                candidate = ParentCandidate(
                    parent_id=chunk.parent_id,
                    score=0.0,
                    vector_score=0.0,
                    keyword_score=0.0,
                    chunk_hits=[],
                )
                parent_map[chunk.parent_id] = candidate
            candidate.score += hit.score
            candidate.chunk_hits.append(hit)
            if hit.source == "vector":
                candidate.vector_score = max(candidate.vector_score, hit.score)
            if hit.source == "keyword":
                candidate.keyword_score += hit.score
        return list(parent_map.values())

    def rerank_parents(
        self, question: str, parents: list[ParentCandidate], answer_type: str
    ) -> list[RerankResult]:
        if not parents:
            return []
        vector_scores = {p.parent_id: p.vector_score for p in parents}
        max_vector = max(vector_scores.values()) or 1.0
        if self.deps.dashscope.is_configured():
            try:
                reranked = self._rerank_with_llm(question, parents)
            except Exception:
                reranked = self._rerank_fallback(parents)
        else:
            reranked = self._rerank_fallback(parents)
        for item in reranked:
            vector_score = vector_scores.get(item.parent_id, 0.0) / max_vector
            combined = settings.vector_weight * vector_score + settings.llm_weight * item.relevance_score
            if answer_type == "numeric":
                parent = self.deps.store.get_parent(item.parent_id)
                if parent and parent.parent_type == "table":
                    combined += 0.15
                if parent and _looks_numeric(parent.text):
                    combined += 0.05
            item.combined_score = combined
        return reranked

    def pick_top_parents(self, reranked: list[RerankResult]) -> list[ParentBlock]:
        chosen = sorted(
            reranked,
            key=lambda item: item.combined_score
            if item.combined_score is not None
            else item.relevance_score,
            reverse=True,
        )[
            : settings.top_k_parents
        ]
        parents: list[ParentBlock] = []
        for item in chosen:
            parent = self.deps.store.get_parent(item.parent_id)
            if parent:
                parents.append(parent)
        return parents

    def answer_from_parents(
        self, question: str, answer_type: str, parents: list[ParentBlock]
    ) -> AnswerPayload:
        if self.deps.dashscope.is_configured() and parents:
            try:
                return self._answer_with_llm(question, answer_type, parents)
            except Exception:
                pass
        return self._fallback_answer(answer_type, parents)

    def _vector_search(self, question: str, top_k: int) -> list[RetrievalChunkHit]:
        if self.deps.dashscope.is_configured():
            embedding = self.deps.dashscope.embed([question])[0]
        else:
            embedding = local_hash_embedding(question)
        chunk_ids = self.deps.vector_index.search(embedding, top_k=top_k)
        hits: list[RetrievalChunkHit] = []
        for chunk_id in chunk_ids:
            chunk = self.deps.store.get_chunk(chunk_id)
            if not chunk:
                continue
            chunk_embedding = self.deps.vector_index.embeddings.get(chunk_id, [])
            score = cosine_similarity(embedding, chunk_embedding)
            hits.append(RetrievalChunkHit(chunk_id=chunk_id, score=score, source="vector"))
        return hits

    def _rerank_with_llm(
        self, question: str, parents: list[ParentCandidate]
    ) -> list[RerankResult]:
        parent_payloads = []
        for parent in parents:
            block = self.deps.store.get_parent(parent.parent_id)
            if not block:
                continue
            preview = " ".join(block.text.split()[: settings.max_parent_tokens])
            parent_payloads.append(
                {
                    "parent_id": parent.parent_id,
                    "title": block.title,
                    "page_number": block.page_number,
                    "section_path": block.section_path,
                    "text": preview,
                }
            )
        system_prompt = (
            "You are a reranker. Return JSON list of objects with fields: "
            "parent_id, relevance_score (0-1), why_relevant, support_type."
        )
        user_prompt = (
            f"Question: {question}\nParents:\n{parent_payloads}\nReturn JSON only."
        )
        data = self.deps.dashscope.chat_json_with_repair(system_prompt, user_prompt)
        results: list[RerankResult] = []
        for item in data:
            results.append(RerankResult(**item))
        return results

    def _rerank_fallback(self, parents: list[ParentCandidate]) -> list[RerankResult]:
        if not parents:
            return []
        max_score = max(parent.score for parent in parents) or 1.0
        results: list[RerankResult] = []
        for parent in parents:
            results.append(
                RerankResult(
                    parent_id=parent.parent_id,
                    relevance_score=min(parent.score / max_score, 1.0),
                    why_relevant="Overlap with query terms.",
                    support_type="explanation",
                )
            )
        return results

    def _first_sentence(self, text: str) -> str:
        if not text:
            return ""
        for sentence in text.replace("\n", " ").split("."):
            sentence = sentence.strip()
            if len(sentence.split()) > 4:
                return sentence + "."
        tokens = simple_tokenize(text)
        return " ".join(tokens[:25]) + ("..." if len(tokens) > 25 else "")

    def _answer_with_llm(
        self, question: str, answer_type: str, parents: list[ParentBlock]
    ) -> AnswerPayload:
        parent_payloads = []
        selected_parents = parents[:3]
        for parent in selected_parents:
            preview = " ".join(parent.text.split()[: settings.max_parent_tokens])
            parent_payloads.append(
                {
                    "parent_id": parent.parent_id,
                    "page_number": parent.page_number,
                    "section_path": parent.section_path,
                    "title": parent.title,
                    "text": preview,
                }
            )
        system_prompt = (
            "You answer using only the provided parents. "
            "You MUST answer directly using only the provided context. "
            "Do not answer with only page references. "
            "Return JSON with keys: answer, evidence, confidence, not_found_reason. "
            "answer: 1-3 short sentences. "
            "evidence: list of 1-5 items with parent_id, page_number, section_path, quote. "
            "confidence: high|medium|low. "
            "If not answerable, set answer to 'Not found in provided context' "
            "and fill not_found_reason."
        )
        instructions = _prompt_template(answer_type, question)
        user_prompt = (
            f"{instructions}\n\nQuestion: {question}\nParents:\n{parent_payloads}\n"
            "Return JSON only."
        )
        data = self._generate_llm_answer(system_prompt, user_prompt)
        payload = self._map_llm_output(data, selected_parents)
        if self._should_retry_llm(payload):
            retry_prompt = (
                "Your previous response did not include a direct answer. "
                "Rewrite with a concise, direct answer first, then evidence. "
                "Return JSON only."
            )
            retry_user_prompt = (
                f"{retry_prompt}\n\nQuestion: {question}\nParents:\n{parent_payloads}\n"
                "Return JSON only."
            )
            data = self._generate_llm_answer(system_prompt, retry_user_prompt)
            payload = self._map_llm_output(data, selected_parents)
        if not self._validate_answer(payload, selected_parents):
            raise RuntimeError("LLM answer failed validation.")
        return payload

    def _validate_answer(self, payload: AnswerPayload, parents: list[ParentBlock]) -> bool:
        parent_ids = {parent.parent_id for parent in parents}
        if not payload.sentences:
            return False
        if payload.answer.strip().lower().startswith("not found"):
            return True
        for sentence in payload.sentences:
            if not sentence.citations:
                return False
            for citation in sentence.citations:
                if citation.parent_id not in parent_ids:
                    return False
        return True

    def _generate_llm_answer(self, system_prompt: str, user_prompt: str) -> dict:
        return self.deps.dashscope.chat_json_with_repair(system_prompt, user_prompt)

    def _map_llm_output(
        self, data: dict, parents: list[ParentBlock]
    ) -> AnswerPayload:
        answer_text = data.get("answer", "")
        evidence = data.get("evidence") or []
        citations: list[Citation] = []
        parent_ids = {parent.parent_id for parent in parents}
        for item in evidence:
            parent_id = item.get("parent_id")
            if parent_id not in parent_ids:
                continue
            citations.append(
                Citation(
                    parent_id=parent_id,
                    quote=item.get("quote", ""),
                    page_number=item.get("page_number"),
                    section_path=item.get("section_path"),
                )
            )
        if not citations and parents:
            parent = parents[0]
            citations.append(
                Citation(
                    parent_id=parent.parent_id,
                    quote=self._first_sentence(parent.text),
                    page_number=parent.page_number,
                    section_path=parent.section_path,
                )
            )
        sentence = AnswerSentence(text=answer_text, citations=citations)
        return AnswerPayload(
            answer=answer_text,
            sentences=[sentence],
            citations=citations,
            parents_used=[c.parent_id for c in citations],
            confidence=data.get("confidence"),
            not_found_reason=data.get("not_found_reason"),
        )

    def _should_retry_llm(self, payload: AnswerPayload) -> bool:
        if not payload.answer:
            return True
        if payload.answer.strip().lower().startswith("not found"):
            return False
        if _looks_like_page_dump(payload.answer):
            return True
        if len(payload.answer.split()) < 8:
            return True
        return False

    def _fallback_answer(
        self, answer_type: str, parents: list[ParentBlock]
    ) -> AnswerPayload:
        sentences: list[AnswerSentence] = []
        citations: list[Citation] = []
        if not parents:
            return AnswerPayload(
                answer="No relevant filing content was found.",
                sentences=[],
                citations=[],
                parents_used=[],
            )
        selected = parents[:2] if answer_type != "locate" else parents[:3]
        for parent in selected:
            snippet = parent.text.strip().replace("\n", " ")
            snippet = snippet[:260] + ("..." if len(snippet) > 260 else "")
            label = parent.title or parent.section_path or "Relevant section"
            page = f"p.{parent.page_number}" if parent.page_number else ""
            if answer_type == "locate":
                sentence_text = f"{label} {page}".strip() + "."
            elif answer_type == "numeric":
                line = _best_numeric_line(parent.text) or self._first_sentence(parent.text)
                sentence_text = line.strip()
            else:
                sentence_text = self._first_sentence(parent.text)
            if not sentence_text.endswith("."):
                sentence_text += "."
            citation = Citation(
                parent_id=parent.parent_id,
                quote=snippet,
                page_number=parent.page_number,
                section_path=parent.section_path,
            )
            sentences.append(AnswerSentence(text=sentence_text, citations=[citation]))
            citations.append(citation)
        answer_text = " ".join(sentence.text for sentence in sentences)
        return AnswerPayload(
            answer=answer_text,
            sentences=sentences,
            citations=citations,
            parents_used=[parent.parent_id for parent in selected],
        )


def _prompt_template(answer_type: str, question: str) -> str:
    if answer_type == "locate":
        return (
            "Return citations first. Respond with where the term appears, "
            "using short quotes and page/section references. "
            "Use short sentences only."
        )
    if answer_type == "numeric":
        return (
            "Prefer numeric values, specify units, and cite table or page. "
            "If multiple numbers exist, pick the latest context."
        )
    if answer_type == "compare":
        return (
            "Compare across provided parents, highlight differences, "
            "and cite both sides. Use short sentences only."
        )
    return (
        "Explain the concept in 2-4 short sentences (<=30 words each). "
        "Do not paste long excerpts; keep quotes short. "
        "Cite each sentence."
    )


def _looks_numeric(text: str) -> bool:
    if not text:
        return False
    digits = sum(ch.isdigit() for ch in text)
    if digits >= 20:
        return True
    if "$" in text or "%" in text:
        return True
    return False


def _looks_like_page_dump(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    if "see page" in lowered and len(lowered.split()) < 25:
        return True
    tokens = [token for token in lowered.split() if token.isalpha()]
    bad_words = {"see", "page", "relevant", "filing", "language"}
    content_words = [token for token in tokens if token not in bad_words]
    return len(content_words) < 8


def _best_numeric_line(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    best_line = ""
    best_score = 0
    for line in lines:
        digit_count = sum(ch.isdigit() for ch in line)
        score = digit_count
        if "$" in line:
            score += 5
        if "%" in line:
            score += 3
        if score > best_score:
            best_score = score
            best_line = line
    return best_line[:220]
