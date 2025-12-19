from __future__ import annotations


def simple_tokenize(text: str) -> list[str]:
    return [t for t in text.replace("\n", " ").split(" ") if t.strip()]


def chunk_text(
    text: str, max_tokens: int, overlap_tokens: int = 0
) -> list[tuple[str, int, int]]:
    tokens = simple_tokenize(text)
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_value = " ".join(chunk_tokens)
        chunks.append((chunk_text_value, start, end))
        if end == len(tokens):
            break
        start = max(end - overlap_tokens, 0)
    return chunks
