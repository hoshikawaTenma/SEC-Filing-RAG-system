from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

import pdfplumber

try:
    import camelot
except Exception:
    camelot = None


@dataclass
class ExtractedTable:
    table_id: int
    page: int
    flavor: str
    bbox: Optional[tuple[float, float, float, float]]
    accuracy: Optional[float]
    whitespace: Optional[float]
    df: pd.DataFrame
    context_before: str
    context_after: str


def extract_tables(
    pdf_path: str, pages: str = "all", max_tables: int | None = None
) -> list[ExtractedTable]:
    if camelot is None:
        raise RuntimeError("camelot is not available")
    candidates: list[tuple[str, Any]] = []
    try:
        lattice_tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
        candidates.extend([("lattice", t) for t in lattice_tables])
    except Exception:
        pass
    try:
        stream_tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream")
        candidates.extend([("stream", t) for t in stream_tables])
    except Exception:
        pass
    if not candidates:
        return []

    best_by_hash: dict[str, dict[str, Any]] = {}
    for flavor, table in candidates:
        df = table.df
        if df is None or df.empty:
            continue
        sha1 = _hash_df(df)
        report = getattr(table, "parsing_report", {}) or {}
        acc = _safe_float(report.get("accuracy"))
        ws = _safe_float(report.get("whitespace"))
        bbox = _get_bbox(table)
        score = (acc if acc is not None else 0.0) + (0.01 if flavor == "lattice" else 0.0)
        prev = best_by_hash.get(sha1)
        if prev is None or score > prev["score"]:
            best_by_hash[sha1] = {
                "score": score,
                "flavor": flavor,
                "table": table,
                "df": df,
                "accuracy": acc,
                "whitespace": ws,
                "bbox": bbox,
            }

    chosen = list(best_by_hash.items())

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, str, str]:
        sha1, rec = item
        table = rec["table"]
        page = int(getattr(table, "page", 0) or 0)
        bbox = rec["bbox"]
        y_top = max(bbox[1], bbox[3]) if bbox else 0.0
        return (page, -y_top, rec["flavor"], sha1)

    chosen.sort(key=sort_key)
    if max_tables is not None:
        chosen = chosen[:max_tables]

    extracted: list[ExtractedTable] = []
    for idx, (_sha1, rec) in enumerate(chosen, start=1):
        table = rec["table"]
        page = int(getattr(table, "page", 0) or 0)
        bbox = rec["bbox"]
        context_before, context_after = _extract_context(pdf_path, page, bbox)
        extracted.append(
            ExtractedTable(
                table_id=idx,
                page=page,
                flavor=rec["flavor"],
                bbox=bbox,
                accuracy=rec["accuracy"],
                whitespace=rec["whitespace"],
                df=rec["df"],
                context_before=context_before,
                context_after=context_after,
            )
        )
    return extracted


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_df_for_hash(df: pd.DataFrame) -> str:
    tmp = df.copy()
    tmp = tmp.fillna("")
    tmp = tmp.astype(str).applymap(lambda s: " ".join(s.split()))
    return "\n".join(["\t".join(row) for row in tmp.values.tolist()]).strip()


def _hash_df(df: pd.DataFrame) -> str:
    norm = _normalize_df_for_hash(df)
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()


def _get_bbox(table: Any) -> Optional[tuple[float, float, float, float]]:
    for attr in ("_bbox", "bbox"):
        if hasattr(table, attr):
            try:
                bb = getattr(table, attr)
                if bb and len(bb) == 4:
                    return tuple(map(float, bb))
            except Exception:
                continue
    return None


def _extract_context(
    pdf_path: str,
    page_num_1indexed: int,
    bbox: Optional[tuple[float, float, float, float]],
    margin: float = 60.0,
    max_chars: int = 1200,
) -> tuple[str, str]:
    if bbox is None:
        return "", ""
    x1, y1, x2, y2 = bbox
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num_1indexed - 1]
            words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
            page_h = page.height
            y_bottom = min(y1, y2)
            y_top = max(y1, y2)
            table_top_from_top = page_h - y_top
            table_bottom_from_top = page_h - y_bottom
            before_words = [
                w
                for w in words
                if (w.get("bottom", 0) <= table_top_from_top)
                and (w.get("bottom", 0) >= table_top_from_top - margin)
            ]
            after_words = [
                w
                for w in words
                if (w.get("top", 0) >= table_bottom_from_top)
                and (w.get("top", 0) <= table_bottom_from_top + margin)
            ]
            return (
                _words_to_text(before_words)[:max_chars],
                _words_to_text(after_words)[:max_chars],
            )
    except Exception:
        return "", ""


def _words_to_text(words: list[dict[str, Any]]) -> str:
    words = sorted(words, key=lambda w: (round(w.get("top", 0), 1), w.get("x0", 0)))
    lines: list[str] = []
    current_y = None
    current_line: list[str] = []
    for word in words:
        y = round(word.get("top", 0), 1)
        if current_y is None or abs(y - current_y) <= 2:
            current_line.append(word.get("text", ""))
            current_y = y if current_y is None else current_y
        else:
            lines.append(" ".join(current_line).strip())
            current_line = [word.get("text", "")]
            current_y = y
    if current_line:
        lines.append(" ".join(current_line).strip())
    return "\n".join([line for line in lines if line]).strip()
