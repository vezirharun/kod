"""Metin eşleşme skoru — eşleşme yoksa 0.0."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_TOKEN_SPLIT = re.compile(r"[^a-z0-9çğıöşü]+", re.IGNORECASE)


def tokenize_name(text: str) -> set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_SPLIT.split(text.lower()) if len(t) >= 2}


def _tokens_match(query_tokens: set[str], haystack_tokens: set[str]) -> bool:
    if not query_tokens or not haystack_tokens:
        return False
    return bool(query_tokens & haystack_tokens)


def text_match_score(query: str, rec: dict[str, Any]) -> float:
    """
    Eşleşme yoksa 0.0.
    Dosya adı token eşleşmesi yüksek, klasör düşük, OCR yüksek, müşteri düşük.
    """
    q = (query or "").strip().lower()
    if not q:
        return 0.0

    q_tokens = tokenize_name(q)
    fname = (rec.get("filename") or "").lower()
    path = (rec.get("path") or "").lower()
    ocr = (rec.get("ocr_text") or "").lower()
    customer = (rec.get("customer") or "").lower()
    source = (rec.get("source_name") or "").lower()

    fname_tokens = tokenize_name(Path(fname).stem)
    folder_part = Path(path).parent.name.lower()
    folder_tokens = tokenize_name(folder_part)

    # Tam dosya adı (uzantısız)
    stem = Path(fname).stem.lower()
    if q == stem or q.replace(" ", "_") == stem:
        return 0.95

    # Token tam eşleşme — dosya adı
    if q_tokens and _tokens_match(q_tokens, fname_tokens):
        overlap = len(q_tokens & fname_tokens) / max(len(q_tokens), 1)
        if overlap >= 0.99:
            return 0.92
        return 0.82 + 0.08 * overlap

    # Alt string dosya adında
    if q in fname:
        return 0.85

    # OCR + marka alias
    from core.brand_aliases import brand_match_score, enrich_ocr_text

    ocr_enriched = enrich_ocr_text(ocr).lower() if ocr else ""
    if q in ocr or (q_tokens and _tokens_match(q_tokens, tokenize_name(ocr))):
        return 0.98
    if ocr_enriched and (
        q in ocr_enriched or (q_tokens and _tokens_match(q_tokens, tokenize_name(ocr_enriched)))
    ):
        return 0.98
    brand = brand_match_score(q, [ocr, fname])
    if brand > 0:
        return brand

    # Müşteri / kaynak — düşük boost
    if q in customer or (q_tokens and _tokens_match(q_tokens, tokenize_name(customer))):
        return 0.55
    if q in source or (q_tokens and _tokens_match(q_tokens, tokenize_name(source))):
        return 0.50

    # Klasör adı — düşük (içindeki her dosyayı leopard sayma)
    if q in folder_part or (q_tokens and _tokens_match(q_tokens, folder_tokens)):
        return 0.38

    # Tam path alt string — en düşük
    if q in path and q not in fname:
        return 0.32

    return 0.0


def filename_text_boost_only(query: str, rec: dict[str, Any]) -> float:
    """Sadece dosya adı — klasör path boost yok."""
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    fname = (rec.get("filename") or "").lower()
    q_tokens = tokenize_name(q)
    fname_tokens = tokenize_name(Path(fname).stem)
    if q_tokens and _tokens_match(q_tokens, fname_tokens):
        return 0.88
    if q in fname:
        return 0.80
    return 0.0
