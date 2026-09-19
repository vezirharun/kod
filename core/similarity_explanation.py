"""Benzerlik açıklaması — skor + neden listesi."""

from __future__ import annotations

from typing import Any


def _ocr_hit(query_ocr: str, cand_ocr: str, breakdown: dict[str, float]) -> str:
    ocr_score = float(breakdown.get("ocr_score") or breakdown.get("ocr_text") or 0)
    if ocr_score < 0.5:
        return ""
    cand = (cand_ocr or "").strip()
    if not cand:
        return ""
    # Kısa logo özeti
    tokens = [t for t in cand.replace("\n", " ").split() if len(t) >= 2][:3]
    label = " ".join(tokens) if tokens else cand[:24]
    return f"OCR: {label.upper()}"


def build_match_explanations(
    *,
    score_percent: float = 0.0,
    debug: dict[str, Any] | None = None,
    text_breakdown: dict[str, float] | None = None,
    text_reasons: list[str] | None = None,
    query_ocr: str = "",
    cand_ocr: str = "",
    text_mode: bool = False,
) -> list[str]:
    """
    Kullanıcıya gösterilecek ✓ satırları.
    Örn: Aynı Pattern DNA, Aynı Repeat, OCR: LV
    """
    d = debug or {}
    tb = text_breakdown or {}
    lines: list[str] = []

    dna_match = d.get("dna_match_fields") or {}
    if isinstance(dna_match, dict):
        if float(d.get("dna_score", 0) or 0) >= 0.55 or sum(1 for v in dna_match.values() if v) >= 2:
            lines.append("Aynı Pattern DNA")
        if dna_match.get("Repeat") or d.get("repeat_match") or float(d.get("repeat_score", 0) or 0) >= 0.50:
            lines.append("Aynı Repeat")
        if dna_match.get("Motif") or d.get("motif_match") or float(d.get("motif_score", 0) or 0) >= 0.50:
            lines.append("Aynı Motif")
        if dna_match.get("Style"):
            lines.append("Aynı Stil")
        if dna_match.get("Semantic") or d.get("semantic_match"):
            lines.append("Aynı Semantic Etiket")

    pal = float(d.get("palette_similarity", d.get("palette_score", 0)) or 0)
    if pal >= 0.72 or d.get("same_color_family"):
        lines.append("Aynı Renk Paleti")
    elif float(d.get("color_score", 0) or 0) >= 0.55:
        lines.append("Benzer Renk")

    # OCR
    ocr_line = _ocr_hit(query_ocr, cand_ocr, {**d, **tb})
    if ocr_line:
        lines.append(ocr_line)
    elif float(tb.get("ocr_score", 0) or 0) >= 0.70:
        lines.append("OCR eşleşmesi")

    if float(tb.get("brand_alias_score", 0) or 0) >= 0.80:
        lines.append("Marka alias eşleşmesi")
    if float(tb.get("nl_score", 0) or 0) >= 0.65:
        for r in (text_reasons or []):
            if r.startswith("Marka:") or r.startswith("Renk:") or r.startswith("Motif:"):
                lines.append(r.replace(":", " eşleşmesi", 1))
                break
    if float(tb.get("auto_tag_score", 0) or 0) >= 0.80:
        lines.append("AI otomatik etiket")

    if not text_mode:
        if float(d.get("patch_score", 0) or 0) >= 0.70 or d.get("patch_match"):
            lines.append("Benzer patch/tile yapısı")
        if d.get("protected_exact") or d.get("is_self_match"):
            lines.append("Aynı dosya")
        if float(d.get("phash_score", 0) or 0) >= 0.90:
            lines.append("Aynı görsel yapı")
        if d.get("same_collection"):
            lines.append("Aynı Koleksiyon")
        if d.get("same_series"):
            lines.append("Aynı Seri")

    if text_mode and text_reasons:
        for r in text_reasons[:3]:
            norm = r.strip()
            if norm and norm not in lines and not any(norm in x for x in lines):
                if "eşleşmesi" in norm or norm.startswith("OCR"):
                    lines.append(norm)
                elif ":" in norm:
                    lines.append(norm)

    # Tekilleştir, sırayı koru
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            out.append(line)
    if not out and score_percent >= 50:
        out.append("Genel benzerlik")
    return out[:8]


def format_similarity_block(
    score_percent: float,
    explanations: list[str],
    *,
    title: str = "Benzerlik",
) -> str:
    """HTML blok — inspector / kart için."""
    pct = f"%{score_percent:.0f}" if score_percent else ""
    head = f"<b>{title}:</b> {pct}" if pct else f"<b>{title}</b>"
    if not explanations:
        return head
    checks = "<br>".join(f"✓ {line}" for line in explanations)
    return f"{head}<br><br><b>Neden:</b><br>{checks}"


def format_similarity_plain(score_percent: float, explanations: list[str]) -> str:
    """Kısa tek satır — sonuç kartı."""
    if not explanations:
        return f"%{score_percent:.0f}" if score_percent else ""
    short = " · ".join(f"✓ {x}" for x in explanations[:3])
    return f"%{score_percent:.0f} — {short}"
