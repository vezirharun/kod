"""Confidence Engine — skor + güven seviyesi + insan-okunur sebepler.

Kullanım:
    from core.confidence_engine import compute_confidence
    conf = compute_confidence(result, mode="text")  # veya "image"
    # conf.level: "Çok Yüksek" / "Yüksek" / "Orta" / "Düşük"
    # conf.reasons: ["✓ Aynı Pattern DNA", "✓ Dosya adı eşleşmesi", ...]
    # conf.warnings: ["✗ Family farklı", ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ── Güven seviyeleri ─────────────────────────────────────────────────────────

LEVEL_VERY_HIGH = "Çok Yüksek"
LEVEL_HIGH      = "Yüksek"
LEVEL_MEDIUM    = "Orta"
LEVEL_LOW       = "Düşük"

MATCH_EXACT = "Kesin"
MATCH_BRAND = "Marka"
MATCH_CONCEPT = "Kavram"
MATCH_VISUAL = "Görsel"

_LEVEL_ORDER = {
    LEVEL_VERY_HIGH: 4,
    LEVEL_HIGH:      3,
    LEVEL_MEDIUM:    2,
    LEVEL_LOW:       1,
}


@dataclass
class ConfidenceResult:
    score: float             # 0-1
    score_percent: float     # 0-100
    level: str               # "Çok Yüksek" vb.
    reasons: list[str]       # ✓ satırlar
    warnings: list[str]      # ✗ satırlar
    tier_label: str = ""     # "İsim eşleşmesi", "OCR eşleşmesi" vb.
    match_type: str = ""     # Kesin / Marka / Kavram / Görsel
    confidence_score: float = 0.0   # 0-1, güven kalitesi
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Tek satır özet: 'Benzerlik: %92 · Güven: Yüksek'"""
        kind = f" · {self.match_type}" if self.match_type else ""
        return f"Benzerlik: %{self.score_percent:.0f}{kind} · Güven: {self.level}"


# ── Sinyaller tablosu ────────────────────────────────────────────────────────

_IMAGE_SIGNALS = [
    # (breakdown_key, eşik, pozitif mesaj, ağırlık)
    ("dna_score",           0.55, "Aynı Pattern DNA",        3),
    ("repeat_score",        0.50, "Aynı Repeat",             2),
    ("motif_score",         0.50, "Aynı Motif",              2),
    ("palette_similarity",  0.72, "Aynı Renk Paleti",        1),
    ("color_score",         0.55, "Benzer Renk",             1),
    ("phash",               0.88, "Hash eşleşmesi",          3),
    ("texture",             0.70, "Doku eşleşmesi",          2),
    ("faiss_score",         0.75, "AI Embedding eşleşmesi",  3),
    ("patch_sim",           0.65, "Patch benzerliği",        2),
    ("semantic_match",      0.50, "Aynı Semantic Etiket",    2),
]

_TEXT_SIGNALS = [
    # (breakdown_key, eşik, mesaj, ağırlık)
    ("clip_score",         0.30, "CLIP görsel eşleşmesi",     2),
    ("filename_score",     0.85, "Dosya adı eşleşmesi",      4),
    ("ocr_score",          0.90, "OCR metin eşleşmesi",       3),
    ("brand_alias_score",  0.85, "Marka eşleşmesi",           3),
    ("family_score",       0.75, "Aynı Desen Ailesi",         2),
    ("texture_score",      0.70, "Doku etiketi eşleşmesi",    2),
    ("nl_score",           0.65, "Doğal dil eşleşmesi",       2),
    ("auto_tag_score",     0.80, "AI etiketi eşleşmesi",      2),
    ("group_score",        0.80, "Aynı Pattern Group",        2),
    ("feedback_score",     0.80, "Kullanıcı etiket eşleşmesi",1),
    ("text_score",         0.65, "Blob metin eşleşmesi",      1),
]

_WARNING_CHECKS = [
    # (breakdown_key, max_eşik_kabul, uyarı mesajı)
    ("family_score",     0.40, "Family farklı"),
    ("filename_score",   0.30, "Dosya adında eşleşme yok"),
    ("ocr_score",        0.20, "OCR verisi yok / eşleşmiyor"),
]


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def compute_confidence(
    result: Any,
    *,
    mode: str = "image",   # "image" | "text"
) -> ConfidenceResult:
    """SearchResult nesnesinden güven sonucu hesapla."""
    score = float(getattr(result, "score", 0) or 0)
    score_pct = float(getattr(result, "score_percent", score * 100) or 0)
    breakdown: dict[str, float] = dict(getattr(result, "breakdown", {}) or {})
    debug: dict[str, Any]       = dict(getattr(result, "debug", {}) or {})
    match_exp: list[str]        = list(getattr(result, "match_explanations", []) or [])

    signals = _TEXT_SIGNALS if mode == "text" else _IMAGE_SIGNALS
    reasons: list[str] = []
    warnings: list[str] = []
    total_weight = 0
    hit_weight   = 0
    tier_label   = ""

    # Mevcut match_explanations ekle (zaten hesaplanmış)
    for exp in match_exp[:4]:
        if exp and f"✓ {exp}" not in reasons:
            reasons.append(f"✓ {exp}")

    # Sinyal tablosunu tara
    for key, thr, msg, w in signals:
        val = breakdown.get(key, debug.get(key, 0.0))
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        val = float(val or 0)
        total_weight += w
        if val >= thr:
            hit_weight += w
            label = f"+ {msg}"
            if label not in reasons:
                reasons.append(label)
            if not tier_label and key in (
                "filename_score", "ocr_score", "brand_alias_score", "dna_score", "phash"
            ):
                tier_label = msg

    # DNA detayı (görsel mod)
    if mode == "image":
        dna_match = debug.get("dna_match_fields") or {}
        if isinstance(dna_match, dict):
            for field_name, field_label in [
                ("Repeat", "Ayni Repeat"), ("Motif", "Ayni Motif"),
                ("Style", "Ayni Stil"),    ("Semantic", "Ayni Semantic"),
            ]:
                if dna_match.get(field_name) and f"+ {field_label}" not in reasons:
                    reasons.append(f"+ {field_label}")

    # Uyarılar
    for key, max_thr, msg in _WARNING_CHECKS:
        val = float(breakdown.get(key, 0) or 0)
        if val < max_thr:
            warnings.append(f"- {msg}")

    # Family uyarısı
    query_pf = debug.get("query_family_hints", {})
    if isinstance(query_pf, dict):
        query_pf = query_pf.get("pattern_family", "")
    result_pf = getattr(result, "pattern_family", "") or ""
    if query_pf and result_pf and query_pf != result_pf and result_pf != "unknown":
        warnings.append(f"- Beklenen aile: {query_pf} / Bulunan: {result_pf}")

    # Güven skoru (0-1). Score must not impersonate confidence: a lone weak
    # CLIP hit at 0.38 cannot become "high confidence" just because percent is high.
    conf_score = (hit_weight / max(total_weight, 1)) if total_weight else 0.0
    conf_score = min(1.0, conf_score * (min(score, 0.85) ** 0.5))

    brand_alias = float(breakdown.get("brand_alias_score") or debug.get("brand_alias_score") or 0.0)
    protected = bool(
        debug.get("protected_exact")
        or getattr(result, "is_self_match", False)
    )
    visual_pattern = bool(debug.get("pattern_visual_positive"))
    person_hit = bool(
        debug.get("face_gender_match")
        or debug.get("face_match")
        or float(debug.get("gender_visual_score") or 0.0) >= 0.18
        or debug.get("fusion_person_score", 0) >= 0.5
    )
    fq = debug.get("fusion_query_v2") if isinstance(debug.get("fusion_query_v2"), dict) else {}
    lane = str(fq.get("ranking_lane") or "")
    accuracy_lane = str(fq.get("accuracy_lane") or debug.get("accuracy_lane") or "")
    evidence_ok = debug.get("evidence_ok")
    kanit_yok = bool(debug.get("kanit_yok"))
    if kanit_yok:
        label = str(debug.get("kanit_yok_label") or "").strip()
        if label and f"- {label}" not in warnings:
            warnings.insert(0, f"- {label}")
        elif not any("Kanıt yok" in w for w in warnings):
            warnings.insert(0, "- Kanıt yok")
        if "- Benzerlik ≠ doğruluk" not in reasons:
            reasons.append("- Benzerlik ≠ doğruluk")
    brand_hit = bool(
        debug.get("brand_evidence")
        or debug.get("brand_match")
        or debug.get("protected_exact")
        or float(breakdown.get("brand_evidence_hit") or 0.0) >= 0.5
        or brand_alias >= 0.85
        or float(fq.get("brand") or 0.0) >= 0.5
    )

    if protected:
        match_type = MATCH_EXACT
    elif brand_hit and (score >= 0.85 or lane == "A"):
        match_type = MATCH_BRAND
    elif kanit_yok or (accuracy_lane == "object" and evidence_ok is False):
        match_type = MATCH_VISUAL
        conf_score = min(conf_score, 0.22)
    elif visual_pattern or (person_hit and evidence_ok is not False) or (
        lane == "B" and accuracy_lane != "object"
    ):
        match_type = MATCH_CONCEPT
    else:
        match_type = MATCH_VISUAL

    # Exact brand / protected matches: confidence follows evidence kind, not
    # missing filename/OCR warnings. Weak CLIP stays low.
    if protected or (match_type == MATCH_BRAND and score >= 0.90):
        conf_score = max(conf_score, 0.92)
        warnings = [
            w for w in warnings
            if "Dosya adında" not in w and "OCR verisi" not in w
        ]
        level = LEVEL_VERY_HIGH
    elif score >= 0.90 and conf_score >= 0.60:
        level = LEVEL_VERY_HIGH
    elif score >= 0.75 and conf_score >= 0.40:
        level = LEVEL_HIGH
    elif score >= 0.55 and conf_score >= 0.25:
        level = LEVEL_MEDIUM
    else:
        level = LEVEL_LOW

    # Tutarsızlık: uyarı varsa seviye bir düşür (not for protected/brand floor)
    if warnings and match_type not in {MATCH_EXACT, MATCH_BRAND} and _LEVEL_ORDER[level] > 1:
        levels = [LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH, LEVEL_VERY_HIGH]
        level = levels[_LEVEL_ORDER[level] - 2]

    return ConfidenceResult(
        score=score,
        score_percent=score_pct,
        level=level,
        reasons=reasons[:6],
        warnings=warnings[:4],
        tier_label=tier_label,
        match_type=match_type,
        confidence_score=round(conf_score, 3),
        raw={"breakdown": breakdown, "score": score, "conf_weight": conf_score, "match_type": match_type},
    )
