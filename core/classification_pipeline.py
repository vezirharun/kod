"""Sınıflandırma öncelik zinciri — unknown en son."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_UNKNOWN = "unknown"


@dataclass
class PatternIdentity:
    family: str = _UNKNOWN
    subtype: str = ""
    confidence: float = 0.0
    source: str = "unknown"
    label: str = ""


def _subtype_from_path(category_path: str) -> str:
    parts = [p.strip() for p in (category_path or "").split("/") if p.strip()]
    return parts[-1].lower().replace(" ", "_") if len(parts) > 1 else ""


def _family_from_path(category_path: str) -> str:
    from core.category_tree import pattern_fields_for_path

    fields = pattern_fields_for_path(category_path or "")
    return str(fields.get("pattern_family") or fields.get("family") or "")


def resolve_pattern_identity(
    *,
    texture_map: dict[str, Any] | None,
    filename: str = "",
    category_path: str = "",
    manual_category_path: str = "",
    feedback_labels: list[str] | None = None,
    gold_category_path: str = "",
) -> PatternIdentity:
    """
    Öncelik:
    1. Pattern DNA (kullanıcı düzeltmeli / gold)
    2. Semantic tags
    3. Gold dataset kategori
    4. AI tahminleri
    5. Kullanıcı geri bildirim etiketleri
    6. Heuristik texture_map
    7. unknown (son çare)
    """
    tm = dict(texture_map or {})
    feedback_labels = [str(x).strip().lower() for x in (feedback_labels or []) if x]

    # 1 — Pattern DNA (user/gold kaynaklı)
    dna = tm.get("pattern_dna") or {}
    if isinstance(dna, dict):
        dna_source = str(dna.get("source") or "")
        dna_family = str(dna.get("family") or "")
        if dna_family and dna_family != _UNKNOWN and dna_source in (
            "user_feedback",
            "gold_dataset",
            "manual",
        ):
            return PatternIdentity(
                family=dna_family,
                subtype=str(dna.get("subfamily") or "").lower().replace(" ", "_"),
                confidence=max(float(dna.get("confidence") or 0), 0.95),
                source=dna_source,
                label=_display_label(dna_family, dna.get("subfamily", "")),
            )

    # Manuel etiket — texture_map
    if tm.get("user_labeled"):
        fam = str(tm.get("pattern_family") or "")
        if fam and fam != _UNKNOWN:
            sub = str(tm.get("animal_print_type") or tm.get("pattern_subtype") or "")
            return PatternIdentity(
                family=fam,
                subtype=sub,
                confidence=max(float(tm.get("classification_confidence") or 0), 0.98),
                source="user_feedback",
                label=_display_label(fam, sub),
            )

    # 2 — Semantic tags
    semantic = tm.get("semantic_tags") or {}
    if isinstance(semantic, dict):
        sem_family = str(semantic.get("family") or "")
        sem_conf = float(semantic.get("confidence") or 0)
        if sem_family and sem_family != _UNKNOWN and sem_conf >= 0.55:
            sub = str(semantic.get("subtype") or "")
            return PatternIdentity(
                family=sem_family,
                subtype=sub,
                confidence=sem_conf,
                source="semantic_tags",
                label=_display_label(sem_family, sub),
            )

    # 3 — Gold dataset / manuel kategori
    gold_path = gold_category_path or manual_category_path or category_path
    if gold_path:
        fam = _family_from_path(gold_path)
        if fam and fam != _UNKNOWN:
            sub = _subtype_from_path(gold_path)
            return PatternIdentity(
                family=fam,
                subtype=sub,
                confidence=0.92,
                source="gold_dataset",
                label=gold_path.replace("/", " / "),
            )

    # 4 — AI tahminleri
    # Legacy ai_category_predictions is intentionally ignored. Older rows may
    # contain labels derived only from texture/family scores, not Pattern DNA.

    # 5 — Kullanıcı geri bildirim etiketleri
    for label in feedback_labels:
        if not label or label in ("wrong", "not_family"):
            continue
        from core.textile_terms import query_family_hints

        hints = query_family_hints(label, include_semantic=True)
        if hints:
            fam = hints[0]
            if fam and fam != _UNKNOWN:
                return PatternIdentity(
                    family=fam,
                    subtype=label.replace(" ", "_"),
                    confidence=0.88,
                    source="user_feedback",
                    label=label.title(),
                )

    # 6 — Heuristik texture_map
    fam = str(tm.get("pattern_family") or "")
    conf = float(tm.get("classification_confidence") or 0)
    sub = str(tm.get("animal_print_type") or tm.get("pattern_subtype") or "")
    if fam and fam != _UNKNOWN and conf >= 0.45:
        return PatternIdentity(
            family=fam,
            subtype=sub,
            confidence=conf,
            source="heuristic",
            label=_display_label(fam, sub),
        )

    # Dosya adı ipucu (düşük güven)
    # Filename hints remain searchable but never create category identity.

    # 7 — unknown (son)
    return PatternIdentity(
        family=_UNKNOWN,
        subtype="",
        confidence=0.0,
        source="unknown",
        label="Belirsiz",
    )


def apply_identity_to_texture_map(
    texture_map: dict[str, Any],
    identity: PatternIdentity,
) -> dict[str, Any]:
    """Kimliği texture_map'e yazar; kullanıcı etiketli kayıtları ezmez."""
    if texture_map.get("user_labeled"):
        return texture_map
    merged = dict(texture_map)
    if identity.family and identity.family != _UNKNOWN:
        merged["pattern_family"] = identity.family
        merged["classification_confidence"] = identity.confidence
        merged["classification_source"] = identity.source
        if identity.family == "animal_print" and identity.subtype:
            merged["animal_print_type"] = identity.subtype
        elif identity.family == "animal_print":
            merged.pop("animal_print_type", None)
            if merged.get("pattern_subtype") == "leopard":
                merged["pattern_subtype"] = ""
        elif identity.subtype:
            merged["pattern_subtype"] = identity.subtype
    return merged


def _display_label(family: str, subtype: str | Any = "") -> str:
    from core.group_gates import candidate_family_display_label

    return candidate_family_display_label(
        type(
            "M",
            (),
            {
                "pattern_family": family,
                "pattern_subtype": str(subtype or ""),
                "animal_print_type": str(subtype or "") if family == "animal_print" else "",
                "classification_confidence": 0.8,
            },
        )()
    )
