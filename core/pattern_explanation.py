"""Sonuç başına insan okunur AI açıklaması — şablon tabanlı."""

from __future__ import annotations

from typing import Any


def build_result_explanation(result: Any) -> dict[str, Any]:
    """Sonuç kartı ve inspector için açıklama paketi."""
    debug = dict(getattr(result, "debug", {}) or {})
    texture = dict(debug.get("texture_map") or {})
    dna = dict(texture.get("pattern_dna") or {})
    family = str(dna.get("family") or "")
    subtype = str(dna.get("subfamily") or "")
    confidence = float(dna.get("confidence") or 0)
    label = subtype or family.replace("_", " ").title() if family else "Belirsiz"
    title = f"Bu desen {label} olarak değerlendirildi."
    if confidence >= 0.7:
        title = f"Bu desen {label} olarak değerlendirildi (%{confidence * 100:.0f} güven)."

    exact = bool(debug.get("protected_exact") or debug.get("is_self_match"))
    qev = dict(debug.get("query_evidence_report") or {})
    visual_verdict = str(
        debug.get("visual_verdict") or qev.get("visual_verdict") or ""
    )
    if not visual_verdict:
        from core.query_evidence import visual_verdict_text

        visual_verdict = visual_verdict_text(qev) if qev else ""
    search_reason = (
        "Aynı dosya nedeniyle sonuç geldi"
        if exact
        else str(getattr(result, "cluster_reason", "") or "Benzerlik skoruyla sonuç geldi")
    )
    if visual_verdict:
        search_reason = f"{visual_verdict}. {search_reason}"
    if dna:
        category_reason = (
            "Kategori tahmini: Pattern DNA "
            f"({str(dna.get('source') or 'pattern_dna')}) etiketinden geldi"
        )
    else:
        category_reason = "Kategori tahmini: Pattern DNA yok — Kararsız"

    reasons: list[str] = []
    if visual_verdict:
        reasons.append(visual_verdict)
    grade = str(debug.get("visual_grade") or qev.get("visual_grade") or "")
    if grade:
        reasons.append(f"Görsel kanıt: {grade}")
    if dna.get("motif"):
        reasons.append(f"Motif: {dna['motif']}")
    if dna.get("repeat"):
        reasons.append(f"Repeat: {dna['repeat']}")
    if dna.get("texture"):
        reasons.append(f"Doku: {dna['texture']}")
    repeat_pct = float(texture.get("repeat_density", 0) or 0)
    if repeat_pct > 0:
        reasons.append(f"Repeat oranı %{repeat_pct * 100:.0f}")
    tex_sim = float(debug.get("texture_score", 0) or 0)
    if tex_sim > 0:
        reasons.append(f"Doku benzerliği %{tex_sim * 100:.0f}")
    if texture.get("classification_source") == "gold_dataset":
        reasons.append("Gold Dataset eşleşmesi")
    elif texture.get("user_labeled"):
        reasons.append("Kullanıcı düzeltmesi")
    semantic = texture.get("semantic_tags") or {}
    if isinstance(semantic, dict) and semantic.get("motifs"):
        motifs = ", ".join(str(m) for m in semantic["motifs"][:3])
        reasons.append(f"Semantik etiketler: {motifs}")
    dino = float(debug.get("dino_score", 0) or 0)
    clip = float(debug.get("clip_score", 0) or 0)
    if dino > 0:
        reasons.append(f"DINO skoru %{dino * 100:.0f}")
    if clip > 0:
        reasons.append(f"CLIP skoru %{clip * 100:.0f}")
    cluster = getattr(result, "cluster_reason", "") or debug.get("cluster_reason", "")
    if cluster:
        reasons.append(str(cluster))
    text_reason = getattr(result, "text_match_reason", "") or ""
    if text_reason:
        reasons.append(str(text_reason))

    tier = str(texture.get("user_similarity_tier") or debug.get("similarity_tier") or "")
    tier_labels = {
        "exact": "Aynı desen",
        "near_variant": "Yakın varyant",
        "same_family": "Aynı aile",
        "similar_motif": "Benzer motif",
        "similar_texture": "Benzer doku",
        "similar_color": "Benzer renk",
    }
    group_label = tier_labels.get(tier, "")

    return {
        "title": title,
        "reasons": reasons[:8],
        "group_label": group_label,
        "confidence": confidence,
        "dna": dna,
        "search_reason": search_reason,
        "category_reason": category_reason,
    }


def format_explanation_text(explanation: dict[str, Any]) -> str:
    lines = [
        explanation.get("search_reason", ""),
        explanation.get("category_reason", ""),
    ]
    for reason in explanation.get("reasons") or []:
        lines.append(f"✓ {reason}")
    return "\n".join(lines)
