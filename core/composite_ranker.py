"""Faz 3 — birleşik sorgu sıralayıcı.

Bu katman yalnızca arama sonuçlarını yeniden sıralar. Index, SQLite, FAISS,
thumbnail, preview veya kuyruklara dokunmaz.

Amaç:
    "leopard gül" gibi sorgularda aynı kayıtta iki kavramın birlikte bulunmasını,
    tek kavramı çok yüksek skorla taşıyan kayıttan daha yukarı almaktır.
"""
from __future__ import annotations

from typing import Any

from core.multi_concept_composition import MultiConceptCompositionEngine


_CONCEPT_MIN = 0.22
_ANIMAL_MIN = 0.26


def _debug(row: Any) -> dict[str, Any]:
    return getattr(row, "debug", {}) or {}


def _qev(row: Any) -> dict[str, Any]:
    return _debug(row).get("query_evidence_report") or {}


def _clips(row: Any) -> dict[str, float]:
    raw = _debug(row).get("v2_clip") or {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = max(0.0, min(1.0, float(v)))
        except Exception:
            continue
    return out


def _concept_aliases(concept: str) -> tuple[str, ...]:
    c = str(concept or "").strip().lower()
    aliases = {
        "rose": ("rose", "gül", "gul"),
        "leopard": ("leopard", "leopar", "leo"),
        "floral": ("floral", "flower", "çiçek", "cicek"),
        "leaf": ("leaf", "yaprak"),
        "geometric": ("geometric", "geometrik"),
        "daisy": ("daisy", "papatya"),
        "tulip": ("tulip", "lale"),
    }
    return aliases.get(c, (c,))


def _clip_for_concept(clips: dict[str, float], concept: str) -> float:
    return max((float(clips.get(k) or 0.0) for k in _concept_aliases(concept)), default=0.0)


def _soft_family_signal(clips: dict[str, float], concept: str) -> float:
    # Gül/papatya/lale gibi çiçek alt türlerinde genel floral CLIP kanıtı
    # yardımcı kanıttır; tek başına "tam eşleşme" sayılmaz. Bu, özellikle
    # gerçek görselde çiçeğin görünür olduğu halde spesifik rose promptunun
    # düşük kaldığı durumlarda composite aramayı kör bırakmaz.
    if str(concept) in {"rose", "daisy", "tulip", "orchid", "peony"}:
        return max(float(clips.get("floral") or 0.0), float(clips.get("flower") or 0.0))
    return 0.0


def _concept_signals(row: Any, required: list[str]) -> tuple[list[float], list[bool], list[float]]:
    qev = _qev(row)
    concepts = qev.get("concepts") or {}
    clips = _clips(row)
    signals: list[float] = []
    supported: list[bool] = []
    coverage_weights: list[float] = []
    for cid in required:
        ev = concepts.get(str(cid)) or {}
        state = str(ev.get("state") or "")
        ev_score = max(0.0, min(1.0, float(ev.get("score") or 0.0)))
        clip = _clip_for_concept(clips, cid)
        minimum = _ANIMAL_MIN if str(cid) in {"mouse", "bird", "butterfly", "cat", "dog", "horse", "fish", "leopard"} else _CONCEPT_MIN
        clip_support = clip >= minimum
        is_supported = state == "SUPPORTED" or clip_support

        # Query-evidence katmanı artık metadata/semantic/DNA içindeki aile
        # kanıtını da "family_bridge" olarak taşıyor. Böylece örneğin
        # DOLCE ÇİÇEKLİ LEOPAR kaydı, dosya adında "rose" yazmıyor diye
        # gül+leopard composite aramasında leopard-only kayda yenilmiyor.
        channels = ev.get("channels") or {}
        family_bridge = max(
            float(channels.get("family_bridge") or 0.0),
            float(channels.get("soft_family") or 0.0),
        )
        soft_family = max(_soft_family_signal(clips, cid), family_bridge)
        # Soft family kanıtı tam SUPPORTED değildir; yalnızca ikinci kavramın
        # aile düzeyinde aynı görselde bulunma ihtimalini sıralamaya taşır.
        weight = 0.72 if (not is_supported and soft_family >= 0.30) else (1.0 if is_supported else 0.0)
        signal = max(ev_score, clip, soft_family * 0.90)
        signals.append(signal)
        supported.append(is_supported)
        coverage_weights.append(weight)
    return signals, supported, coverage_weights


def composite_rank_score(
    row: Any,
    *,
    query_is_composite: bool,
    required: list[str] | tuple[str, ...] | None = None,
    required_count: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """PRO multi-concept ranking.

    Weight contract for composite queries:
        30% concept coverage
        25% same-composition relationship
        15% regional/patch evidence
        10% semantic evidence
        10% visual embedding evidence
         5% Pattern DNA
         5% texture

    The weights are applied only in this re-ranking layer. Indexing, FAISS,
    embeddings, Preview/Thumbnail and the existing single-concept path are
    untouched.

    A complete multi-concept match still requires every requested concept to
    be supported. Same-composition evidence acts as a gate: concepts found in
    unrelated/independent content cannot masquerade as a full match.
    """
    base = max(0.0, min(1.0, float(getattr(row, "score", 0.0) or 0.0)))
    dbg = _debug(row)
    qev = _qev(row)
    clips = _clips(row)

    if not query_is_composite:
        return base, {
            "formula_version": "multi_pro_v3.0",
            "query_is_composite": False,
            "base": round(base, 4),
        }

    req = []
    for x in (required or ()):
        cid = str(x).strip()
        if cid and cid not in req:
            req.append(cid)
    if required_count is not None and not req:
        req = [f"__required_{i}" for i in range(int(required_count))]
    if not req:
        return base, {
            "formula_version": "multi_pro_v3.0",
            "query_is_composite": True,
            "base": round(base, 4),
        }

    signals, supported, coverage_weights = _concept_signals(row, req)
    n = len(req)
    # Existing Query Evidence / composite evidence is the relationship signal.
    q_evidence = max(0.0, min(1.0, float(qev.get("query_evidence") or 0.0)))
    comp_evidence = max(0.0, min(1.0, float(qev.get("composite_evidence") or 0.0)))
    visual_composite = max(0.0, min(1.0, float(qev.get("visual_composite") or 0.0)))
    joint = max(
        comp_evidence,
        visual_composite,
        float(clips.get("_composite") or 0.0),
    )

    # Existing per-result feature scores. No new index fields are required.
    contrib = dbg.get("contribution_scores") or {}
    patch = max(
        float(dbg.get("multi_scale_patch_score") or 0.0),
        float(dbg.get("patch_score") or 0.0),
        float(contrib.get("patch") or 0.0),
    )
    semantic = max(
        float(dbg.get("semantic_score") or 0.0),
        float(contrib.get("semantic") or 0.0),
    )
    dna = max(
        float(dbg.get("dna_score") or 0.0),
        float(contrib.get("dna") or 0.0),
    )
    texture = max(
        float(dbg.get("texture_score") or 0.0),
        float(dbg.get("texture_family_score") or 0.0),
        float(contrib.get("texture") or 0.0),
    )
    # "Görsel embedding" is represented by the existing AI/embedding score
    # (DINO/CLIP/FAISS) already persisted by the search engine.
    visual_embedding = max(
        float(dbg.get("ai_score") or 0.0),
        float(dbg.get("visual_score") or 0.0),
        float(dbg.get("global_visual_score") or 0.0),
        float(contrib.get("exact") or 0.0),
    )

    def clamp(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    patch = clamp(patch)
    semantic = clamp(semantic)
    dna = clamp(dna)
    texture = clamp(texture)
    visual_embedding = clamp(visual_embedding)
    contradiction = clamp(float(qev.get("contradiction_penalty") or 0.0))
    composition = MultiConceptCompositionEngine().evaluate(
        required=req,
        signals=signals,
        supported=supported,
        soft_weights=coverage_weights,
        relationship_seed=joint,
        q_evidence=q_evidence,
        patch=patch,
        semantic=semantic,
        visual_embedding=visual_embedding,
        pattern_dna=dna,
        texture=texture,
        contradiction=contradiction,
    )
    exact_count = int(composition["supported_count"])
    concept_coverage = float(composition["concept_coverage"])
    same_composition = float(composition["same_composition"])
    contribution = [
        float((composition["relative_contribution"] or {}).get(req[i], 0.0))
        for i in range(n)
    ]
    score = float(composition["final_multi_concept_score"])

    features = {
        "formula_version": "multi_pro_v3.1",
        "query_is_composite": True,
        "weights": composition["weights"],
        "base": round(base, 4),
        "required": req,
        "concept_count": n,
        "supported_count": exact_count,
        "supported": [req[i] for i, ok in enumerate(supported) if ok],
        "soft_supported": composition["soft_supported"],
        "missing": composition["missing_concepts"],
        "missing_concepts": composition["missing_concepts"],
        "concept_coverage": round(concept_coverage, 4),
        "coverage": round(concept_coverage, 4),
        "concept_evidence": composition["concept_evidence"],
        "relative_contribution": composition["relative_contribution"],
        "concept_signals": {req[i]: round(signals[i], 4) for i in range(n)},
        "same_composition": round(same_composition, 4),
        "same_composition_score": composition["same_composition_score"],
        "pairwise_relationships": composition["pairwise_relationships"],
        "group_relationship": composition["group_relationship"],
        "group_coherence": composition["group_coherence"],
        "composition_confidence": composition["composition_confidence"],
        "query_evidence": round(q_evidence, 4),
        "composite_evidence": round(comp_evidence, 4),
        "visual_composite": round(visual_composite, 4),
        "joint_relationship": round(joint, 4),
        "patch": round(patch, 4),
        "semantic": round(semantic, 4),
        "visual_embedding": round(visual_embedding, 4),
        "pattern_dna": round(dna, 4),
        "texture": round(texture, 4),
        "contradiction": round(contradiction, 4),
        "weighted_components": composition["weighted_components"],
        "final_multi_concept_score": composition["final_multi_concept_score"],
    }
    return score, features


def apply_composite_ranking(
    rows: list[Any],
    *,
    query_is_composite: bool,
    required: list[str] | tuple[str, ...] | None = None,
    required_count: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    scored: list[tuple[float, int, Any]] = []
    for pos, row in enumerate(rows):
        score, features = composite_rank_score(
            row,
            query_is_composite=query_is_composite,
            required=required,
            required_count=required_count,
        )
        row.score = score
        row.score_percent = round(score * 100, 1)
        row.debug = {**_debug(row), "composite_ranker": features}
        scored.append((score, pos, row))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [row for _, _, row in scored], {
        "enabled": True,
        "version": "phase3.2",
        "query_is_composite": bool(query_is_composite),
        "required": list(required or ()),
        "required_count": required_count,
        "count": len(scored),
    }
