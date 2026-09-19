"""Textile Re-ranking Engine v2

İlk aday listesini (FAISS/hash sonrası) tekstil-odaklı bileşenlerle yeniden puanlar.

Final ≈
  0.30 Embedding
  0.20 Pattern DNA
  0.15 Texture Descriptor
  0.10 Repeat Similarity
  0.10 Spot Geometry
  0.05 Color Palette
  0.05 Knowledge Graph
  0.05 Semantic

Aile çatışması (ör. leopard sorgusunda floral) agresif cezalandırılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# v2 default weights (sum = 1.0)
V2_WEIGHTS: dict[str, float] = {
    "embedding": 0.30,
    "dna": 0.20,
    "texture": 0.15,
    "repeat": 0.10,
    "spot_geometry": 0.10,
    "color": 0.05,
    "knowledge": 0.05,
    "semantic": 0.05,
}

# Query family → conflicting families (hard demote)
_FAMILY_CONFLICTS: dict[str, frozenset[str]] = {
    "animal_print": frozenset(
        {
            "floral",
            "marble_abstract",
            "plaid_check",
            "geometric",
            "monogram_logo",
            "typography_text",
            "baroque",
            "lace",
            "plain",
        }
    ),
    "floral": frozenset({"animal_print", "plaid_check", "camouflage", "monogram_logo"}),
    "plaid_check": frozenset({"floral", "animal_print", "marble_abstract"}),
    "marble_abstract": frozenset({"animal_print", "plaid_check"}),
}


@dataclass
class SpotGeometry:
    """Leopard / organic spot karakteri — mevcut texture_map'ten türetilir."""

    spot_density: float = 0.0          # benek yoğunluğu
    spot_size: float = 0.0             # baskın benek ölçeği
    size_variance: float = 0.0         # ölçek varyansı proxy
    ring_thickness: float = 0.0        # kenar/kontrast → halka kalınlığı proxy
    elongation: float = 0.0            # stripe vs blob dengesi
    background_ratio: float = 0.0      # negatif alan
    edge_irregularity: float = 0.0
    contrast: float = 0.0
    flow_orientation: float = 0.0      # stripe yönlülük
    organic_score: float = 0.0


@dataclass
class RerankBreakdown:
    embedding: float = 0.0
    dna: float = 0.0
    texture: float = 0.0
    repeat: float = 0.0
    spot_geometry: float = 0.0
    color: float = 0.0
    knowledge: float = 0.0
    semantic: float = 0.0
    family_penalty: float = 0.0
    final: float = 0.0
    matched: list[str] = field(default_factory=list)


def _tm(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        tm = obj.get("texture_map") if "texture_map" in obj else obj
        if isinstance(tm, dict):
            return tm
    return {}


def _f(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return default


def extract_spot_geometry(rec: dict[str, Any] | Any) -> SpotGeometry:
    """texture_map alanlarından spot geometrisi."""
    tm: dict[str, Any] = {}
    if hasattr(rec, "debug"):
        debug = getattr(rec, "debug", None) or {}
        raw = debug.get("texture_map") or {}
        if isinstance(raw, dict):
            tm = dict(raw)
    elif isinstance(rec, dict):
        raw = rec.get("texture_map") if isinstance(rec.get("texture_map"), dict) else {}
        tm = dict(raw)
        # düz alanlar (query_rec)
        for k in (
            "organic_blob_score",
            "stripe_score",
            "scale_pattern_score",
            "edge_density",
            "contrast_score",
            "repeat_density",
            "animal_score",
            "floral_score",
        ):
            if k in rec and k not in tm:
                tm[k] = rec[k]

    blob = _f(tm, "organic_blob_score")
    stripe = _f(tm, "stripe_score")
    scale = _f(tm, "scale_pattern_score")
    edge = _f(tm, "edge_density")
    contrast = _f(tm, "contrast_score")
    repeat = _f(tm, "repeat_density")
    animal = _f(tm, "animal_score")
    floral = _f(tm, "floral_score")

    organic = min(
        1.0,
        blob * 0.55
        + animal * 0.30
        + scale * 0.15
        - floral * 0.35
        - max(0.0, stripe - 0.45) * 0.25,
    )
    organic = max(0.0, organic)

    bg = max(0.0, 1.0 - blob * 0.7 - repeat * 0.3)
    elong = min(1.0, stripe * 0.85 + (1.0 - blob) * 0.15)
    var = min(1.0, abs(scale - blob) * 1.4 + edge * 0.2)
    ring = min(1.0, edge * 0.55 + contrast * 0.45)
    flow = min(1.0, stripe)

    return SpotGeometry(
        spot_density=min(1.0, blob * 0.7 + repeat * 0.3),
        spot_size=scale,
        size_variance=var,
        ring_thickness=ring,
        elongation=elong,
        background_ratio=bg,
        edge_irregularity=min(1.0, edge * 0.8 + (1.0 - contrast) * 0.2),
        contrast=contrast,
        flow_orientation=flow,
        organic_score=organic,
    )


def spot_geometry_similarity(a: SpotGeometry, b: SpotGeometry) -> float:
    """Bileşen bazlı spot benzerliği (L1 soft)."""
    pairs = (
        (a.spot_density, b.spot_density, 0.18),
        (a.spot_size, b.spot_size, 0.14),
        (a.size_variance, b.size_variance, 0.08),
        (a.ring_thickness, b.ring_thickness, 0.12),
        (a.elongation, b.elongation, 0.14),
        (a.background_ratio, b.background_ratio, 0.10),
        (a.edge_irregularity, b.edge_irregularity, 0.08),
        (a.contrast, b.contrast, 0.06),
        (a.flow_orientation, b.flow_orientation, 0.05),
        (a.organic_score, b.organic_score, 0.05),
    )
    score = 0.0
    for x, y, w in pairs:
        score += w * (1.0 - min(1.0, abs(x - y)))
    return max(0.0, min(1.0, score))


def _family_of(result: Any, query_family: str = "") -> str:
    debug = getattr(result, "debug", None) or {}
    fam = (
        getattr(result, "pattern_family", None)
        or debug.get("pattern_family")
        or debug.get("result_family")
        or ""
    )
    return str(fam or "").strip().lower()


def _component_from_debug(result: Any) -> dict[str, float]:
    debug = getattr(result, "debug", None) or {}
    b = getattr(result, "breakdown", None) or {}
    fb = debug.get("family_breakdown") or {}
    return {
        "embedding": float(
            debug.get("ai_score")
            or max(float(b.get("dino", 0) or 0), float(b.get("clip", 0) or 0))
            or 0
        ),
        "dna": float(debug.get("dna_score") or fb.get("dna") or b.get("dna") or 0),
        "texture": float(
            debug.get("texture_score") or fb.get("texture") or b.get("texture") or 0
        ),
        "repeat": float(fb.get("repeat") or debug.get("repeat_score") or 0),
        "color": float(
            debug.get("color_score")
            or getattr(result, "palette_similarity", 0)
            or fb.get("color")
            or 0
        ),
        "knowledge": float(
            debug.get("knowledge_score")
            or fb.get("kb_combined")
            or 0
        ),
        "semantic": float(
            debug.get("semantic_score") or fb.get("semantic") or b.get("semantic") or 0
        ),
        "patch": float(debug.get("patch_score") or fb.get("patch") or 0),
        "phash": float(debug.get("phash_score") or b.get("phash") or 0),
    }


def compute_rerank_score(
    query_rec: dict[str, Any],
    result: Any,
    *,
    query_spot: SpotGeometry | None = None,
    weights: dict[str, float] | None = None,
) -> RerankBreakdown:
    w = dict(weights or V2_WEIGHTS)
    q_tm = _tm(query_rec)
    q_family = str(
        query_rec.get("pattern_family")
        or q_tm.get("pattern_family")
        or ""
    ).strip().lower()
    q_spot = query_spot or extract_spot_geometry(query_rec)
    c_spot = extract_spot_geometry(result)
    spot_sim = spot_geometry_similarity(q_spot, c_spot)

    comps = _component_from_debug(result)
    # Embedding yoksa phash+patch proxy
    emb = comps["embedding"]
    if emb < 0.05:
        emb = 0.55 * comps["phash"] + 0.45 * comps["patch"]

    dna = comps["dna"]
    texture = comps["texture"]
    repeat = comps["repeat"]
    if repeat < 0.05:
        # repeat proxy from scale similarity via spot size
        repeat = 1.0 - min(1.0, abs(q_spot.spot_size - c_spot.spot_size))
    color = comps["color"]
    knowledge = comps["knowledge"]
    semantic = comps["semantic"]

    final = (
        w.get("embedding", 0.30) * emb
        + w.get("dna", 0.20) * dna
        + w.get("texture", 0.15) * texture
        + w.get("repeat", 0.10) * repeat
        + w.get("spot_geometry", 0.10) * spot_sim
        + w.get("color", 0.05) * color
        + w.get("knowledge", 0.05) * knowledge
        + w.get("semantic", 0.05) * semantic
    )

    # Korunan taban: orijinal skorun yumuşak karışımı (sıfırlama yok)
    base = float(getattr(result, "score", 0.0) or 0.0)
    blended = 0.55 * final + 0.45 * base

    c_family = _family_of(result)
    penalty = 0.0
    matched: list[str] = []

    conflicts = _FAMILY_CONFLICTS.get(q_family, frozenset())
    if q_family and c_family and c_family in conflicts:
        # Sert ceza: floral vb. leopard listesinde üste çıkmasın
        penalty = 0.28
        if spot_sim < 0.55:
            penalty += 0.10
        blended = min(blended, 0.72) - penalty * 0.5
        blended = max(0.0, blended)
    elif q_family and c_family and q_family == c_family:
        matched.append(f"family:{q_family}")
        # Aynı aile + zayıf spot → hafif düşür (yanlış aile içi sıralama)
        if q_family == "animal_print" and spot_sim < 0.45:
            blended *= 0.92
        elif q_family == "animal_print" and spot_sim >= 0.70:
            blended = min(1.0, blended + 0.04)
            matched.append("spot_geometry")

    # %75–85 bandında aile/spot uyumsuzsa agresif filtre sinyali
    if 0.74 <= blended <= 0.86:
        if penalty > 0 or (q_family == "animal_print" and spot_sim < 0.50):
            blended = min(blended, 0.73)

    if spot_sim >= 0.65:
        matched.append("spot_ok")
    if dna >= 0.55:
        matched.append("dna")
    if knowledge >= 0.40:
        matched.append("knowledge")

    # Aşama 2A: Object ≠ Pattern (Pattern DNA güçlüyse ceza yok).
    try:
        from core.object_pattern_gate import adjust_score_for_object_vs_pattern

        q_text = str(
            query_rec.get("text")
            or query_rec.get("query_text")
            or q_tm.get("query_text")
            or ""
        )
        blended, op_meta = adjust_score_for_object_vs_pattern(
            blended,
            result,
            query_text=q_text,
            query_family=q_family,
        )
        if op_meta.get("penalty"):
            penalty = max(penalty, float(op_meta.get("penalty") or 0))
            matched.append("object_pattern_gate")
        elif op_meta.get("reason") == "strong_textile":
            matched.append("textile_protected")
    except Exception:
        pass

    return RerankBreakdown(
        embedding=round(emb, 4),
        dna=round(dna, 4),
        texture=round(texture, 4),
        repeat=round(repeat, 4),
        spot_geometry=round(spot_sim, 4),
        color=round(color, 4),
        knowledge=round(knowledge, 4),
        semantic=round(semantic, 4),
        family_penalty=round(penalty, 4),
        final=round(max(0.0, min(1.0, blended)), 4),
        matched=matched,
    )


def textile_rerank_v2(
    query_rec: dict[str, Any],
    results: list[Any],
    *,
    pool_k: int = 300,
    show_k: int = 0,
    weights: dict[str, float] | None = None,
) -> list[Any]:
    """İlk pool_k adayı yeniden puanla; isteğe bağlı show_k ile kes.

    Exact / self sırasını korur.
    """
    if not results:
        return results

    head = list(results[: max(1, int(pool_k))])
    tail = list(results[int(pool_k) :])
    q_spot = extract_spot_geometry(query_rec)

    protected: list[Any] = []
    rerankable: list[tuple[float, Any]] = []

    for rec in head:
        debug = getattr(rec, "debug", None) or {}
        is_self = bool(getattr(rec, "is_self_match", False))
        if is_self or debug.get("protected_exact"):
            protected.append(rec)
            continue
        br = compute_rerank_score(
            query_rec, rec, query_spot=q_spot, weights=weights
        )
        rec.score = br.final
        if hasattr(rec, "score_percent"):
            rec.score_percent = round(br.final * 100, 1)
        debug = dict(debug)
        debug["textile_rerank_v2"] = {
            "embedding": br.embedding,
            "dna": br.dna,
            "texture": br.texture,
            "repeat": br.repeat,
            "spot_geometry": br.spot_geometry,
            "color": br.color,
            "knowledge": br.knowledge,
            "semantic": br.semantic,
            "family_penalty": br.family_penalty,
            "final": br.final,
            "matched": br.matched,
        }
        debug["spot_geometry_score"] = br.spot_geometry
        rec.debug = debug
        rerankable.append((br.final, rec))

    rerankable.sort(
        key=lambda x: (
            -(x[0]),
            -float(getattr(x[1], "hierarchy_score", 0) or 0),
        )
    )
    ordered = protected + [r for _, r in rerankable]
    if show_k and show_k > 0:
        # show_k sadece sıralama ipucu; listeyi kesme — UI zaten sayfalar
        pass
    return ordered + tail
