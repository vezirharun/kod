"""Fusion Query Intelligence v2 — marka + renk + nesne + motif + görsel.

Yalnızca arama sonuçlarını yeniden sıralar. Index/FAISS yazmaz, model yüklemez.
Kanalsız sonuç üretmez; eksik kanal diğerlerini kapatmaz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.entity_intelligence import annotate_search_results, parse_entity_query
from core.visual_concept import category_for_lemma, compile_visual_query, english_lemma
from core.visual_pattern_query import (
    apply_visual_pattern_priority,
    is_visual_pattern_query,
    lookup_pattern_token,
    parse_visual_pattern_concepts,
    visual_pattern_signals,
)

_PERSON_LEMMAS = frozenset(
    {
        "person", "woman", "man", "female", "male", "kadin", "kadın",
        "erkek", "bayan", "kisi", "kişi",
    }
)
_GENDER_ONLY_QUERIES = frozenset(
    {
        "kadin", "kadın", "erkek", "woman", "man", "female", "male",
        "bayan", "women", "men",
    }
)
_OBJECT_LEMMAS = frozenset(
    {
        "handbag", "bag", "canta", "çanta", "purse", "backpack",
        "shoe", "dress", "shirt", "coat", "jacket", "hat",
        "necklace", "eye", "face", "car", "chair",
        "bicycle", "bike", "rabbit", "bird", "tiger", "dog", "cat",
        "child", "kaplan", "tavsan", "tavşan", "kus", "kuş", "bisiklet",
        "araba", "ayakkabi", "ayakkabı",
        "cherry", "kiraz", "apple", "elma", "strawberry", "cilek", "çilek",
    }
)
_GENERIC_BRAND_QUERY = frozenset({"marka", "brand", "brands", "logo"})
_OBJECT_ALIASES = {
    "kaplan": "tiger", "tiger": "tiger",
    "tavsan": "rabbit", "tavşan": "rabbit", "rabbit": "rabbit",
    "kus": "bird", "kuş": "bird", "bird": "bird",
    "bisiklet": "bicycle", "bicycle": "bicycle", "bike": "bicycle",
    "araba": "car", "otomobil": "car", "car": "car",
    "kopek": "dog", "köpek": "dog", "dog": "dog",
    "kedi": "cat", "cat": "cat",
    "cocuk": "child", "çocuk": "child", "child": "child",
    "kiraz": "cherry", "cherry": "cherry",
    "cilek": "strawberry", "çilek": "strawberry", "strawberry": "strawberry",
    "elma": "apple", "apple": "apple",
}
_MOTIF_CATEGORIES = frozenset({"plant"})
_MOTIF_LEMMAS = frozenset(
    {
        "flower", "rose", "tulip", "daisy", "sunflower", "leaf", "floral",
        "paisley",
    }
)
_COLOR_GROUPS = {
    "red": frozenset({"red", "kirmizi", "kırmızı", "crimson", "bordo", "scarlet"}),
    "blue": frozenset({"blue", "mavi", "navy", "lacivert"}),
    "green": frozenset({"green", "yesil", "yeşil"}),
    "yellow": frozenset({"yellow", "sari", "sarı", "gold", "altin", "altın"}),
    "black": frozenset({"black", "siyah"}),
    "white": frozenset({"white", "beyaz"}),
    "pink": frozenset({"pink", "pembe"}),
    "purple": frozenset({"purple", "mor"}),
    "orange": frozenset({"orange", "turuncu"}),
    "brown": frozenset({"brown", "kahverengi", "beige", "bej"}),
    "gray": frozenset({"gray", "grey", "gri"}),
}


@dataclass
class FusionQuery:
    raw: str
    brand: str = ""
    colors: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    motifs: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    token_types: list[str] = field(default_factory=list)

    def channels(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if self.brand:
            out["brand"] = [self.brand]
        if self.colors:
            out["color"] = list(self.colors)
        if self.persons:
            out["person"] = list(self.persons)
        if self.entities:
            out["entity"] = list(self.entities)
        if self.motifs:
            out["motif"] = list(self.motifs)
        if self.patterns:
            out["pattern"] = list(self.patterns)
        return out

    @property
    def multi_channel(self) -> bool:
        bits = (
            int(bool(self.brand))
            + int(bool(self.colors))
            + int(bool(self.entities))
            + int(bool(self.motifs))
            + int(bool(self.patterns))
            + int(bool(self.persons))
        )
        return bits >= 2

    @property
    def ranking_lane(self) -> str:
        """A = exact name/brand. B = conceptual. hybrid = both present."""
        conceptual = bool(self.patterns or self.motifs or self.persons)
        if self.brand and not conceptual and not self.entities:
            return "A"
        if self.brand and (conceptual or self.entities or self.colors):
            return "hybrid"
        if conceptual or self.entities or self.colors:
            return "B"
        return "A" if self.brand else "B"

    def to_debug(self) -> dict[str, Any]:
        return {
            "version": "fusion_query_v2",
            "brand": self.brand,
            "colors": list(self.colors),
            "entities": list(self.entities),
            "motifs": list(self.motifs),
            "patterns": list(self.patterns),
            "persons": list(self.persons),
            "token_types": list(self.token_types),
            "channels": list(self.channels().keys()),
            "multi_channel": self.multi_channel,
            "ranking_lane": self.ranking_lane,
            "visual_pattern_query": bool(self.patterns or is_visual_pattern_query(self.raw)),
            "accuracy_lane": _accuracy_lane_for(self.raw),
        }


def _accuracy_lane_for(raw: str) -> str:
    try:
        from core.search_evidence_gate import classify_accuracy_lane

        return classify_accuracy_lane(raw)
    except Exception:
        return ""


_LAST_GATE_META: dict[str, Any] = {}


def last_evidence_gate_meta() -> dict[str, Any]:
    return dict(_LAST_GATE_META)


def parse_fusion_query(text: str) -> FusionQuery:
    raw = str(text or "").strip()
    eq = parse_entity_query(raw)
    from core.textile_terms import normalize_turkish as _nt

    raw_fold = _nt(raw)
    brand = str(eq.brands[0] if eq.brands else "")
    if raw_fold in _GENERIC_BRAND_QUERY:
        brand = ""
        eq.brands = []
    if not brand:
        try:
            from core.brand_aliases import resolve_brand_alias

            if raw_fold not in _GENERIC_BRAND_QUERY:
                brand = str(resolve_brand_alias(raw) or "")
        except Exception:
            brand = ""
    pattern_hits = parse_visual_pattern_concepts(raw)
    pattern_ids = [c.concept_id for c in pattern_hits]
    pattern_kinds = {c.concept_id: c.kind for c in pattern_hits}
    entities: list[str] = []
    motifs: list[str] = []
    persons: list[str] = []
    for e in eq.entities:
        lemma = str(e.canonical_name or "").strip().lower()
        if not lemma or lemma == brand or lemma in {brand.replace(" ", "_")}:
            continue
        if lookup_pattern_token(lemma) is not None:
            continue
        cat = str(e.category or category_for_lemma(lemma) or "")
        if lemma in _PERSON_LEMMAS or cat in {"person", "people", "face"}:
            if lemma not in persons:
                persons.append(lemma)
            if lemma not in entities:
                entities.append(lemma)
            continue
        if lemma in _MOTIF_LEMMAS or cat in _MOTIF_CATEGORIES:
            if lemma not in motifs:
                motifs.append(lemma)
        else:
            if lemma not in entities:
                entities.append(lemma)
    for c in pattern_hits:
        if c.kind == "motif" and c.concept_id not in motifs:
            motifs.append("flower" if c.concept_id == "floral" else c.concept_id)
    import re as _re
    for tok in _re.findall(r"[a-z0-9ğüşöçıİĞÜŞÖÇ]+", _nt(raw)):
        if lookup_pattern_token(tok) is not None:
            continue
        alias = _OBJECT_ALIASES.get(_nt(tok))
        if alias and alias not in entities:
            entities.append(alias)
    compiled = compile_visual_query(raw)
    colors = [c for c in compiled.colors if c]
    patterns = [cid for cid in pattern_ids if pattern_kinds.get(cid) == "pattern"]
    token_types = classify_fusion_token_types(
        raw, brand=brand, colors=colors, entities=entities, motifs=motifs,
        patterns=patterns, persons=persons,
    )
    return FusionQuery(
        raw=raw,
        brand=brand,
        colors=colors,
        entities=entities,
        motifs=motifs,
        patterns=patterns,
        persons=persons,
        token_types=token_types,
    )


def classify_fusion_token_types(
    text: str,
    *,
    brand: str = "",
    colors: list[str] | None = None,
    entities: list[str] | None = None,
    motifs: list[str] | None = None,
    patterns: list[str] | None = None,
    persons: list[str] | None = None,
) -> list[str]:
    """Ordered unique channel types for Fusion Query v2 (debug + tests)."""
    order: list[str] = []

    def _push(kind: str) -> None:
        if kind and kind not in order:
            order.append(kind)

    raw = str(text or "").strip()
    from core.textile_terms import normalize_turkish
    import re

    toks = [x for x in re.findall(r"[a-z0-9ğüşöçıİĞÜŞÖÇ]+", normalize_turkish(raw)) if len(x) >= 2]
    brand_l = str(brand or "").strip().lower()
    color_set = {str(c).lower() for c in (colors or [])}
    for tok in toks:
        folded = normalize_turkish(tok)
        if brand_l and (folded in brand_l or brand_l in folded):
            _push("brand")
            continue
        try:
            from core.brand_aliases import resolve_brand_alias
            if resolve_brand_alias(tok):
                _push("brand")
                continue
        except Exception:
            pass
        lemma = english_lemma(tok)
        pat = lookup_pattern_token(tok) or lookup_pattern_token(lemma)
        if pat is not None:
            _push("motif" if pat.kind == "motif" else "pattern")
            continue
        if folded in _PERSON_LEMMAS or lemma in _PERSON_LEMMAS:
            _push("person")
            continue
        if lemma in color_set or folded in {
            "kirmizi", "mavi", "yesil", "sari", "siyah", "beyaz", "pembe",
            "mor", "turuncu", "kahverengi", "gri", "red", "blue", "green",
            "yellow", "black", "white", "pink", "purple", "orange", "brown", "gray",
        }:
            _push("color")
            continue
        if lemma in _OBJECT_LEMMAS or folded in _OBJECT_LEMMAS or lemma in set(entities or []):
            _push("object")
            continue
        if lemma in _MOTIF_LEMMAS:
            _push("motif")
            continue
    if not order:
        if brand:
            _push("brand")
        if colors:
            _push("color")
        if patterns:
            _push("pattern")
        if motifs:
            _push("motif")
        if persons:
            _push("person")
        if entities:
            _push("object")
    return order


def _blob(result: Any) -> str:
    dbg = getattr(result, "debug", {}) or {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    parts = [
        str(getattr(result, "filename", "") or ""),
        str(getattr(result, "color_family", "") or ""),
        str(getattr(result, "pattern_family", "") or ""),
        str(dna.get("family") or ""),
        str(dna.get("motif") or ""),
        str(dna.get("color_family") or ""),
        " ".join(str(x) for x in (dna.get("dominant_colors") or [])),
        str(dbg.get("brand_parent") or ""),
        str(dbg.get("brand_child") or ""),
        " ".join(str(x) for x in (dbg.get("entity_query") or [])),
        " ".join(str(x) for x in (dbg.get("entity_debug_lines") or [])),
    ]
    return " ".join(parts).lower()


def _color_hit(query_colors: list[str], result: Any) -> float:
    if not query_colors:
        return 0.0
    blob = _blob(result)
    best = 0.0
    cf = str(getattr(result, "color_family", "") or "").lower()
    dbg = getattr(result, "debug", {}) or {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    cidx = tm.get("color_index") if isinstance(tm, dict) else None
    for col in query_colors:
        keys = _COLOR_GROUPS.get(str(col).lower(), {str(col).lower()})
        if cf and any(k in cf or cf in k for k in keys):
            best = max(best, 0.9)
            continue
        try:
            from core.color_index import color_index_match_score

            if cidx:
                best = max(best, float(color_index_match_score(str(col), cidx) or 0.0))
        except Exception:
            pass
        if any(k in blob for k in keys if len(k) >= 4):
            best = max(best, 0.55)
    return best


def _goi_object_labels(result: Any) -> list[str]:
    dbg = getattr(result, "debug", {}) or {}
    wrap = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    labels: list[str] = []
    for src in ("global_object_intelligence", "visual_concept_dna"):
        blob = wrap.get(src) if isinstance(wrap, dict) else None
        objs = (blob or {}).get("objects") if isinstance(blob, dict) else None
        if not isinstance(objs, list):
            continue
        for obj in objs:
            if isinstance(obj, dict):
                labels.append(str(obj.get("label") or obj.get("lemma") or obj.get("label_tr") or "").lower())
            else:
                labels.append(str(obj).lower())
    return labels


def _color_object_aligned(plan: FusionQuery, result: Any, feat: dict[str, Any]) -> bool:
    """True when color and object evidence can reasonably share one region/file."""
    asked = list(plan.entities) + list(plan.motifs)
    if not plan.colors or not asked:
        return False
    if feat.get("color", 0) < 0.5:
        return False
    kind = str(feat.get("entity_kind") or "")
    motif_ok = feat.get("motif", 0) >= 0.5
    if kind not in {"object_detector", "fusion", "visual_concept"} and not motif_ok:
        return False
    dbg = getattr(result, "debug", {}) or {}
    labels = _goi_object_labels(result)
    ents = {str(x).lower() for x in asked}
    rival = ()
    if ents & {"car", "araba"}:
        rival = ("flower", "rose", "bird", "cat", "dog")
    if ents & {"flower", "rose"}:
        rival = ("car", "truck", "bus")
    blob = " ".join(labels)
    if rival and any(r in blob for r in rival):
        return False
    if dbg.get("object_index_hit") and feat.get("color", 0) >= 0.85:
        return True
    if (kind in {"object_detector", "fusion", "visual_concept"} or motif_ok) and feat.get("color", 0) >= 0.85:
        return True
    return False


def _brand_hit(brand: str, result: Any) -> float:
    if not brand:
        return 0.0
    dbg = getattr(result, "debug", {}) or {}
    bd = getattr(result, "breakdown", {}) or {}
    if float(bd.get("brand_alias_score") or 0.0) >= 0.5:
        return min(1.0, float(bd.get("brand_alias_score") or 0.0))
    if dbg.get("brand_evidence") or dbg.get("brand_match"):
        return 1.0
    needle = str(brand).lower()
    name = str(getattr(result, "filename", "") or "").lower()
    if needle and needle in name:
        return 0.9
    return 0.0


def _motif_hit(motifs: list[str], result: Any) -> float:
    if not motifs:
        return 0.0
    blob = _blob(result)
    dbg = getattr(result, "debug", {}) or {}
    kind = str(dbg.get("entity_evidence_kind") or "")
    lines = " ".join(str(x) for x in (dbg.get("entity_debug_lines") or [])).lower()
    best = 0.0
    for m in motifs:
        aliases = {m, m.replace("_", " ")}
        if m in {"flower", "rose", "tulip", "daisy"}:
            aliases.update({"floral", "cicek", "çiçek", "flower"})
        if any(a and a in blob for a in aliases):
            best = max(best, 0.72)
        if any(a and a in lines for a in aliases):
            if kind in {"object_detector", "fusion"}:
                best = max(best, 0.92)
            else:
                best = max(best, 0.55)
    return best


def _entity_hit(entities: list[str], result: Any) -> tuple[float, str]:
    if not entities:
        return 0.0, "none"
    dbg = getattr(result, "debug", {}) or {}
    kind = str(dbg.get("entity_evidence_kind") or "none")
    score = float(dbg.get("entity_evidence_score") or dbg.get("entity_fusion_score") or 0.0)
    qents = {str(x).lower() for x in (dbg.get("entity_query") or [])}
    wanted = {str(x).lower() for x in entities}
    if not (qents & wanted) and wanted:
        # annotate used the raw query; still trust kind/score
        pass
    if kind == "object_detector":
        return max(score, 0.5), kind
    if kind == "fusion":
        return max(score, 0.45), kind
    if kind == "visual_concept":
        return max(score, 0.5), kind
    if kind == "openclip":
        return min(0.55, max(score, float(dbg.get("entity_semantic_score") or 0.0))), kind
    if kind == "pattern_dna":
        return min(0.55, max(score, 0.4)), kind
    return 0.0, kind or "none"


def _is_gender_only_query(raw: str) -> bool:
    from core.textile_terms import normalize_turkish

    return normalize_turkish(raw or "") in _GENDER_ONLY_QUERIES


def _person_hit(persons: list[str], result: Any, *, gender_only: bool = False) -> float:
    if not persons:
        return 0.0
    dbg = getattr(result, "debug", {}) or {}
    if dbg.get("face_gender_match") or dbg.get("face_match"):
        return 0.95
    if gender_only:
        # Bare "kadın"/"erkek": a generic person/COCO box is not male/female
        # evidence. entity_query is the *query* lemma list (always "person"),
        # so any detector hit was previously treated as gender and boosted to
        # 0.72+0.26*0.8 = 0.928. "kadın yüzü" keeps the paths below.
        return 0.0
    if float(dbg.get("gender_visual_score") or 0.0) >= 0.18:
        return 0.9
    if float(dbg.get("human_semantic_score") or 0.0) >= 0.18:
        return 0.85
    kind = str(dbg.get("entity_evidence_kind") or "")
    if kind in {"object_detector", "fusion"} and float(
        dbg.get("entity_evidence_score") or 0.0
    ) >= 0.45:
        qents = {str(x).lower() for x in (dbg.get("entity_query") or [])}
        if qents & _PERSON_LEMMAS or qents & {"woman", "female", "person", "man", "male"}:
            return 0.8
    return 0.0


def _pattern_hit(patterns: list[str], result: Any) -> tuple[float, dict[str, Any]]:
    if not patterns:
        return 0.0, {}
    from core.visual_pattern_query import parse_visual_pattern_concepts

    raw = " ".join(patterns)
    cons = parse_visual_pattern_concepts(raw)
    if not cons:
        # concept ids (ekose, stripe) rather than raw query tokens
        cons = []
        for p in patterns:
            hit = lookup_pattern_token(p)
            if hit is not None:
                cons.append(hit)
        cons = [c for c in cons if c is not None]
    sig = visual_pattern_signals(result, cons)
    if sig.get("positive"):
        return max(0.72, float(sig.get("visual") or 0.0)), sig
    if sig.get("negative"):
        return 0.05, sig
    return min(0.35, float(sig.get("visual") or 0.0)), sig


def score_fusion_result(result: Any, plan: FusionQuery) -> dict[str, Any]:
    brand_s = _brand_hit(plan.brand, result)
    color_s = _color_hit(plan.colors, result)
    motif_s = _motif_hit(plan.motifs, result)
    entity_s, entity_kind = _entity_hit(plan.entities, result)
    person_s = _person_hit(plan.persons, result, gender_only=_is_gender_only_query(plan.raw))
    pattern_s, pattern_sig = _pattern_hit(plan.patterns, result)
    ch = plan.channels()
    present = []
    real = []
    if "brand" in ch:
        present.append("brand")
        if brand_s >= 0.5:
            real.append("brand")
    if "color" in ch:
        present.append("color")
        if color_s >= 0.5:
            real.append("color")
    if "person" in ch:
        present.append("person")
        if person_s >= 0.5:
            real.append("person")
    if "entity" in ch:
        present.append("entity")
        if entity_s >= 0.45 and entity_kind in {"object_detector", "fusion", "visual_concept"}:
            real.append("entity")
        elif entity_s >= 0.45:
            present.append("entity_weak")
    if "motif" in ch:
        present.append("motif")
        if motif_s >= 0.5:
            real.append("motif")
    if "pattern" in ch:
        present.append("pattern")
        if pattern_s >= 0.5 and pattern_sig.get("positive"):
            real.append("pattern")
    asked = [k for k in ("brand", "color", "person", "entity", "motif", "pattern") if k in ch]
    entity_and = True
    if plan.entities:
        qlines = " ".join(str(x) for x in ((getattr(result, "debug", {}) or {}).get("entity_debug_lines") or [])).lower()
        entity_and = all(
            lemma in qlines or lemma in _blob(result)
            for lemma in plan.entities
        ) or (entity_kind in {"object_detector", "fusion"} and entity_s >= 0.45)
    full_and = bool(asked) and set(real) >= set(asked) and entity_and
    coverage = (len(real) / len(asked)) if asked else 1.0
    dbg = getattr(result, "debug", {}) or {}
    visual = max(
        float(dbg.get("clip_score") or 0.0),
        float(dbg.get("dino_score") or 0.0),
        float((getattr(result, "breakdown", {}) or {}).get("clip") or 0.0),
        float((getattr(result, "breakdown", {}) or {}).get("dino") or 0.0),
        float(pattern_sig.get("visual") or 0.0),
    )
    return {
        "brand": round(brand_s, 4),
        "color": round(color_s, 4),
        "entity": round(entity_s, 4),
        "entity_kind": entity_kind,
        "motif": round(motif_s, 4),
        "person": round(person_s, 4),
        "pattern": round(pattern_s, 4),
        "visual": round(visual, 4),
        "coverage": round(coverage, 4),
        "full_and": full_and,
        "real_channels": real,
        "clip_as_detector": entity_kind == "openclip",
        "pattern_visual_negative": bool(pattern_sig.get("negative")),
    }


def _stamp_fusion(row: Any, plan: FusionQuery, extra: dict[str, Any] | None = None) -> None:
    dbg = dict(getattr(row, "debug", {}) or {})
    payload = plan.to_debug()
    if extra:
        payload.update(extra)
    dbg["fusion_query_v2"] = payload
    row.debug = dbg


def apply_conceptual_person_priority(results: list[Any], plan: FusionQuery) -> list[Any]:
    """Person/visual concept first; filename/OCR fabric must not occupy top slots."""
    if not plan.persons or not results:
        return results
    gender_only = _is_gender_only_query(plan.raw)
    scored: list[tuple[tuple, int, Any]] = []
    for pos, row in enumerate(results):
        person_s = _person_hit(plan.persons, row, gender_only=gender_only)
        base = max(0.0, min(1.0, float(getattr(row, "score", 0.0) or 0.0)))
        bd = getattr(row, "breakdown", {}) or {}
        text_only = max(float(bd.get("filename_score") or 0.0), float(bd.get("ocr_score") or 0.0))
        new_score = base
        if person_s >= 0.5:
            new_score = max(new_score, 0.72 + 0.26 * person_s)
        else:
            new_score = min(new_score, 0.46 if text_only >= 0.50 else min(new_score, 0.52))
        row.score = new_score
        row.score_percent = round(new_score * 100.0, 1)
        dbg = dict(getattr(row, "debug", {}) or {})
        dbg["fusion_person_score"] = round(person_s, 4)
        dbg["fusion_person_priority"] = True
        row.debug = dbg
        key = (0 if person_s >= 0.5 else 1, -new_score, -person_s, pos)
        scored.append((key, pos, row))
    scored.sort(key=lambda x: x[0])
    return [row for _, _, row in scored]


def apply_exact_brand_lane(results: list[Any], plan: FusionQuery) -> list[Any]:
    """Keep evidenced brand hits; drop generic prints below the 0.90 brand floor."""
    if not results:
        return results
    for row in results:
        feat = score_fusion_result(row, plan)
        base = max(0.0, min(1.0, float(getattr(row, "score", 0.0) or 0.0)))
        if feat["brand"] >= 0.5:
            new_score = max(base, 0.90)
        else:
            new_score = min(base, 0.48)
        row.score = new_score
        row.score_percent = round(new_score * 100.0, 1)
        _stamp_fusion(row, plan, {**feat, "ranking_lane": "A"})
    results.sort(
        key=lambda r: (
            0 if (getattr(r, "debug", {}) or {}).get("fusion_query_v2", {}).get("brand", 0) >= 0.5 else 1,
            -float(getattr(r, "score", 0.0) or 0.0),
        )
    )
    return results


def _apply_visual_if_needed(results: list[Any], text: str, plan: FusionQuery) -> list[Any]:
    try:
        from core.visual_concept_dna import parse_search_bags

        if parse_search_bags(text).query_sense == "animal":
            return results
    except Exception:
        pass
    if not (plan.patterns or plan.motifs or is_visual_pattern_query(text)):
        return results
    reject = plan.ranking_lane == "B"
    return apply_visual_pattern_priority(results, text, reject_unmatched=reject)


def _finish_fusion(results: list[Any], text: str, plan: FusionQuery) -> list[Any]:
    global _LAST_GATE_META
    from core.search_evidence_gate import apply_search_evidence_gate

    gated, meta = apply_search_evidence_gate(
        results, text, multi_channel=bool(plan.multi_channel),
    )
    _LAST_GATE_META = dict(meta)
    for row in gated:
        dbg = dict(getattr(row, "debug", {}) or {})
        fq = dict(dbg.get("fusion_query_v2") or plan.to_debug())
        fq["accuracy_lane"] = meta.get("accuracy_lane") or fq.get("accuracy_lane")
        fq["empty_state"] = meta.get("empty_state") or ""
        dbg["fusion_query_v2"] = fq
        if meta.get("empty_state") and not dbg.get("kanit_yok_label"):
            dbg["kanit_yok_label"] = meta["empty_state"]
        row.debug = dbg
    return gated


def apply_fusion_query_v2(
    results: list[Any],
    text: str,
    *,
    object_db_path: str = "",
) -> list[Any]:
    """Additive rerank. Does not zero results; does not write index."""
    plan = parse_fusion_query(text)
    if not results:
        return _finish_fusion(results, text, plan)
    lane = plan.ranking_lane
    if lane == "A":
        return _finish_fusion(apply_exact_brand_lane(results, plan), text, plan)
    if not plan.multi_channel:
        for row in results:
            _stamp_fusion(row, plan)
        if plan.entities or plan.persons:
            annotate_search_results(results, text, object_db_path=object_db_path)
        if plan.persons:
            results = apply_conceptual_person_priority(results, plan)
        results = _apply_visual_if_needed(results, text, plan)
        return _finish_fusion(results, text, plan)
    if not (getattr(results[0], "debug", {}) or {}).get("entity_query") and (
        plan.entities or plan.motifs
    ):
        annotate_search_results(results, text, object_db_path=object_db_path)

    scored: list[tuple[float, int, Any]] = []
    any_full = False
    for pos, row in enumerate(results):
        feat = score_fusion_result(row, plan)
        base = max(0.0, min(1.0, float(getattr(row, "score", 0.0) or 0.0)))
        bonus = 0.0
        # Lane B: brand evidence must not promote a conceptual query.
        if lane != "B" and feat["brand"] >= 0.5:
            bonus += 0.02 * feat["brand"]
        if feat["color"] >= 0.5:
            if plan.entities or plan.motifs:
                if _color_object_aligned(plan, row, feat):
                    bonus += 0.06 * feat["color"]
                    feat["full_and"] = bool(feat.get("entity", 0) >= 0.5 or feat.get("motif", 0) >= 0.5)
                else:
                    bonus += 0.01 * feat["color"]
                    feat["full_and"] = False
            else:
                bonus += 0.045 * feat["color"]
        if feat["full_and"]:
            any_full = True
        if feat["entity_kind"] in {"object_detector", "fusion", "visual_concept"}:
            bonus += 0.06 * feat["entity"]
        elif feat["entity"] > 0:
            bonus += 0.02 * feat["entity"]
        if feat["motif"] >= 0.5:
            bonus += 0.05 * feat["motif"]
        if feat["pattern"] >= 0.5:
            bonus += 0.08 * feat["pattern"]
        elif feat.get("pattern_visual_negative"):
            bonus -= 0.04
        if feat["person"] >= 0.5:
            bonus += 0.08 * feat["person"]
        elif plan.persons:
            bonus -= 0.04
        if feat["full_and"]:
            bonus += 0.04
        elif feat["coverage"] >= 0.5:
            bonus += 0.015 * feat["coverage"]
        new_score = min(1.0, base + bonus)
        if plan.brand and feat["brand"] < 0.5:
            # Marka sorgusunda markasız kaydı markalıların üzerine çıkarma.
            new_score = min(new_score, base + 0.02)
        if feat.get("pattern_visual_negative"):
            new_score = min(new_score, 0.42)
        if plan.persons and feat["person"] < 0.5:
            new_score = min(new_score, 0.46)
        if plan.colors and plan.entities and feat.get("color", 0) < 0.5:
            new_score = min(new_score, 0.50)
        row.score = new_score
        row.score_percent = round(new_score * 100.0, 1)
        dbg = dict(getattr(row, "debug", {}) or {})
        dbg["fusion_query_v2"] = {**plan.to_debug(), **feat, "second_tier": False}
        row.debug = dbg
        person_rank = 0 if (plan.persons and feat["person"] >= 0.5) else (1 if plan.persons else 0)
        scored.append((person_rank, new_score, pos, row))
    if not any_full:
        for _pr, _s, _p, row in scored:
            dbg = dict(getattr(row, "debug", {}) or {})
            fq = dict(dbg.get("fusion_query_v2") or {})
            fq["second_tier"] = True
            dbg["fusion_query_v2"] = fq
            row.debug = dbg
        scored.sort(
            key=lambda x: (
                x[0],
                -(x[3].debug or {}).get("fusion_query_v2", {}).get("coverage", 0),
                -x[1],
                x[2],
            )
        )
    else:
        scored.sort(key=lambda x: (x[0], -x[1], x[2]))
    out = [row for _, _, _, row in scored]
    if plan.persons:
        out = apply_conceptual_person_priority(out, plan)
    out = _apply_visual_if_needed(out, text, plan)
    return _finish_fusion(out, text, plan)
