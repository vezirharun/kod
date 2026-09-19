"""Görsel Nesne/Kavram DNA — persisted per file, separate from Pattern DNA.

Index writes object_index.db + texture_map keys. Search is read-only.
CLIP is never a detector: visual_concept rows are concept evidence, not boxes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.textile_terms import normalize_turkish
from core.visual_concept import category_for_lemma, english_lemma

_CAT_TR = {
    "fruit": "meyve",
    "animal": "hayvan",
    "person": "insan",
    "vehicle": "araç",
    "plant": "bitki",
    "body": "yüz",
    "accessory": "aksesuar",
    "clothing": "giysi",
}

_PATTERN_TOKS = frozenset(
    {
        "ekose", "kareli", "tartan", "plaid", "cizgi", "cizgili", "stripe",
        "geometrik", "geometric", "yilan", "snake", "leopar", "leopard",
        "zebra", "kamuflaj", "camouflage", "paisley", "cicek", "floral",
        "gul", "rose", "yaprak", "leaf", "leaves", "foliage",
        "barok", "baroque", "rokoko", "rococo",
        "suluboya", "watercolor",
        "etnik", "ethnic_print",
        "puantiye", "damask", "jakarli", "jakarli",
        "firca", "brushstroke", "sicrama", "paint_splatter", "splatter",
        "vintage", "retro",
    }
)
_DESEN_TOKS = frozenset({"desen", "deseni", "print", "pattern"})
_BRAND_GENERIC = frozenset({"marka", "brand", "logo"})


def _tokens(text: str) -> list[str]:
    try:
        from core.visual_concept import rewrite_visual_phrases

        blob = rewrite_visual_phrases(text)
    except Exception:
        blob = normalize_turkish(text or "")
    return [
        x for x in re.findall(r"[a-z0-9_ğüşöçıİĞÜŞÖÇ]+", blob)
        if len(x) >= 2
    ]


def _cat_tr(lemma: str) -> str:
    cat = str(category_for_lemma(lemma) or "")
    return _CAT_TR.get(cat, cat or "")


def _label_tr(lemma: str, explicit: str = "") -> str:
    if explicit:
        return str(explicit)
    try:
        from core.search_evidence_gate import concept_label_tr

        return concept_label_tr(lemma)
    except Exception:
        return lemma.capitalize()


def build_visual_concept_dna(
    *,
    objects: list[dict[str, Any]] | None = None,
    concepts: list[dict[str, Any]] | None = None,
    backend: str = "",
) -> dict[str, Any]:
    """Build the persisted blob. objects = detector/face; concepts ≠ detected."""
    objs: list[dict[str, Any]] = []
    for raw in objects or []:
        if not isinstance(raw, dict):
            continue
        lemma = str(raw.get("canonical_name") or english_lemma(str(raw.get("label") or "")) or "").strip()
        if not lemma:
            continue
        conf = float(raw.get("confidence") or 0.0)
        src = str(raw.get("evidence_source") or raw.get("evidence") or "object_detector")
        detected = src in {"object_detector", "face", "rtdetr", "rt-detr"}
        objs.append(
            {
                "lemma": lemma,
                "label": str(raw.get("label") or lemma),
                "label_tr": str(raw.get("label_tr") or _label_tr(lemma)),
                "category": str(raw.get("category") or category_for_lemma(lemma) or ""),
                "category_tr": _cat_tr(lemma),
                "confidence": round(max(0.0, min(1.0, conf)), 4),
                "evidence": "object_detector" if detected else src,
                "detected": bool(detected),
                "bbox": list(raw.get("bbox") or []) or None,
            }
        )
    cons: list[dict[str, Any]] = []
    for raw in concepts or []:
        if not isinstance(raw, dict):
            continue
        lemma = str(raw.get("canonical_name") or english_lemma(str(raw.get("label") or raw.get("lemma") or "")) or "").strip()
        if not lemma:
            continue
        conf = float(raw.get("confidence") or 0.0)
        cons.append(
            {
                "lemma": lemma,
                "label": str(raw.get("label") or lemma),
                "label_tr": str(raw.get("label_tr") or _label_tr(lemma)),
                "category": str(raw.get("category") or category_for_lemma(lemma) or ""),
                "category_tr": _cat_tr(lemma),
                "confidence": round(max(0.0, min(1.0, conf)), 4),
                "evidence": "visual_concept",
                "detected": False,
            }
        )
    best = max(objs + cons, key=lambda x: float(x.get("confidence") or 0.0), default=None)
    names = [x["label_tr"] for x in objs] or [x["label_tr"] for x in cons]
    cat = (best or {}).get("category_tr") or ""
    kav = (best or {}).get("label_tr") or ""
    gvn = float((best or {}).get("confidence") or 0.0)
    summary = ""
    if names or kav:
        parts = []
        if names:
            parts.append("Nesneler: " + "; ".join(dict.fromkeys(names)))
        if cat:
            parts.append(f"Kategori: {cat}")
        if kav:
            parts.append(f"Görsel kavram: {kav}")
        if gvn:
            parts.append(f"Güven: {gvn:.2f}")
        summary = "; ".join(parts)
    return {
        "version": 1,
        "clip_as_detector": False,
        "backend": str(backend or ""),
        "objects": objs,
        "concepts": cons,
        "summary": summary,
    }


def read_visual_concept_dna(result: Any) -> dict[str, Any]:
    dbg = getattr(result, "debug", {}) or {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    if not tm and isinstance(result, dict):
        tm = result.get("texture_map") if isinstance(result.get("texture_map"), dict) else {}
        dbg = result
    blob = tm.get("visual_concept_dna") if isinstance(tm, dict) else None
    if not isinstance(blob, dict):
        goi = tm.get("global_object_intelligence") if isinstance(tm, dict) else None
        if isinstance(goi, dict) and (goi.get("objects") or goi.get("concepts") or goi.get("visual_concept_dna")):
            inner = goi.get("visual_concept_dna")
            if isinstance(inner, dict):
                blob = inner
            else:
                blob = build_visual_concept_dna(
                    objects=list(goi.get("objects") or []),
                    concepts=list(goi.get("concepts") or []),
                    backend=str(goi.get("backend") or ""),
                )
    return blob if isinstance(blob, dict) else {}


def dna_hits_for_lemmas(dna: dict[str, Any], lemmas: tuple[str, ...]) -> list[dict[str, Any]]:
    wanted = {str(x).lower() for x in lemmas if x}
    if not wanted or not dna:
        return []
    hits = []
    for row in list(dna.get("objects") or []) + list(dna.get("concepts") or []):
        if not isinstance(row, dict):
            continue
        lemma = str(row.get("lemma") or "").lower()
        lab = str(row.get("label") or "").lower()
        tr = str(row.get("label_tr") or "").lower()
        if lemma in wanted or lab in wanted or any(w in (lemma, lab, tr) for w in wanted):
            hits.append(row)
            continue
        try:
            if english_lemma(lemma) in wanted or english_lemma(lab) in wanted:
                hits.append(row)
        except Exception:
            pass
    return hits


def detected_label(lemma: str, confidence: float) -> str:
    return f"{_label_tr(lemma)} tespit edildi — %{int(round(confidence * 100))}"


def missing_label(lemma: str) -> str:
    return f"{_label_tr(lemma)} — Kanıt yok"


_ANIMAL_SENSE = frozenset({"hayvan", "hayvani", "foto", "fotograf", "gercek", "real"})
_SKIN_SENSE = frozenset({"deri", "derisi", "skin"})
_PERSON_UVI = frozenset({"person", "female_person", "male_person", "child", "girl"})
_PATTERN_UVI = frozenset({
    "leopard", "zebra", "snake", "tiger", "flower", "rose", "daisy", "tulip",
    "floral", "leaf", "paisley",
    "baroque_pattern", "plaid", "ethnic_print", "watercolor",
    "brushstroke", "paint_splatter", "rococo",
})


@dataclass
class SearchBags:
    raw: str
    pattern: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    parent_objects: list[str] = field(default_factory=list)
    brand: str = ""
    pattern_weighted: bool = False
    query_sense: str = "both"
    bags: tuple[str, ...] = ()

    @property
    def need_object_retrieval(self) -> bool:
        if self.pattern_weighted and not self.objects and not self.persons:
            return False
        return bool(self.objects or self.persons or self.parent_objects)


def parse_search_bags(text: str) -> SearchBags:
    """Separate retrieval bags: desen / nesne / insan / marka."""
    raw = str(text or "").strip()
    toks = _tokens(raw)
    folded = [normalize_turkish(t) for t in toks]
    has_desen = any(t in _DESEN_TOKS for t in folded)
    pattern = [t for t in folded if t in _PATTERN_TOKS]
    animal_sense = any(t in _ANIMAL_SENSE for t in folded)
    skin_sense = any(t in _SKIN_SENSE for t in folded)
    pattern_weighted = bool((has_desen or skin_sense) and pattern and not animal_sense)
    query_sense = "both"
    if animal_sense and not (has_desen or skin_sense):
        query_sense = "animal"
    elif (has_desen or skin_sense) and not animal_sense:
        query_sense = "pattern"
    brand = ""
    if folded and not all(t in _BRAND_GENERIC for t in folded):
        try:
            from core.brand_aliases import resolve_brand_alias

            brand = str(resolve_brand_alias(raw) or "")
            if not brand:
                for t in toks:
                    brand = str(resolve_brand_alias(t) or "")
                    if brand:
                        break
        except Exception:
            brand = ""
    objects: list[str] = []
    persons: list[str] = []
    try:
        from core.search_evidence_gate import object_query_lemmas

        for can, _lab in object_query_lemmas(raw):
            if can in {"woman", "man", "person", "face", "child"}:
                if can not in persons:
                    persons.append(can)
            elif can not in objects:
                objects.append(can)
    except Exception:
        pass
    # Fruit / open lemmas not in the person/pattern skip list.
    for tok in folded:
        if tok in _PATTERN_TOKS or tok in _DESEN_TOKS or tok in _BRAND_GENERIC:
            continue
        try:
            lemma = english_lemma(tok)
        except Exception:
            lemma = tok
        if lemma and lemma not in objects and lemma not in persons and tok not in _PATTERN_TOKS:
            cat = str(category_for_lemma(lemma) or "")
            if cat in {"fruit", "animal", "vehicle", "plant"} and lemma not in {
                "flower", "rose", "stripe", "paisley", "leaf",
            }:
                if lemma not in objects:
                    objects.append(lemma)
    if not pattern_weighted:
        _print_obj = {
            "leopar": "leopard", "leopard": "leopard",
            "zebra": "zebra",
            "yilan": "snake", "snake": "snake",
        }
        for tok in folded:
            mapped = _print_obj.get(tok)
            if mapped and mapped not in objects:
                objects.append(mapped)
    parent_objects: list[str] = []
    try:
        from core.universal_visual_intel import ONTOLOGY, parse_universal_query
        from core.visual_concept import detector_labels_for_lemma

        uq = parse_universal_query(raw)
        nid = str(getattr(uq, "node_id", "") or "")
        pattern_only = bool(pattern) and all(
            t in _PATTERN_TOKS or t in _DESEN_TOKS for t in folded
        )
        expand_parents = bool(nid) and nid not in _PERSON_UVI and (
            nid not in _PATTERN_UVI or query_sense == "animal"
        )
        if pattern_only:
            expand_parents = False
        if expand_parents:
            if nid not in objects and nid not in persons:
                objects.append(nid)
            parent = str((ONTOLOGY.get(nid) or {}).get("parent") or "")
            if parent and parent not in _PERSON_UVI:
                labs = tuple(detector_labels_for_lemma(parent) or ())
                for lab in labs:
                    if lab and lab not in objects and lab not in parent_objects:
                        parent_objects.append(lab)
        for lemma in list(objects):
            for lab in detector_labels_for_lemma(lemma) or ():
                if lab and lab not in objects:
                    objects.append(lab)
            if lemma in _PATTERN_UVI and query_sense != "animal":
                continue
            hyp_parent = str((ONTOLOGY.get(lemma) or {}).get("parent") or "")
            for lab in detector_labels_for_lemma(hyp_parent) or ():
                if lab and lab not in objects and lab not in parent_objects:
                    parent_objects.append(lab)
    except Exception:
        pass
    bags: list[str] = []
    if pattern:
        bags.append("pattern")
    if objects:
        bags.append("object")
    if persons:
        bags.append("person")
    if brand:
        bags.append("brand")
    return SearchBags(
        raw=raw,
        pattern=pattern,
        objects=objects,
        persons=persons,
        parent_objects=parent_objects,
        brand=brand,
        pattern_weighted=pattern_weighted,
        query_sense=query_sense,
        bags=tuple(bags),
    )


def lookup_concept_file_ids(object_db_path: str, lemmas: list[str], *, limit: int = 400) -> list[int]:
    """Read-only object_index lookup. Never writes patterns.db."""
    path = str(object_db_path or "").strip()
    wanted = [str(x) for x in lemmas if str(x).strip() and not str(x).startswith("open:")]
    if not path or not wanted:
        return []
    from pathlib import Path

    if not Path(path).is_file():
        return []
    try:
        from core.object_index import ObjectIndexStore

        store = ObjectIndexStore(path, readonly=True)
        ids = list(store.search_label(wanted, limit=limit))
        extra = store.search_concepts(wanted, limit=limit)
        seen = set(ids)
        for i in extra:
            if i not in seen:
                ids.append(i)
                seen.add(i)
        return ids[: int(limit)]
    except Exception:
        return []
