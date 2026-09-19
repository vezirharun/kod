"""Kanıt tabanlı görsel varlık çıkarımı.

Marka kanıtındaki prensibi uygular: açık kategori, indekslenmiş görsel nesne
metası, semantic objects ve kontrollü alias/metadata kanıtları tek kümeye iner.
Dosya adı tek başına güçlü görsel kanıt sayılmaz.
"""
from __future__ import annotations

import json
from typing import Any

from core.entity_aliases import resolve_entity_alias, normalize_entity_key


def _walk_strings(value: Any, max_depth: int = 5) -> list[str]:
    if value is None or max_depth < 0:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_walk_strings(v, max_depth - 1))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for v in value:
            out.extend(_walk_strings(v, max_depth - 1))
        return out
    s = str(value).strip()
    return [s] if s else []


def _add_alias(found: set[str], value: Any) -> None:
    node = resolve_entity_alias(str(value or ""))
    if not node:
        return
    found.add(node)
    # Persist the ontology ancestry as well.  A detected `cat` therefore
    # becomes evidence for `cat -> mammal -> animal -> entity`, allowing
    # family/category queries to use the same indexed evidence.
    try:
        from core.universal_visual_intel import ONTOLOGY
        seen = {node}
        current = node
        while current in ONTOLOGY:
            parent = str(ONTOLOGY[current].get("parent") or "")
            if not parent or parent in seen:
                break
            found.add(parent)
            seen.add(parent)
            current = parent
    except Exception:
        pass


def extract_entity_evidence(rec: dict[str, Any]) -> set[str]:
    """Return canonical entity IDs explicitly supported by an indexed record."""
    if not isinstance(rec, dict):
        return set()
    found: set[str] = set()
    tm = rec.get("texture_map") or {}
    if isinstance(tm, str):
        try:
            tm = json.loads(tm or "{}")
        except Exception:
            tm = {}
    if not isinstance(tm, dict):
        tm = {}

    # 1) Persisted global-object detections (strongest machine evidence).
    object_meta = tm.get("global_object_intelligence") or rec.get("global_object_intelligence") or {}
    if isinstance(object_meta, dict):
        for obj in object_meta.get("objects") or ():
            if isinstance(obj, dict):
                _add_alias(found, obj.get("label"))
                _add_alias(found, obj.get("label_tr"))
                try:
                    from core.visual_concept import english_lemma

                    lemma = english_lemma(str(obj.get("label") or ""))
                    if lemma:
                        found.add(lemma)
                        found.add(f"open:{lemma.replace(' ', '_')}")
                except Exception:
                    pass

    # 2) Fine/global CLIP concepts that are not represented by COCO boxes
    # (e.g. accessory/brooch/baroque). They are secondary evidence but become
    # persistent so text search can reuse the one-time index analysis.
    concept_meta = tm.get("global_object_intelligence") or rec.get("global_object_intelligence") or {}
    if isinstance(concept_meta, dict):
        for concept in concept_meta.get("concepts") or ():
            if isinstance(concept, dict):
                _add_alias(found, concept.get("label"))
                _add_alias(found, concept.get("label_tr"))
                try:
                    from core.visual_concept import english_lemma

                    lemma = english_lemma(str(concept.get("label") or ""))
                    if lemma:
                        found.add(lemma)
                        found.add(f"open:{lemma.replace(' ', '_')}")
                except Exception:
                    pass

    # 3) Explicit semantic object labels / category metadata.
    sem = tm.get("semantic_tags") or {}
    if isinstance(sem, dict):
        for value in sem.get("objects") or ():
            _add_alias(found, value)
        for value in sem.get("entities") or ():
            _add_alias(found, value)

    for key in ("manual_category_path", "category_path"):
        path = str(rec.get(key) or tm.get(key) or "").strip()
        if path:
            parts = [p.strip() for p in path.split("/") if p.strip()]
            for part in parts:
                _add_alias(found, part)

    # 3) Çoklu kullanıcı nitelikleri. Örn. Araç Detayı/SUV +
    # Araba Markası/Togg + Araba Modeli/T10X tek görselde birlikte saklanır.
    for attr in tm.get("entity_attributes") or []:
        if not isinstance(attr, dict):
            continue
        for value in (attr.get("path"), attr.get("label"), attr.get("source")):
            _add_alias(found, value)
            if "/" in str(value or ""):
                for part in str(value).split("/"):
                    _add_alias(found, part)

    # 4) Explicit user/admin/gold labels and auto tags.
    for key in ("entity", "entity_id", "object", "object_type", "object_label"):
        _add_alias(found, rec.get(key) or tm.get(key))
    for value in tm.get("auto_tags") or ():
        _add_alias(found, value)

    return found


def entity_evidence_strength(rec: dict[str, Any], node_id: str) -> float:
    """Return the strongest persisted evidence for one canonical entity."""
    if node_id not in extract_entity_evidence(rec):
        return 0.0
    tm = rec.get("texture_map") or {}
    if not isinstance(tm, dict):
        tm = {}
    if tm.get("user_labeled") or tm.get("category_source") in ("manual_user", "gold_dataset"):
        return 1.0

    object_meta = tm.get("global_object_intelligence") or rec.get("global_object_intelligence") or {}
    best = 0.0
    if isinstance(object_meta, dict):
        for obj in object_meta.get("objects") or ():
            if not isinstance(obj, dict):
                continue
            labels = {
                normalize_entity_key(str(obj.get("label") or "")),
                normalize_entity_key(str(obj.get("label_tr") or "")),
            }
            if node_id in labels or resolve_entity_alias(str(obj.get("label") or "")) == node_id or resolve_entity_alias(str(obj.get("label_tr") or "")) == node_id:
                try:
                    best = max(best, float(obj.get("confidence") or 0.0))
                except (TypeError, ValueError):
                    pass
    if best:
        return min(1.0, max(0.18, best))
    if object_meta:
        return 0.97
    return 0.94
