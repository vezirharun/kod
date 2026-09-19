"""Varlık (nesne/hayvan/insan) alias çözümleyici.

Marka tarafındaki normalize + alias yaklaşımının görsel varlık karşılığıdır.
Kaynak gerçekliği Universal Visual Intelligence ontolojisinden alınır; burada
ikinci bir dev ontoloji kopyası tutulmaz.
"""
from __future__ import annotations

from typing import Iterable

from core.textile_terms import normalize_turkish


def _ontology():
    from core.universal_visual_intel import ONTOLOGY
    return ONTOLOGY


def normalize_entity_key(value: str) -> str:
    return normalize_turkish(str(value or "")).strip().replace(" ", "_")


def resolve_entity_alias(value: str) -> str:
    norm = normalize_turkish(str(value or "")).strip()
    if not norm:
        return ""
    # Prefer the existing UVI parser for multi-word/compound aliases.
    try:
        from core.universal_visual_intel import parse_universal_query
        uq = parse_universal_query(norm)
        if uq.node_id and not str(uq.node_id).startswith("open:"):
            return str(uq.node_id)
    except Exception:
        pass
    for node_id, meta in _ontology().items():
        if norm == normalize_turkish(node_id):
            return node_id
        for alias in meta.get("aliases") or ():
            if norm == normalize_turkish(alias):
                return node_id
    return ""


def expand_entity_terms(value: str) -> list[str]:
    node_id = resolve_entity_alias(value)
    if not node_id:
        return []
    ontology = _ontology()
    meta = ontology.get(node_id) or {}
    terms = [node_id]
    terms.extend(str(x) for x in (meta.get("aliases") or ()) if str(x).strip())
    # Parent aliases make family queries useful without turning a leaf query
    # into a broad unrelated search.
    parent = str(meta.get("parent") or "")
    if parent and parent in ontology:
        terms.append(parent)
    return list(dict.fromkeys(normalize_turkish(x) for x in terms if x))


def is_entity_query(value: str) -> bool:
    return bool(resolve_entity_alias(value))


def canonical_entity_label(node_id: str) -> str:
    meta = _ontology().get(str(node_id)) or {}
    aliases = tuple(meta.get("aliases") or ())
    return str(aliases[0] if aliases else node_id).strip()
