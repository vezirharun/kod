"""Vezir Entity Brain v2 — merkezi varlık sorgu ve kanıt katmanı.

Amaç: insan, hayvan, araç, marka/model ve nesne sorgularını Pattern aramasından
ayrı bir kanıt kanalı olarak yönetmek. İndeks sırasında üretilen entity_evidence
ve global_object_intelligence verisini kullanır; yeni model zorunlu kılmaz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.entity_evidence import entity_evidence_strength, extract_entity_evidence
from core.universal_visual_intel import ONTOLOGY, parse_universal_query


@dataclass
class EntityQueryPlan:
    raw: str
    nodes: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    relation: str = "single"
    primary: str = ""
    families: list[str] = field(default_factory=list)

    @property
    def composite(self) -> bool:
        return len(self.required) >= 2


def _is_entity_node(node_id: str) -> bool:
    meta = ONTOLOGY.get(str(node_id)) or {}
    return bool(meta) and meta.get("parent", "") is not None


def build_entity_query_plan(text: str) -> EntityQueryPlan:
    uq = parse_universal_query(text)
    nodes: list[str] = []
    for node in [uq.node_id, *uq.required, *getattr(uq, "open_ids", [])]:
        node = str(node or "")
        if node and node not in nodes:
            if node in ONTOLOGY or str(node).startswith("open:"):
                nodes.append(node)

    from core.visual_concept import compile_visual_query

    for c in compile_visual_query(text).concepts:
        cid = str(c.concept_id or "")
        if cid and cid not in nodes:
            nodes.append(cid)

    # Keep only concrete/entity concepts here. Pattern-only terms continue through
    # Pattern Intelligence and do not accidentally become object constraints.
    entity_nodes = [n for n in nodes if _is_entity_node(n) or str(n).startswith("open:")]
    required = list(dict.fromkeys(entity_nodes))

    relation = "single"
    low = str(text or "").casefold()
    if len(required) >= 2:
        if any(token in low for token in ("karışık", "karisik", "ve", "+", "ile", "birlikte")):
            relation = "AND"
        else:
            relation = "AND"

    families: list[str] = []
    for node in required:
        cur = node
        seen: set[str] = set()
        while cur in ONTOLOGY and cur not in seen:
            seen.add(cur)
            parent = str(ONTOLOGY[cur].get("parent") or "")
            if not parent:
                break
            cur = parent
        if cur and cur not in families:
            families.append(cur)

    return EntityQueryPlan(
        raw=str(text or ""),
        nodes=nodes,
        required=required,
        relation=relation,
        primary=str(uq.node_id or ""),
        families=families,
    )


def evidence_for_record(rec: dict[str, Any]) -> set[str]:
    return extract_entity_evidence(rec)


def score_record(rec: dict[str, Any], plan: EntityQueryPlan) -> tuple[float, list[str], list[str]]:
    """Return (score, matched, missing) using persisted entity evidence only."""
    if not plan.required:
        return 0.0, [], []
    matched: list[str] = []
    missing: list[str] = []
    scores: list[float] = []
    for node in plan.required:
        score = entity_evidence_strength(rec, node)
        if score >= 0.18:
            matched.append(node)
            scores.append(score)
        else:
            missing.append(node)
    if not scores:
        return 0.0, matched, missing
    # Composite entity queries require every requested entity. This prevents a
    # picture containing only a leopard from passing "yılan ve leopar".
    if plan.composite and missing:
        return 0.0, matched, missing
    return min(scores) if plan.composite else max(scores), matched, missing
