"""Small deterministic query planner.

The production NLP/LLM parser can feed this contract. This layer guarantees
that downstream search receives structured concepts instead of raw text only.
"""

from __future__ import annotations
import re
from .intelligence_contract import QueryConcept, QueryPlan

_RELATIONS = [
    (" ve ", "AND"),
    (" ile ", "AND"),
    (" karışık", "COMBINATION"),
]

def plan_query(text: str) -> QueryPlan:
    raw = (text or "").strip()
    low = raw.lower()
    relation = "AND"
    normalized = low
    for token, rel in _RELATIONS:
        if token in normalized:
            relation = rel
            normalized = normalized.replace(token, "|")
    chunks = [x.strip() for x in re.split(r"[|,+]", normalized) if x.strip()]

    concepts = []
    for chunk in chunks:
        kind = "semantic"
        if any(k in chunk for k in ("marka", "togg", "mercedes", "bmw")):
            kind = "brand"
        elif any(k in chunk for k in ("kadın", "erkek", "çocuk", "insan", "yüz")):
            kind = "entity"
        elif any(k in chunk for k in ("desen", "geometrik", "mermer", "boya", "çiçek")):
            kind = "pattern"
        concepts.append(QueryConcept(text=chunk, kind=kind, confidence=0.5))

    return QueryPlan(raw=raw, concepts=concepts, relation=relation)
