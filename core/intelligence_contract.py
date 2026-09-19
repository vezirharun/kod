"""V12.4.12 canonical contracts.

Pure-Python, dependency-light data contracts for the unified decision layer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class QueryConcept:
    text: str
    kind: str
    path: List[str] = field(default_factory=list)
    required: bool = True
    confidence: float = 0.0
    aliases: List[str] = field(default_factory=list)

@dataclass
class Evidence:
    source: str
    label: str
    score: float
    polarity: str = "positive"
    metadata: Dict[str, object] = field(default_factory=dict)

@dataclass
class QueryPlan:
    raw: str
    concepts: List[QueryConcept] = field(default_factory=list)
    relation: str = "AND"
    intent: str = "semantic"
    unknown_terms: List[str] = field(default_factory=list)

@dataclass
class Decision:
    label: str
    confidence: float
    accepted: bool
    contradictory: bool = False
    evidence: List[Evidence] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    explanation: str = ""
