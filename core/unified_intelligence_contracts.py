"""
V12.4.13 Unified Intelligence Contracts
Safe decision contracts: no unsupported guessing, explicit contradiction and calibration.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Evidence:
    source: str
    label: str
    score: float
    polarity: str = "positive"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clipped(self):
        return max(0.0, min(1.0, float(self.score)))

@dataclass
class Decision:
    label: str
    confidence: float
    accepted: bool
    contradictory: bool
    missing: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    reason: str = ""

@dataclass
class QueryRequirement:
    label: str
    kind: str = "semantic"
    required: bool = True
    threshold: float = 0.72
