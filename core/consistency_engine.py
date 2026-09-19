"""V12.4.12 deterministic evidence fusion, contradiction and calibration helpers."""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
from .intelligence_contract import Evidence, Decision

def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

DEFAULT_SOURCE_WEIGHTS = {
    "user_positive": 1.00,
    "user_negative": 1.00,
    "entity": 0.95,
    "object": 0.92,
    "brand": 0.90,
    "pattern": 0.70,
    "clip": 0.65,
    "dino": 0.65,
    "metadata": 0.40,
}

def fuse(evidence: Iterable[Evidence], threshold: float = 0.72) -> Decision:
    ev = list(evidence)
    if not ev:
        return Decision("", 0.0, False, explanation="Kanıt bulunamadı.")

    pos: Dict[str, float] = {}
    neg: Dict[str, float] = {}
    by_label: Dict[str, List[Evidence]] = {}
    for e in ev:
        w = DEFAULT_SOURCE_WEIGHTS.get(e.source, 0.5)
        s = _clip(e.score) * w
        by_label.setdefault(e.label, []).append(e)
        if e.polarity == "negative":
            neg[e.label] = max(neg.get(e.label, 0.0), s)
        else:
            pos[e.label] = max(pos.get(e.label, 0.0), s)

    candidates = []
    for label, score in pos.items():
        candidates.append((max(0.0, score - neg.get(label, 0.0)), label))
    candidates.sort(reverse=True)
    score, label = candidates[0]

    second = candidates[1][0] if len(candidates) > 1 else 0.0
    contradictory = second >= score * 0.92 if score else False
    accepted = score >= threshold and not contradictory

    parts = [f"{label}: {score:.2f}"]
    if contradictory:
        parts.append("çelişen güçlü kanıt var")
    elif not accepted:
        parts.append("güven eşiği aşılmadı")

    return Decision(
        label=label,
        confidence=_clip(score),
        accepted=accepted,
        contradictory=contradictory,
        evidence=ev,
        explanation="; ".join(parts),
    )

def require_all(decisions: Iterable[Decision]) -> Tuple[bool, List[str]]:
    missing = [d.label for d in decisions if not d.accepted]
    return not missing, missing
