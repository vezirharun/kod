"""
V12.4.13 Unified Decision Engine
- source-aware evidence weighting
- hard negative evidence
- contradiction detection
- margin requirement
- calibrated confidence
- required-concept gate
- deterministic, auditable decisions
"""
from __future__ import annotations
from typing import Dict, Iterable, List
from .unified_intelligence_contracts import Evidence, Decision, QueryRequirement

SOURCE_WEIGHT = {
    "user_positive": 1.00,
    "user_negative": 1.00,
    "entity": 0.96,
    "face": 0.95,
    "object": 0.94,
    "vehicle": 0.94,
    "brand": 0.92,
    "ocr": 0.82,
    "pattern_dna": 0.82,
    "dino": 0.72,
    "clip": 0.70,
    "pattern": 0.68,
    "metadata": 0.35,
}

def _c(v): return max(0.0, min(1.0, float(v)))

def _source_factor(source: str) -> float:
    return SOURCE_WEIGHT.get(source, 0.50)

def _calibrate(raw: float, support_count: int, contradiction: bool, margin: float) -> float:
    # Conservative calibration: more independent support helps; contradiction and
    # tiny winner/runner-up margins reduce confidence.
    raw=_c(raw)
    support_bonus=min(0.08, max(0, support_count-1)*0.025)
    margin_bonus=min(0.06, max(0.0, margin)*0.12)
    penalty=(0.20 if contradiction else 0.0)
    return _c(raw + support_bonus + margin_bonus - penalty)

def decide(evidence: Iterable[Evidence], requirements: Iterable[QueryRequirement]=(),
           accept_threshold: float=0.72, contradiction_ratio: float=0.92,
           min_margin: float=0.08) -> Decision:
    ev=list(evidence)
    if not ev:
        return Decision("",0.0,False,False,reason="Kanıt yok.")

    pos: Dict[str,float]={}
    neg: Dict[str,float]={}
    # Count independent evidence channels, not duplicate records from the same source.
    support_sources: Dict[str,set[str]]={}
    labels=set()

    for e in ev:
        s=_c(e.score)*_source_factor(e.source)
        labels.add(e.label)
        if e.polarity=="negative":
            neg[e.label]=max(neg.get(e.label,0.0),s)
        else:
            pos[e.label]=max(pos.get(e.label,0.0),s)
            support_sources.setdefault(e.label, set()).add(e.source)

    scored=[]
    for label,score in pos.items():
        scored.append((max(0.0,score-neg.get(label,0.0)),label))
    scored.sort(reverse=True)
    if not scored:
        return Decision("",0.0,False,False,reason="Yalnız negatif kanıt var.")

    winner,wlabel=scored[0]
    runner=scored[1][0] if len(scored)>1 else 0.0
    margin=winner-runner
    contradiction = bool(runner >= winner*contradiction_ratio and runner > 0.45)

    # Required concepts are hard gates: a missing required concept cannot be
    # rescued by generic pattern similarity.
    missing=[]
    for req in requirements:
        s=pos.get(req.label,0.0)-neg.get(req.label,0.0)
        if req.kind in ("entity","person","animal","object","vehicle","face"):
            allowed={"entity","object","vehicle","face"}
            typed=max(
                (_c(e.score)*_source_factor(e.source) for e in ev
                 if e.label==req.label and e.polarity=="positive" and e.source in allowed),
                default=0.0
            )
            s=typed-neg.get(req.label,0.0)
        elif req.kind in ("brand",):
            typed=max(
                (_c(e.score)*_source_factor(e.source) for e in ev
                 if e.label==req.label and e.polarity=="positive" and e.source=="brand"),
                default=0.0
            )
            s=typed-neg.get(req.label,0.0)
        if req.required and s < req.threshold:
            missing.append(req.label)

    conf=_calibrate(winner,len(support_sources.get(wlabel, {"unknown"})),contradiction,margin)
    accepted=(conf>=accept_threshold and not contradiction and not missing and
              (margin>=min_margin or len(scored)==1))

    reason=f"{wlabel} güven={conf:.3f}; marj={margin:.3f}"
    if contradiction: reason+="; çelişki"
    if missing: reason+="; eksik="+",".join(missing)

    return Decision(wlabel,conf,accepted,contradiction,missing,ev,reason)


class UnifiedDecisionEngine:
    """Small compatibility wrapper for callers that prefer an engine object."""

    def __init__(self, *, accept_threshold: float = 0.72,
                 contradiction_ratio: float = 0.92, min_margin: float = 0.08):
        self.accept_threshold = accept_threshold
        self.contradiction_ratio = contradiction_ratio
        self.min_margin = min_margin

    def decide(self, evidence: Iterable[Evidence],
               requirements: Iterable[QueryRequirement] = ()) -> Decision:
        return decide(
            evidence, requirements,
            accept_threshold=self.accept_threshold,
            contradiction_ratio=self.contradiction_ratio,
            min_margin=self.min_margin,
        )
