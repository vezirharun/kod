"""N-concept visual composition engine.

Query-time ranking helper only. It does not read or write indexes, SQLite,
FAISS, thumbnails, previews, or embedding stores.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


class MultiConceptCompositionEngine:
    """Scores any N-concept query with one generic composition model."""

    weights = {
        "concept_coverage": 0.30,
        "same_composition": 0.25,
        "patch": 0.15,
        "semantic": 0.10,
        "visual_embedding": 0.10,
        "pattern_dna": 0.05,
        "texture": 0.05,
    }

    def evaluate(
        self,
        *,
        required: list[str] | tuple[str, ...],
        signals: list[float],
        supported: list[bool],
        soft_weights: list[float] | None = None,
        relationship_seed: float = 0.0,
        q_evidence: float = 0.0,
        patch: float = 0.0,
        semantic: float = 0.0,
        visual_embedding: float = 0.0,
        pattern_dna: float = 0.0,
        texture: float = 0.0,
        contradiction: float = 0.0,
    ) -> dict[str, Any]:
        concepts = _dedupe(required)
        n = len(concepts)
        signals = [_clamp(x) for x in list(signals)[:n]]
        supported = [bool(x) for x in list(supported)[:n]]
        soft_weights = [
            _clamp(x) for x in list(soft_weights or [0.0] * n)[:n]
        ]
        while len(signals) < n:
            signals.append(0.0)
        while len(supported) < n:
            supported.append(False)
        while len(soft_weights) < n:
            soft_weights.append(0.0)

        exact_count = sum(1 for ok in supported if ok)
        weighted_count = sum(1.0 if supported[i] else soft_weights[i] for i in range(n))
        concept_coverage = exact_count / float(n) if n else 0.0
        soft_coverage = weighted_count / float(n) if n else 0.0

        total_signal = sum(signals)
        if total_signal > 1e-9:
            relative = {concepts[i]: round(signals[i] / total_signal, 4) for i in range(n)}
        else:
            relative = {concepts[i]: 0.0 for i in range(n)}

        rel_seed = _clamp(relationship_seed)
        pairwise: dict[str, dict[str, Any]] = {}
        for i, j in combinations(range(n), 2):
            pair_key = f"{concepts[i]}::{concepts[j]}"
            pair_support = min(
                1.0 if supported[i] else soft_weights[i],
                1.0 if supported[j] else soft_weights[j],
            )
            pair_signal = min(signals[i], signals[j])
            pair_score = _clamp((0.50 * rel_seed) + (0.30 * pair_support) + (0.20 * pair_signal))
            pairwise[pair_key] = {
                "concepts": [concepts[i], concepts[j]],
                "score": round(pair_score, 4),
                "both_supported": bool(supported[i] and supported[j]),
                "relationship_seed": round(rel_seed, 4),
                "min_concept_signal": round(pair_signal, 4),
            }

        pair_scores = [float(v["score"]) for v in pairwise.values()]
        pair_avg = sum(pair_scores) / len(pair_scores) if pair_scores else rel_seed
        balance = min(signals) / max(signals) if signals and max(signals) > 1e-9 else 0.0
        group_coherence = _clamp((0.55 * pair_avg) + (0.30 * rel_seed) + (0.15 * balance))
        same_composition = rel_seed
        composition_confidence = _clamp(
            (0.45 * same_composition) + (0.35 * concept_coverage) + (0.20 * group_coherence)
        )

        patch = _clamp(patch)
        semantic = _clamp(semantic)
        visual_embedding = _clamp(visual_embedding)
        pattern_dna = _clamp(pattern_dna)
        texture = _clamp(texture)
        contradiction = _clamp(contradiction)

        weighted_score = (
            self.weights["concept_coverage"] * concept_coverage
            + self.weights["same_composition"] * same_composition
            + self.weights["patch"] * patch
            + self.weights["semantic"] * semantic
            + self.weights["visual_embedding"] * visual_embedding
            + self.weights["pattern_dna"] * pattern_dna
            + self.weights["texture"] * texture
        )
        if exact_count < n:
            weighted_score *= 0.82
        weighted_score -= 0.10 * contradiction
        if n >= 2 and exact_count == n and same_composition < 0.35:
            weighted_score = min(weighted_score, 0.60)
        weighted_score = _clamp(weighted_score)

        concept_evidence = {
            concepts[i]: {
                "signal": round(signals[i], 4),
                "supported": supported[i],
                "soft_weight": round(soft_weights[i], 4),
                "relative_contribution": relative[concepts[i]],
            }
            for i in range(n)
        }

        return {
            "concepts": concepts,
            "concept_count": n,
            "supported_count": exact_count,
            "concept_coverage": round(concept_coverage, 4),
            "soft_coverage": round(soft_coverage, 4),
            "concept_evidence": concept_evidence,
            "relative_contribution": relative,
            "pairwise_relationships": pairwise,
            "group_relationship": {
                "score": round(group_coherence, 4),
                "pairwise_average": round(pair_avg, 4),
                "balance": round(balance, 4),
                "soft_coverage": round(soft_coverage, 4),
            },
            "group_coherence": round(group_coherence, 4),
            "same_composition_score": round(same_composition, 4),
            "same_composition": round(same_composition, 4),
            "composition_confidence": round(composition_confidence, 4),
            "missing_concepts": [
                concepts[i] for i in range(n) if not supported[i] and soft_weights[i] <= 0
            ],
            "soft_supported": [
                concepts[i] for i in range(n) if not supported[i] and soft_weights[i] > 0
            ],
            "contradiction": round(contradiction, 4),
            "weights": dict(self.weights),
            "weighted_components": {
                "concept_coverage": round(self.weights["concept_coverage"] * concept_coverage, 4),
                "same_composition": round(self.weights["same_composition"] * same_composition, 4),
                "patch": round(self.weights["patch"] * patch, 4),
                "semantic": round(self.weights["semantic"] * semantic, 4),
                "visual_embedding": round(self.weights["visual_embedding"] * visual_embedding, 4),
                "pattern_dna": round(self.weights["pattern_dna"] * pattern_dna, 4),
                "texture": round(self.weights["texture"] * texture, 4),
            },
            "final_multi_concept_score": round(weighted_score, 4),
        }
