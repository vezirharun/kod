"""Consistency / outlier scoring from already-indexed fields.

Reuses taxonomy distinct-concept guards and teach_me label signals.
No new models; never writes classifications.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


_UNKNOWN = frozenset({"", "unknown", "none", "null", "belirsiz", "other", "diger", "diğer"})


@dataclass
class ConsistencyReport:
    score: float
    kind: str  # ok | suspicious | outlier
    reason: str
    guess: str = ""
    labels: list[str] = field(default_factory=list)
    conflict_pairs: list[tuple[str, str]] = field(default_factory=list)

    def to_store_fields(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "consistency": self.score,
            "guess": self.guess,
            "labels_json": json.dumps(self.labels, ensure_ascii=False),
        }


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _conf(tm: dict[str, Any], file_row: dict[str, Any]) -> float:
    for src in (
        tm.get("classification_confidence"),
        file_row.get("pattern_confidence"),
        tm.get("pattern_confidence"),
    ):
        try:
            c = float(src or 0)
        except (TypeError, ValueError):
            c = 0.0
        if c > 0:
            return max(0.0, min(1.0, c))
    return 0.0


def collect_label_candidates(file_row: dict[str, Any], tm: dict[str, Any]) -> list[str]:
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    sem = tm.get("semantic_tags") if isinstance(tm.get("semantic_tags"), dict) else {}
    raw = [
        file_row.get("pattern_family"),
        tm.get("pattern_family"),
        dna.get("family"),
        dna.get("main_family"),
        dna.get("motif_family"),
        dna.get("motif_class"),
        sem.get("family"),
        tm.get("animal_print_type"),
        file_row.get("category_path"),
        tm.get("category_path"),
    ]
    motifs = sem.get("motifs") if isinstance(sem.get("motifs"), list) else []
    raw.extend(motifs[:3])
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        # category_path → leaf only
        text = str(item or "").strip()
        if "/" in text:
            text = text.rsplit("/", 1)[-1]
        n = _norm(text)
        if not n or n in _UNKNOWN or n in seen:
            continue
        seen.add(n)
        out.append(text.strip())
    return out


def _distinct(a: str, b: str) -> bool:
    try:
        from core.canonical_correction import are_distinct_concepts

        return bool(are_distinct_concepts(a, b))
    except Exception:
        return False


def score_file_consistency(
    file_row: dict[str, Any],
    texture_map: dict[str, Any] | None,
) -> ConsistencyReport:
    tm = dict(texture_map or {})
    if bool(tm.get("user_labeled")) or str(tm.get("user_label_source") or "") in (
        "teach_me",
        "manual_user",
        "user",
    ):
        guess = str(tm.get("pattern_family") or file_row.get("pattern_family") or "")
        return ConsistencyReport(
            score=1.0,
            kind="ok",
            reason="Kullanıcı doğrulamış",
            guess=guess,
            labels=[guess] if guess else [],
        )

    labels = collect_label_candidates(file_row, tm)
    conf = _conf(tm, file_row)
    conflicts: list[tuple[str, str]] = []
    for a, b in combinations(labels, 2):
        if _distinct(a, b):
            conflicts.append((a, b))

    score = 1.0
    reasons: list[str] = []
    if conflicts:
        score -= 0.50
        pair = conflicts[0]
        reasons.append(f"Çelişen kavram: {pair[0]} ≠ {pair[1]}")
    elif len(labels) >= 2:
        score -= 0.28
        reasons.append("Motorlar farklı sınıf diyor")
    if not labels:
        score -= 0.40
        reasons.append("Anlamlı sınıf yok")
    if conf < 0.55:
        score -= 0.30
        reasons.append("Güven düşük")
    elif conf < 0.72 and labels:
        score -= 0.10
        reasons.append("Güven orta")

    score = max(0.0, min(1.0, score))
    guess = labels[0] if labels else ""
    if conflicts:
        kind = "outlier"
        reason = reasons[0] if reasons else "Kavram kimliği çelişkisi"
    elif score < 0.55 or not labels:
        kind = "suspicious"
        reason = "; ".join(reasons) if reasons else "Şüpheli sınıflandırma"
    else:
        kind = "ok"
        reason = "Tutarlı"
    return ConsistencyReport(
        score=score,
        kind=kind,
        reason=reason,
        guess=guess,
        labels=labels,
        conflict_pairs=conflicts,
    )
