"""Evrensel family group gate — her özel gruba yalnızca kanıtlı aday girer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.dynamic_groups import (
    G_COLOR,
    G_CROP,
    G_EXACT,
    G_FAR,
    G_FORMAT,
    G_RELATED,
    G_RESOLUTION,
    G_SAME_CLOSE,
    G_SAME_STYLE,
    G_UNRELATED,
    QueryContext,
    normalize_cluster_key,
)
from core.taxonomy import normalize_family

# Özel family grupları — sıkı kapı
SPECIAL_CLUSTER_KEYS = frozenset(
    {
        G_COLOR,
        G_SAME_CLOSE,
        G_SAME_STYLE,
        G_RELATED,
    }
)

# Exact / format — protected veya güçlü görsel eşleşme hariç
EXACT_TIER_KEYS = frozenset(
    {
        G_EXACT,
        G_FORMAT,
        G_RESOLUTION,
        G_CROP,
    }
)

MARBLE_FAMILIES = frozenset({"marble", "abstract", "marble_abstract", "texture_ground"})

ANIMAL_SUBTYPES = frozenset(
    {
        "leopard",
        "jaguar",
        "cheetah",
        "tiger",
        "zebra",
        "snake",
        "crocodile",
        "cow",
        "giraffe",
        "mixed_animal",
        "unknown_animal",
    }
)

FLORAL_SUBTYPES = frozenset(
    {
        "rose",
        "daisy",
        "leaf",
        "small_floral",
        "ditsy_floral",
        "mixed_floral",
        "big_flower",
        "tropical_flower",
    }
)

PLAID_SUBTYPES = frozenset({"plaid", "check", "houndstooth", "gingham"})

STRIPE_SUBTYPES = frozenset(
    {
        "vertical_stripe",
        "horizontal_stripe",
        "diagonal_stripe",
        "wavy_stripe",
    }
)

NON_PATTERN_FAMILIES = frozenset(
    {
        "document",
        "plain",
        "garment_photo",
        "icon_logo_non_textile",
        "unknown",
    }
)

# Query family → adayın girebileceği candidate family kümesi (özel gruplar için)
FAMILY_ALLOWED_CANDIDATES: dict[str, frozenset[str]] = {
    "animal_print": frozenset({"animal_print"}),
    "floral": frozenset({"floral"}),
    "marble_abstract": frozenset(
        {"marble_abstract", "marble", "abstract", "texture_ground"}
    ),
    "plaid_check": frozenset({"plaid_check"}),
    "stripe": frozenset({"stripe"}),
    "paisley": frozenset({"paisley", "scarf_border"}),
    "scarf_border": frozenset({"paisley", "scarf_border"}),
    "baroque": frozenset({"baroque", "chain"}),
    "chain": frozenset({"baroque", "chain"}),
    "monogram_logo": frozenset({"monogram_logo"}),
    "geometric": frozenset({"geometric"}),
    "lace": frozenset({"lace"}),
    "polka_dot": frozenset({"polka_dot", "geometric"}),
}

# Yakın aile (related_family) için genişletilmiş aday kümesi
RELATED_ALLOWED_CANDIDATES: dict[str, frozenset[str]] = {
    "animal_print": frozenset({"animal_print"}),
    "floral": frozenset({"floral"}),
    "marble_abstract": frozenset(
        {"marble_abstract", "marble", "abstract", "texture_ground"}
    ),
    "plaid_check": frozenset({"plaid_check", "stripe"}),
    "stripe": frozenset({"stripe", "plaid_check"}),
    "paisley": frozenset({"paisley", "scarf_border", "baroque"}),
    "scarf_border": frozenset({"paisley", "scarf_border", "baroque"}),
    "baroque": frozenset({"baroque", "chain", "paisley", "scarf_border"}),
    "chain": frozenset({"baroque", "chain"}),
    "monogram_logo": frozenset({"monogram_logo"}),
    "geometric": frozenset({"geometric", "polka_dot"}),
    "lace": frozenset({"lace", "geometric"}),
}

# Çapraz family — tamamen alakasız
HARD_BLOCK_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("animal_print", "floral"),
        ("floral", "animal_print"),
        ("animal_print", "marble_abstract"),
        ("marble_abstract", "animal_print"),
        ("animal_print", "plaid_check"),
        ("plaid_check", "animal_print"),
        ("animal_print", "monogram_logo"),
        ("monogram_logo", "animal_print"),
        ("floral", "marble_abstract"),
        ("marble_abstract", "floral"),
        ("floral", "plaid_check"),
        ("plaid_check", "floral"),
        ("floral", "monogram_logo"),
        ("monogram_logo", "floral"),
        ("marble_abstract", "plaid_check"),
        ("plaid_check", "marble_abstract"),
        ("marble_abstract", "monogram_logo"),
        ("monogram_logo", "marble_abstract"),
        ("animal_print", "baroque"),
        ("baroque", "animal_print"),
        ("floral", "baroque"),
        ("baroque", "floral"),
    }
)


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str = ""
    score_cap: float = 1.0
    fallback_cluster: str = G_FAR


@dataclass(frozen=True)
class CandidateMeta:
    pattern_family: str = "unknown"
    pattern_subtype: str = ""
    animal_print_type: str = ""
    classification_confidence: float = 0.0
    texture_map: dict[str, Any] | None = None

    @classmethod
    def from_profile(cls, prof) -> "CandidateMeta":
        from core.texture_profile import TextureProfile

        if isinstance(prof, CandidateMeta):
            return prof
        if isinstance(prof, TextureProfile):
            return cls(
                pattern_family=prof.pattern_family or "unknown",
                pattern_subtype=getattr(prof, "pattern_subtype", "")
                or prof.animal_print_type
                or "",
                animal_print_type=prof.animal_print_type or "",
                classification_confidence=float(
                    getattr(prof, "classification_confidence", 0) or 0
                ),
                texture_map=getattr(prof, "texture_map", None),
            )
        if isinstance(prof, QueryContext):
            return cls(
                pattern_family=prof.pattern_family or "unknown",
                pattern_subtype=prof.pattern_subtype or prof.animal_print_type or "",
                animal_print_type=prof.animal_print_type or "",
                classification_confidence=float(prof.classification_confidence or 0),
            )
        return cls()

    @property
    def normalized_family(self) -> str:
        fam = normalize_family(self.pattern_family or "unknown")
        if fam in MARBLE_FAMILIES:
            return "marble_abstract"
        return fam

    @property
    def effective_subtype(self) -> str:
        return (self.pattern_subtype or self.animal_print_type or "").strip().lower()


def _query_family(ctx) -> str:
    if isinstance(ctx, QueryContext):
        fam = ctx.effective_family
    else:
        from core.pattern_classifier import effective_query_family
        from core.texture_profile import TextureProfile

        prof = CandidateMeta.from_profile(ctx)
        fam = effective_query_family(
            TextureProfile(
                pattern_family=prof.pattern_family,
                animal_print_type=prof.animal_print_type,
                pattern_subtype=prof.pattern_subtype,
                classification_confidence=prof.classification_confidence,
            )
        )
    fam = normalize_family(fam or "unknown")
    if fam in MARBLE_FAMILIES:
        return "marble_abstract"
    return fam


def family_relationship(query_family: str, candidate_family: str) -> str:
    """same | related | unrelated"""
    qf = normalize_family(query_family or "unknown")
    cf = normalize_family(candidate_family or "unknown")
    if cf in MARBLE_FAMILIES:
        cf = "marble_abstract"
    if qf in MARBLE_FAMILIES:
        qf = "marble_abstract"

    # Belirsiz veya desen dışı sınıflar aile eşleşmesi değildir.
    # unknown == unknown durumunu SAME kabul etmek, family gate'in
    # belirsiz görselleri yanlışlıkla aynı aileye yükseltmesine neden olur.
    if qf in NON_PATTERN_FAMILIES or cf in NON_PATTERN_FAMILIES:
        return "unrelated"

    if qf == cf:
        return "same"
    if (qf, cf) in HARD_BLOCK_PAIRS:
        return "unrelated"
    rel = RELATED_ALLOWED_CANDIDATES.get(qf, frozenset())
    if cf in rel:
        return "related"
    return "unrelated"


def family_score_cap(
    query_family: str,
    candidate: CandidateMeta,
    *,
    protected_exact: bool = False,
    same_subtype: bool = False,
    phash_sim: float = 0.0,
    dhash_sim: float = 0.0,
) -> float | None:
    """None = cap yok (gerçek exact eşleşme)."""
    rel = family_relationship(query_family, candidate.normalized_family)
    if protected_exact:
        if phash_sim >= 0.95 and dhash_sim >= 0.90:
            return None
        if phash_sim >= 0.92 and dhash_sim >= 0.88 and rel == "same":
            return None
        if rel == "unrelated":
            return 0.68
        if rel == "related":
            return 0.78
        return 0.85
    if rel == "same":
        if same_subtype:
            return 1.0
        return 0.75
    if rel == "related":
        return 0.60
    # Unrelated family labels, but same animal_print_type (e.g. floral mislabel
    # on a leopard fabric) — do not hard-cap at 0.40; keep pattern identity.
    if same_subtype:
        return 0.88
    return 0.40


def _animal_evidence(candidate: CandidateMeta) -> bool:
    from core.pattern_classifier import is_confident_animal_print
    from core.texture_profile import TextureProfile

    if candidate.normalized_family != "animal_print":
        return False
    prof = TextureProfile(
        pattern_family=candidate.pattern_family,
        animal_print_type=candidate.animal_print_type,
        pattern_subtype=candidate.pattern_subtype,
        classification_confidence=candidate.classification_confidence,
    )
    if candidate.animal_print_type in ANIMAL_SUBTYPES:
        return candidate.classification_confidence >= 0.50 or is_confident_animal_print(
            prof
        )
    tm = candidate.texture_map or {}
    return float(tm.get("animal_score", 0)) >= 0.45 or is_confident_animal_print(prof)


def _floral_evidence(candidate: CandidateMeta) -> bool:
    if candidate.normalized_family != "floral":
        return False
    tm = candidate.texture_map or {}
    return (
        candidate.classification_confidence >= 0.50
        or float(tm.get("floral_score", 0)) >= 0.45
        or candidate.effective_subtype in FLORAL_SUBTYPES
    )


def _marble_evidence(candidate: CandidateMeta) -> bool:
    if (
        candidate.normalized_family not in MARBLE_FAMILIES
        and candidate.normalized_family != "marble_abstract"
    ):
        return False
    tm = candidate.texture_map or {}
    return (
        candidate.classification_confidence >= 0.48
        or float(tm.get("marble_score", 0)) >= 0.40
    )


def _plaid_evidence(candidate: CandidateMeta) -> bool:
    return candidate.normalized_family == "plaid_check" and (
        candidate.classification_confidence >= 0.48
        or candidate.effective_subtype in PLAID_SUBTYPES
    )


def _stripe_evidence(candidate: CandidateMeta) -> bool:
    return candidate.normalized_family == "stripe" and (
        candidate.classification_confidence >= 0.48
        or candidate.effective_subtype in STRIPE_SUBTYPES
    )


def _generic_family_evidence(candidate: CandidateMeta, expected: str) -> bool:
    return (
        candidate.normalized_family == expected
        and candidate.classification_confidence >= 0.45
    )


FAMILY_EVIDENCE_CHECKS = {
    "animal_print": _animal_evidence,
    "floral": _floral_evidence,
    "marble_abstract": _marble_evidence,
    "plaid_check": _plaid_evidence,
    "stripe": _stripe_evidence,
    "paisley": lambda c: _generic_family_evidence(c, "paisley"),
    "scarf_border": lambda c: _generic_family_evidence(c, "scarf_border"),
    "baroque": lambda c: _generic_family_evidence(c, "baroque"),
    "chain": lambda c: _generic_family_evidence(c, "chain"),
    "monogram_logo": lambda c: _generic_family_evidence(c, "monogram_logo"),
    "geometric": lambda c: _generic_family_evidence(c, "geometric"),
    "lace": lambda c: _generic_family_evidence(c, "lace"),
}


def _candidate_has_evidence(candidate: CandidateMeta, family: str) -> bool:
    if candidate.normalized_family in NON_PATTERN_FAMILIES:
        return False
    check = FAMILY_EVIDENCE_CHECKS.get(family)
    if check:
        return check(candidate)
    allowed = FAMILY_ALLOWED_CANDIDATES.get(family, frozenset())
    return candidate.normalized_family in allowed


def _allowed_candidate_families(query_family: str, cluster_key: str) -> frozenset[str]:
    qf = normalize_family(query_family)
    if qf in MARBLE_FAMILIES:
        qf = "marble_abstract"
    if cluster_key == G_RELATED:
        return RELATED_ALLOWED_CANDIDATES.get(
            qf, FAMILY_ALLOWED_CANDIDATES.get(qf, frozenset({qf}))
        )
    return FAMILY_ALLOWED_CANDIDATES.get(qf, frozenset({qf}))


def can_enter_group(
    query_context,
    candidate_meta,
    group_name: str,
    score_breakdown: dict[str, float] | None = None,
    *,
    protected_exact: bool = False,
) -> GateDecision:
    """Özel family grubuna aday girebilir mi?"""
    score_breakdown = score_breakdown or {}
    cluster = normalize_cluster_key(group_name or G_FAR)

    if protected_exact:
        return GateDecision(allowed=True, reason="Korumalı exact eşleşme")

    if cluster in EXACT_TIER_KEYS:
        phash = float(score_breakdown.get("phash", 0))
        patch = float(score_breakdown.get("patch", 0))
        if phash >= 0.92 or patch >= 0.82:
            return GateDecision(
                allowed=True, reason="Güçlü görsel eşleşme — format/crop"
            )
        return GateDecision(
            allowed=False,
            reason="Exact/format için yetersiz görsel kanıt",
            score_cap=0.55,
            fallback_cluster=G_FAR,
        )

    if cluster not in SPECIAL_CLUSTER_KEYS:
        return GateDecision(allowed=True)

    qf = _query_family(query_context)
    cand = CandidateMeta.from_profile(candidate_meta)
    cf = cand.normalized_family

    if qf in ("unknown", "texture_ground", "plain", ""):
        return GateDecision(allowed=True, reason="Belirsiz sorgu — gevşek kapı")

    allowed_fams = _allowed_candidate_families(qf, cluster)
    rel = family_relationship(qf, cf)

    if cf in NON_PATTERN_FAMILIES:
        return GateDecision(
            allowed=False,
            reason=f"Alakasız — {cf} (desen değil)",
            score_cap=0.35,
            fallback_cluster=G_UNRELATED,
        )

    if rel == "unrelated" or cf not in allowed_fams:
        return GateDecision(
            allowed=False,
            reason=f"Family uyuşmuyor — {qf} sorgu / {cf} aday",
            score_cap=0.40,
            fallback_cluster=G_UNRELATED,
        )

    if not _candidate_has_evidence(cand, qf if cluster != G_RELATED else cf):
        return GateDecision(
            allowed=False,
            reason=f"Yetersiz {cf} kanıtı — özel gruba alınamaz",
            score_cap=0.45,
            fallback_cluster=G_FAR,
        )

    # related_family: related ama aynı family değil
    if cluster == G_RELATED and rel != "related" and rel != "same":
        return GateDecision(
            allowed=False,
            reason="Yakın aile değil",
            score_cap=0.50,
            fallback_cluster=G_FAR,
        )

    cap = family_score_cap(qf, cand, protected_exact=False)
    return GateDecision(
        allowed=True,
        reason=f"Family kapısı geçti — {cf}",
        score_cap=cap if cap is not None else 1.0,
    )


def enforce_cluster_gate(
    query_context,
    candidate_meta,
    cluster_group: str,
    cluster_reason: str,
    score: float,
    *,
    protected_exact: bool = False,
    phash_sim: float = 0.0,
    dhash_sim: float = 0.0,
    patch_sim: float = 0.0,
    query_animal_type: str = "",
    candidate_animal_type: str = "",
) -> tuple[str, str, float, str]:
    """
    Özel grupları doğrula, skor tavanı uygula.
    Dönüş: cluster_group, cluster_reason, score, gate_note
    """
    norm = normalize_cluster_key(cluster_group)
    breakdown = {"phash": phash_sim, "dhash": dhash_sim, "patch": patch_sim}

    qf = _query_family(query_context)
    cand = CandidateMeta.from_profile(candidate_meta)
    same_subtype = bool(
        query_animal_type
        and candidate_animal_type
        and query_animal_type == candidate_animal_type
    ) or (
        qf == cand.normalized_family
        and cand.effective_subtype
        and cand.effective_subtype
        == CandidateMeta.from_profile(query_context).effective_subtype
    )

    cap = family_score_cap(
        qf,
        cand,
        protected_exact=protected_exact,
        same_subtype=same_subtype,
        phash_sim=phash_sim,
        dhash_sim=dhash_sim,
    )
    if cap is not None:
        score = min(score, cap)

    if protected_exact:
        rel = family_relationship(qf, cand.normalized_family)
        out_group = cluster_group
        if rel == "unrelated" and phash_sim < 0.92:
            out_group = G_FAR
            cluster_reason = (
                cluster_reason
                or f"Yapısal benzerlik — farklı aile ({qf} / {cand.normalized_family})"
            )
        return out_group, cluster_reason, score, "protected_exact"

    decision = can_enter_group(
        query_context,
        candidate_meta,
        norm,
        breakdown,
        protected_exact=False,
    )

    if not decision.allowed and norm in SPECIAL_CLUSTER_KEYS:
        fallback = decision.fallback_cluster
        reason = decision.reason
        score = min(score, decision.score_cap)
        return fallback, reason, score, f"gate_blocked:{reason}"

    if cap is not None and score > cap:
        score = cap

    gate_note = decision.reason if decision.allowed else decision.reason
    return cluster_group, cluster_reason, score, gate_note


def candidate_family_display_label(candidate_meta) -> str:
    """Kartta gösterilecek adayın gerçek family etiketi."""
    cand = CandidateMeta.from_profile(candidate_meta)
    fam = cand.normalized_family
    sub = cand.effective_subtype
    if fam == "animal_print" and cand.animal_print_type:
        return f"animal_print / {cand.animal_print_type}"
    if sub and sub != fam:
        return f"{fam} / {sub}"
    return fam or "unknown"
