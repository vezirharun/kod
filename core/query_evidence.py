"""Query Evidence — parsed visual slots become ranking evidence.

Does not rebuild indexes, FAISS, CLIP/DINO, or Pattern DNA.
Does not replace base_score. Additive layer only:

    final = clamp(base + query_evidence + composite_evidence - contradiction_penalty)

Weights come from measured score gaps (filename ties at ~0.76, Exact/OCR ~0.90,
nl_color unused because of max()). Deltas are smaller than Exact so existing
true positives stay on top.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from core.natural_language_query import COLOR_PHRASES, MOTIF_PHRASES, ParsedQuery, parse_natural_query
from core.textile_terms import expand_query_terms, normalize_turkish, tokenize

STATE_SUPPORTED = "SUPPORTED"
STATE_CONTRADICTED = "CONTRADICTED"
STATE_UNKNOWN = "UNKNOWN"
EvidenceState = Literal["SUPPORTED", "CONTRADICTED", "UNKNOWN"]

CHANNELS = (
    "filename",
    "ocr",
    "semantic",
    "visual",
    "dna",
    "texture",
    "family",
    "family_bridge",
)

# Measured: equal rose filename 0.76; +0.08 lifts red above unknown; Exact 0.90 stays above.
W_COLOR_SUPPORT = 0.08
W_COLOR_CONTRA = 0.10
W_COMPOSITE_ALL = 0.10
W_COMPOSITE_RELATED = 0.06
W_SECONDARY_ONLY = 0.04
W_TEXT_SUPPORT = 0.06
W_CONFLICT = 0.05
W_VISUAL_SUPPORT = 0.08
VISUAL_CHANNEL_MIN = 0.20
CLIP_OWN_MIN = 0.22
SUBTYPE_MARGIN = 0.025
VISUAL_EXACT_MIN = 0.30
VISUAL_EXACT_MARGIN = 0.05

GRADE_EXACT = "visual_exact"
GRADE_STRONG = "visual_strong"
GRADE_FAMILY = "visual_family"
GRADE_UNKNOWN = "visual_unknown"
GRADE_CONFLICT = "visual_conflict"

# Query concept_id → CLIP score keys already stored by rivals / UVI / v2.
_CLIP_KEYS = {
    "araba": ("car", "araba", "otomobil"),
    "otomobil": ("car", "araba"),
    "car": ("car", "araba"),
    "kus": ("bird", "kus"),
    "bird": ("bird", "kus"),
    "togg": ("togg",),
    "suv": ("suv",),
}

_VERDICT = {
    GRADE_EXACT: "CLIP görseli gördü (exact) — dosya adı tek başına değil",
    GRADE_STRONG: "CLIP görseli gördü (strong)",
    GRADE_FAMILY: "CLIP aile benzerliği — kesin nesne kanıtı yok",
    GRADE_CONFLICT: "Görsel çelişki: CLIP ile dosya adı/metadata aynı şeyi söylemiyor",
    GRADE_UNKNOWN: "Görsel kanıt yok — yalnızca dosya adı/OCR/metadata",
}

_STOP = frozenset(
    {
        "ve",
        "ile",
        "bir",
        "the",
        "a",
        "an",
        "desen",
        "pattern",
        "print",
        "kumas",
        "kumaş",
        "fabric",
        "uzerine",
        "üstüne",
        "bakan",
        "iki",
        "suv",
    }
)

_RELATED = {
    "rose": "floral",
    "daisy": "floral",
    "tulip": "floral",
    "orchid": "floral",
    "peony": "floral",
    "floral": "floral",
    "flower": "floral",
}

_ANIMAL = frozenset({"leopard", "snake", "zebra", "tiger", "cheetah", "giraffe", "cow"})


@dataclass
class ConceptSpec:
    concept_id: str
    kind: str  # object | motif | color | text
    aliases: list[str] = field(default_factory=list)
    required: bool = True


@dataclass
class SearchEvidencePlan:
    raw: str = ""
    objects: list[ConceptSpec] = field(default_factory=list)
    attributes: list[ConceptSpec] = field(default_factory=list)
    and_groups: list[list[str]] = field(default_factory=list)
    residual: list[str] = field(default_factory=list)

    def all_concepts(self) -> list[ConceptSpec]:
        return list(self.objects) + list(self.attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "objects": [asdict(c) for c in self.objects],
            "attributes": [asdict(c) for c in self.attributes],
            "and_groups": [list(g) for g in self.and_groups],
            "residual": list(self.residual),
        }


@dataclass
class ChannelHit:
    source: str
    score: float = 0.0


@dataclass
class ConceptEvidence:
    concept_id: str
    kind: str
    state: str = STATE_UNKNOWN
    score: float = 0.0
    channels: dict[str, float] = field(default_factory=dict)
    conflict_with: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEvidenceReport:
    concepts: dict[str, ConceptEvidence] = field(default_factory=dict)
    conflict: bool = False
    composite: str = ""
    delta: float = 0.0
    query_evidence: float = 0.0
    composite_evidence: float = 0.0
    contradiction_penalty: float = 0.0
    confidence: str = "LOW"
    visual_object: float = 0.0
    visual_attribute: float = 0.0
    visual_similarity: float = 0.0
    visual_family: float = 0.0
    visual_conflict: bool = False
    visual_grade: str = GRADE_UNKNOWN
    visual_verdict: str = ""
    visual_rival: str = ""
    visual_composite: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "conflict": self.conflict,
            "composite": self.composite,
            "delta": round(self.delta, 4),
            "query_evidence": round(self.query_evidence, 4),
            "composite_evidence": round(self.composite_evidence, 4),
            "contradiction_penalty": round(self.contradiction_penalty, 4),
            "confidence": self.confidence,
            "visual_object": round(self.visual_object, 4),
            "visual_attribute": round(self.visual_attribute, 4),
            "visual_similarity": round(self.visual_similarity, 4),
            "visual_family": round(self.visual_family, 4),
            "visual_conflict": self.visual_conflict,
            "visual_grade": self.visual_grade,
            "visual_verdict": self.visual_verdict,
            "visual_rival": self.visual_rival,
            "visual_composite": round(self.visual_composite, 4),
        }


def _alias_map() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for phrase, canon in MOTIF_PHRASES.items():
        nid = normalize_turkish(canon)
        out.setdefault(nid, set()).update(
            {normalize_turkish(phrase), nid, normalize_turkish(canon)}
        )
    for phrase, canon in COLOR_PHRASES.items():
        nid = normalize_turkish(canon)
        out.setdefault(nid, set()).update(
            {normalize_turkish(phrase), nid, normalize_turkish(canon)}
        )
    try:
        from core.textile_terms import TERM_SYNONYMS
    except Exception:
        TERM_SYNONYMS = {}
    for key, vals in (TERM_SYNONYMS or {}).items():
        kn = normalize_turkish(key)
        if kn not in out and normalize_turkish(str(vals[0] if vals else "")) not in out:
            continue
        bucket = out.setdefault(kn, {kn})
        for v in vals:
            vn = normalize_turkish(v)
            if not vn or vn in ("animal print", "animal_print", "desen", "pattern", "print"):
                continue
            bucket.add(vn)
            if vn in out:
                out[vn].add(kn)
    return out


_ALIASES = _alias_map()


def concept_aliases(concept_id: str) -> list[str]:
    cid = normalize_turkish(concept_id or "")
    if not cid:
        return []
    raw = _ALIASES.get(cid, {cid}) | {cid}
    extra = expand_query_terms(cid, include_semantic=False)[:8]
    merged = []
    for x in list(raw) + extra + list(_CLIP_KEYS.get(cid, ())):
        n = normalize_turkish(str(x))
        if n and n not in merged and len(n) >= 2:
            merged.append(n)
    return merged[:16]


def _spec(concept_id: str, kind: str, *, required: bool = True) -> ConceptSpec:
    cid = normalize_turkish(concept_id)
    return ConceptSpec(
        concept_id=cid,
        kind=kind,
        aliases=concept_aliases(cid),
        required=required,
    )


def build_search_plan(
    query: str,
    *,
    parsed: ParsedQuery | None = None,
    v2q: Any | None = None,
    hints: dict[str, str] | None = None,
) -> SearchEvidencePlan:
    raw = (query or "").strip()
    parsed = parsed or parse_natural_query(raw)
    plan = SearchEvidencePlan(raw=raw, residual=tokenize(parsed.residual or ""))
    seen: set[str] = set()

    def _add(spec: ConceptSpec, bucket: list[ConceptSpec]) -> None:
        if not spec.concept_id or spec.concept_id in seen:
            return
        seen.add(spec.concept_id)
        bucket.append(spec)

    if v2q is not None:
        for oid in list(getattr(v2q, "objects", None) or []):
            _add(_spec(str(oid), "object"), plan.objects)
        for mid in list(getattr(v2q, "motifs", None) or []):
            kind = "object" if str(mid) in _ANIMAL else "motif"
            _add(_spec(str(mid), kind), plan.objects)
    if parsed.motif:
        # Bir sorguda birden fazla motif/nesne olabilir: "gül leopard"
        # yalnızca ilk motifi (gül) kaybetmemeli; tüm motif tokenlarını
        # canonical kavramlara çevirip AND grubuna dahil et.
        from core.natural_language_query import MOTIF_PHRASES
        motif_values = [parsed.motif] + list(getattr(parsed, "motif_tokens", []) or [])
        seen_motifs: set[str] = set()
        for raw_motif in motif_values:
            raw_n = normalize_turkish(str(raw_motif))
            canonical = normalize_turkish(MOTIF_PHRASES.get(raw_n, raw_n))
            if not canonical or canonical in seen_motifs:
                continue
            seen_motifs.add(canonical)
            kind = "object" if canonical in _ANIMAL else "motif"
            _add(_spec(canonical, kind), plan.objects)
    if hints:
        apt = str(hints.get("animal_print_type") or hints.get("pattern_type") or "")
        if apt:
            _add(_spec(apt, "object"), plan.objects)
        pf = str(hints.get("pattern_family") or "")
        if pf and pf not in ("animal_print", "unknown") and not plan.objects:
            _add(_spec(pf, "motif"), plan.objects)

    if parsed.color:
        _add(_spec(parsed.color, "color"), plan.attributes)
    if parsed.brand:
        _add(_spec(parsed.brand, "text"), plan.attributes)

    for tok in plan.residual:
        if tok in _STOP or tok in seen or len(tok) < 3:
            continue
        _add(_spec(tok, "object", required=True), plan.objects)

    raw_toks = tokenize(raw)
    if "suv" in raw_toks and "suv" not in seen:
        _add(_spec("suv", "object", required=not plan.objects), plan.objects)

    if plan.objects and not any(c.required for c in plan.objects):
        for spec in plan.objects:
            spec.required = True

    obj_ids = [c.concept_id for c in plan.objects if c.required]
    if len(obj_ids) >= 2:
        plan.and_groups.append(obj_ids)
    color_ids = [c.concept_id for c in plan.attributes if c.kind == "color"]
    if obj_ids and color_ids:
        plan.and_groups.append([obj_ids[0], color_ids[0]])
    text_ids = [c.concept_id for c in plan.attributes if c.kind == "text"]
    if obj_ids and text_ids:
        plan.and_groups.append([obj_ids[0], text_ids[0]])
    return plan


def plan_retrieval_needles(plan: SearchEvidencePlan, *, limit: int = 12) -> list[str]:
    """Object/motif aliases for candidate retrieval. Color-only terms stay ranking-side."""
    out: list[str] = []
    for spec in plan.objects:
        for a in spec.aliases:
            if a and a not in out and a not in _STOP:
                out.append(a)
            if len(out) >= limit:
                return out
    for spec in plan.attributes:
        if spec.kind != "text":
            continue
        for a in spec.aliases[:4]:
            if a and a not in out:
                out.append(a)
            if len(out) >= limit:
                return out
    return out


def _haystacks(rec: dict[str, Any], texture_map: dict[str, Any]) -> dict[str, str]:
    fname = normalize_turkish(str(rec.get("filename") or "")).replace("-", " ")
    path = normalize_turkish(str(rec.get("path") or "")).replace("-", " ")
    ocr = normalize_turkish(str(rec.get("ocr_text") or "")).replace("-", " ")
    # Adayın tek bir "pattern_family" alanına güvenme. Örneğin aynı desen
    # hem animal_print hem floral olabilir. Önceki kod rec.pattern_family
    # doluysa texture_map içindeki ikinci aileyi tamamen kaybediyordu.
    pf_values = [
        str(rec.get("pattern_family") or ""),
        str(texture_map.get("pattern_family") or ""),
        str(texture_map.get("pattern_subfamily") or ""),
        str(rec.get("pattern_subfamily") or ""),
    ]
    pf = normalize_turkish(" ".join(v for v in pf_values if v))
    pt = normalize_turkish(
        " ".join(
            str(v)
            for v in (
                rec.get("pattern_type"),
                rec.get("pattern_subtype"),
                texture_map.get("animal_print_type"),
            )
            if v
        )
    )
    animal = normalize_turkish(str(texture_map.get("animal_print_type") or ""))
    sem = texture_map.get("semantic_tags") or {}
    if not isinstance(sem, dict):
        sem = {}
    sem_bits = []
    for v in sem.values():
        if isinstance(v, list):
            sem_bits.extend(str(x) for x in v)
        elif v not in (None, False, True):
            sem_bits.append(str(v))
    # Semantic tags de family kanıtıdır; ayrı haystack olarak korunur.
    semantic = normalize_turkish(" ".join(sem_bits))
    dna = texture_map.get("pattern_dna") if isinstance(texture_map.get("pattern_dna"), dict) else {}
    dna_blob = normalize_turkish(
        " ".join(str(dna.get(k) or "") for k in ("motif_class", "motif", "texture_class"))
    )
    return {
        "filename": f"{fname} {path}",
        "ocr": ocr,
        "semantic": semantic,
        "family": pf,
        "texture": f"{pt} {animal}",
        "dna": dna_blob,
        "color": normalize_turkish(
            " ".join(
                str(v)
                for v in (
                    dna.get("color_family"),
                    texture_map.get("color_family"),
                    rec.get("color_family"),
                )
                if v
            )
        ),
        "animal": animal,
        "pattern_family": pf,
    }


def _alias_hit(aliases: list[str], blob: str) -> float:
    if not blob:
        return 0.0
    import re

    best = 0.0
    for a in aliases:
        if not a or len(a) < 3:
            continue
        if a in {"print", "desen", "pattern", "animal"}:
            continue
        if len(a) <= 4:
            if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", blob):
                best = max(best, 0.82)
            continue
        if a in blob:
            best = max(best, 0.90 if len(a) >= 5 else 0.84)
    return best


def _state_from_channels(
    spec: ConceptSpec,
    channels: dict[str, float],
    hay: dict[str, str],
) -> tuple[str, float, str]:
    vis = float(channels.get("visual") or 0.0)
    lexical = max(
        (v for k, v in channels.items() if k != "visual"),
        default=0.0,
    )
    vis_ok = vis >= VISUAL_CHANNEL_MIN
    support = max(vis if vis_ok else 0.0, lexical)
    conflict = ""
    if spec.kind == "color":
        cand = hay.get("color") or ""
        if vis_ok or lexical >= 0.5:
            if cand and cand != spec.concept_id:
                return STATE_SUPPORTED, support, cand
            return STATE_SUPPORTED, support, ""
        if cand and cand != spec.concept_id:
            return STATE_CONTRADICTED, 0.0, cand
        return STATE_UNKNOWN, 0.0, ""
    if spec.kind == "text":
        if support >= 0.5 or vis_ok:
            return STATE_SUPPORTED, support, ""
        return STATE_UNKNOWN, 0.0, ""

    animal = hay.get("animal") or ""
    family = hay.get("pattern_family") or ""
    if spec.concept_id in _ANIMAL and animal and animal != spec.concept_id:
        if vis_ok or lexical >= 0.5:
            return STATE_SUPPORTED, support, animal
        return STATE_CONTRADICTED, 0.0, animal
    related = _RELATED.get(spec.concept_id, "")
    if spec.concept_id in ("rose", "daisy", "tulip") and family:
        family_tokens = set(family.replace("/", " ").replace(",", " ").split())
        if related in family_tokens:
            # Çoklu aile (ör. animal_print + floral) çelişki değildir.
            pass
        elif family not in ("unknown", "") and (vis_ok or lexical >= 0.5):
            return STATE_SUPPORTED, support, family
        elif family not in ("unknown", "") and family_tokens:
            return STATE_CONTRADICTED, 0.0, family
    if vis_ok or lexical >= 0.5:
        return STATE_SUPPORTED, support, conflict
    if related and related in family.replace("/", " ").replace(",", " ").split() and support < 0.5:
        return STATE_UNKNOWN, 0.35, ""
    return STATE_UNKNOWN, 0.0, ""


def _clip_lookup(spec: ConceptSpec, clip_scores: dict[str, float]) -> float:
    keys = [spec.concept_id, *spec.aliases, *_CLIP_KEYS.get(spec.concept_id, ())]
    best = 0.0
    for k in keys:
        if not k:
            continue
        best = max(best, float(clip_scores.get(k) or 0.0))
        kn = normalize_turkish(str(k))
        if kn != k:
            best = max(best, float(clip_scores.get(kn) or 0.0))
    return best


def _visual_grade(
    plan: SearchEvidencePlan,
    report: CandidateEvidenceReport,
    clip_scores: dict[str, float],
) -> tuple[str, str]:
    own_ids = [c.concept_id for c in plan.objects if c.required] or [
        c.concept_id for c in plan.objects
    ]
    own_vis = float(report.visual_object or 0.0)
    rival_name = ""
    best_rival = 0.0
    for name, sc in clip_scores.items():
        kn = normalize_turkish(str(name))
        if kn in ("_similarity", "_family") or kn in own_ids:
            continue
        if kn not in _ANIMAL and kn not in ("floral", "flower", "denim", "car", "bird"):
            continue
        if kn in {a for cid in own_ids for a in _CLIP_KEYS.get(cid, ())}:
            continue
        scf = float(sc or 0.0)
        if scf > best_rival:
            best_rival = scf
            rival_name = kn
    if best_rival >= VISUAL_CHANNEL_MIN and own_vis + SUBTYPE_MARGIN < best_rival:
        return GRADE_CONFLICT, rival_name
    if own_vis >= VISUAL_EXACT_MIN and own_vis > best_rival + VISUAL_EXACT_MARGIN:
        return GRADE_EXACT, ""
    if own_vis >= CLIP_OWN_MIN and own_vis > best_rival + SUBTYPE_MARGIN:
        return GRADE_STRONG, ""
    if own_vis >= VISUAL_CHANNEL_MIN:
        return GRADE_STRONG, ""
    if float(report.visual_family or 0.0) >= VISUAL_CHANNEL_MIN:
        return GRADE_FAMILY, rival_name
    return GRADE_UNKNOWN, ""


def visual_verdict_text(report: CandidateEvidenceReport | dict[str, Any]) -> str:
    if isinstance(report, dict):
        grade = str(report.get("visual_grade") or GRADE_UNKNOWN)
        stored = str(report.get("visual_verdict") or "")
        if stored:
            return stored
    else:
        grade = str(report.visual_grade or GRADE_UNKNOWN)
        if report.visual_verdict:
            return report.visual_verdict
    return _VERDICT.get(grade, _VERDICT[GRADE_UNKNOWN])


def collect_candidate_evidence(
    plan: SearchEvidencePlan,
    rec: dict[str, Any],
    *,
    texture_map: dict[str, Any] | None = None,
    breakdown: dict[str, float] | None = None,
    clip_scores: dict[str, float] | None = None,
) -> CandidateEvidenceReport:
    tm = texture_map or {}
    if isinstance(tm, str):
        tm = {}
    hay = _haystacks(rec, tm)
    bd = breakdown or {}
    clip_scores = clip_scores or {}
    report = CandidateEvidenceReport()
    composite_visual = float(clip_scores.get("_composite") or 0.0)
    report.visual_composite = composite_visual

    for spec in plan.all_concepts():
        ch: dict[str, float] = {}
        fn = _alias_hit(spec.aliases, hay["filename"])
        if fn:
            ch["filename"] = max(fn, float(bd.get("filename_score") or 0) if spec.kind != "color" else fn)
        ocr = _alias_hit(spec.aliases, hay["ocr"])
        if ocr:
            ch["ocr"] = max(ocr, float(bd.get("ocr_score") or 0))
        sem = _alias_hit(spec.aliases, hay["semantic"])
        if sem:
            ch["semantic"] = max(sem, float(bd.get("semantic_score") or 0))
        fam = _alias_hit(spec.aliases, hay["family"]) or _alias_hit(spec.aliases, hay["texture"])
        if fam or (
            spec.concept_id in _ANIMAL and hay.get("animal") == spec.concept_id
        ):
            ch["family"] = 0.86 if hay.get("animal") == spec.concept_id else 0.70
        tex = _alias_hit(spec.aliases, hay["texture"])
        if tex:
            ch["texture"] = 0.86
        dna = _alias_hit(spec.aliases, hay["dna"])
        if dna:
            ch["dna"] = 0.80

        # Genel motif ailesi köprüsü:
        # rose/daisy/tulip gibi spesifik çiçekler, AI'nın "floral/flower"
        # ailesini bulduğu gerçek bir görselde yardımcı kanıt alır. Bu
        # "kesin gül" iddiası değildir; sadece composite sıralamada ikinci
        # kavramın aile düzeyinde desteklenmesini sağlar.
        family_bridge = 0.0
        related_family = _RELATED.get(spec.concept_id, "")
        if related_family:
            family_blob = f"{hay.get('family', '')} {hay.get('semantic', '')} {hay.get('dna', '')}"
            bridge_aliases = {
                "floral": ("floral", "flower", "çiçek", "cicek", "blossom"),
            }.get(related_family, (related_family,))
            if _alias_hit(list(bridge_aliases), family_blob):
                family_bridge = 0.62

        vis = _clip_lookup(spec, clip_scores)
        if (
            vis < VISUAL_CHANNEL_MIN
            and spec.kind in ("object", "motif")
            and spec.required
            and sum(1 for s in plan.objects if s.required) == 1
        ):
            vis = max(vis, float(clip_scores.get("_similarity") or 0.0))
        if vis >= VISUAL_CHANNEL_MIN:
            ch["visual"] = vis
        if spec.kind == "color":
            if hay.get("color") == spec.concept_id:
                ch["texture"] = max(ch.get("texture", 0.0), 0.88)
            cs = float(bd.get("nl_color") or 0.0)
            if cs >= 0.5:
                ch["texture"] = max(ch.get("texture", 0.0), cs)
        # family_bridge, kavramın kesin SUPPORT durumuna çevrilmemeli.
        # Önce bağımsız kanallardan state hesaplanır, sonra bridge yalnızca
        # composite ranker'a yardımcı kanıt olarak eklenir.
        state, score, conflict_with = _state_from_channels(spec, ch, hay)
        if family_bridge and state == STATE_UNKNOWN:
            ch["family_bridge"] = family_bridge
        # Composite CLIP yalnızca "aynı görselde birlikte bulunma" kanıtıdır.
        # Tek başına bir kavramın varlığını kanıtlamaz: örneğin "gül leopard"
        # sorgusunda sadece leopard olan bir dosyanın composite prompt skoru
        # yüksekse gül kavramını da otomatik olarak SUPPORTED yapmamalıdır.
        # Önce her kavramın bağımsız kanıtı gerekir; composite skor daha sonra
        # AND grubunun ilişki puanını yükseltir.
        if (
            spec.required
            and spec.kind in ("object", "motif")
            and len([c for c in plan.objects if c.required]) >= 2
            and composite_visual >= VISUAL_CHANNEL_MIN
        ):
            ch["visual_composite"] = composite_visual
        ev = ConceptEvidence(
            concept_id=spec.concept_id,
            kind=spec.kind,
            state=state,
            score=round(score, 4),
            channels={k: round(v, 4) for k, v in ch.items()},
            conflict_with=conflict_with,
        )
        if conflict_with and state == STATE_SUPPORTED:
            report.conflict = True
        if state == STATE_CONTRADICTED:
            report.conflict = True
        report.concepts[spec.concept_id] = ev
    vis_obj = [
        ev.channels.get("visual", 0.0)
        for spec in plan.objects
        for ev in [report.concepts.get(spec.concept_id)]
        if ev
    ]
    vis_attr = [
        ev.channels.get("visual", 0.0)
        for spec in plan.attributes
        for ev in [report.concepts.get(spec.concept_id)]
        if ev
    ]
    report.visual_object = max(vis_obj, default=0.0)
    report.visual_attribute = max(vis_attr, default=0.0)
    report.visual_similarity = max(
        float(clip_scores.get("_similarity") or 0.0),
        report.visual_object,
        report.visual_attribute,
        report.visual_composite,
    )
    animal_vis = [
        float(clip_scores.get(a) or 0.0)
        for a in _ANIMAL
        if float(clip_scores.get(a) or 0.0) >= VISUAL_CHANNEL_MIN
    ]
    report.visual_family = max(
        float(clip_scores.get("_family") or 0.0),
        max(animal_vis, default=0.0),
    )
    grade, rival = _visual_grade(plan, report, clip_scores)
    report.visual_grade = grade
    report.visual_rival = rival
    report.visual_conflict = grade == GRADE_CONFLICT
    report.visual_verdict = visual_verdict_text(report)
    return report


def _group_delta(
    plan: SearchEvidencePlan, report: CandidateEvidenceReport, group: list[str]
) -> tuple[float, str]:
    states = [report.concepts.get(cid) for cid in group]
    supported = [s for s in states if s and s.state == STATE_SUPPORTED]
    if len(supported) == len(group) and group:
        # Aynı kayıtta tüm zorunlu kavramlar bağımsız olarak destekleniyorsa
        # AND geçer. Composite CLIP mevcutsa bu, kavramların aynı görsel
        # kompozisyonda birlikte bulunduğuna ek güçlü kanıt olarak kullanılır.
        if report.visual_composite >= VISUAL_CHANNEL_MIN:
            return W_COMPOSITE_ALL + min(0.12, report.visual_composite * 0.08), "PASS"
        return W_COMPOSITE_ALL, "PASS"
    if len(group) >= 2 and len(supported) == 1:
        sid = supported[0].concept_id
        other = next((c for c in group if c != sid), "")
        if _RELATED.get(sid) == other or _RELATED.get(other) == sid:
            return W_COMPOSITE_RELATED, "PARTIAL_RELATED"
        primary = group[0]
        if sid == primary:
            return 0.0, "PARTIAL_PRIMARY"
        return -W_SECONDARY_ONLY, "PARTIAL_SECONDARY"
    if not supported:
        return 0.0, "NONE"
    return round(W_COMPOSITE_ALL * (len(supported) / max(len(group), 1)), 4), "PARTIAL"


def _composite_delta(plan: SearchEvidencePlan, report: CandidateEvidenceReport) -> tuple[float, str]:
    if not plan.and_groups:
        return 0.0, ""
    deltas = [_group_delta(plan, report, group) for group in plan.and_groups if group]
    if not deltas:
        return 0.0, ""
    labels = [lab for _, lab in deltas]
    if all(lab == "PASS" for lab in labels):
        return W_COMPOSITE_ALL, "PASS"
    if any(lab == "PASS" for lab in labels):
        return max(d for d, lab in deltas if lab == "PASS"), "PARTIAL"
    return deltas[0]


def apply_query_evidence(
    base_score: float,
    plan: SearchEvidencePlan,
    report: CandidateEvidenceReport,
) -> tuple[float, CandidateEvidenceReport]:
    q_ev = 0.0
    penalty = 0.0
    for spec in plan.attributes:
        ev = report.concepts.get(spec.concept_id)
        if not ev:
            continue
        if spec.kind == "color":
            if ev.state == STATE_SUPPORTED:
                q_ev += W_COLOR_SUPPORT
            elif ev.state == STATE_CONTRADICTED:
                penalty += W_COLOR_CONTRA
        if spec.kind == "text" and ev.state == STATE_SUPPORTED:
            q_ev += W_TEXT_SUPPORT
    for spec in plan.objects:
        ev = report.concepts.get(spec.concept_id)
        if not ev:
            continue
        if spec.required and ev.state != STATE_SUPPORTED:
            # Aile köprüsü (ör. floral) ikinci kavram için gerçek ama
            # "kesin tür" olmayan kanıttır. Tam bilinmiyor cezası uygulanır,
            # fakat çıplak UNKNOWN kadar ağır değildir.
            family_bridge = float(ev.channels.get("family_bridge") or 0.0)
            penalty += 0.03 if family_bridge >= 0.30 else 0.12
        vis_ok = float(ev.channels.get("visual") or 0) >= VISUAL_CHANNEL_MIN
        if (
            ev.state == STATE_SUPPORTED
            and vis_ok
            and report.visual_grade in (GRADE_EXACT, GRADE_STRONG)
        ):
            q_ev += W_VISUAL_SUPPORT
        if ev.state == STATE_SUPPORTED and (
            "filename" in ev.channels or "ocr" in ev.channels
        ):
            if ev.conflict_with or report.visual_grade == GRADE_CONFLICT:
                penalty += W_CONFLICT
            elif not ev.channels.get("family") and not vis_ok:
                q_ev += W_TEXT_SUPPORT * 0.5
        elif ev.state == STATE_CONTRADICTED:
            penalty += W_CONFLICT
    if report.visual_grade == GRADE_CONFLICT:
        penalty += W_CONFLICT
    comp, label = _composite_delta(plan, report)
    if (
        len([c for c in plan.objects if c.required]) >= 2
        and report.visual_composite >= VISUAL_CHANNEL_MIN
        and all(
            (report.concepts.get(c.concept_id) or ConceptEvidence(c.concept_id, c.kind)).state
            == STATE_SUPPORTED
            for c in plan.objects if c.required
        )
    ):
        comp = max(comp, W_COMPOSITE_ALL)
        label = "PASS"
    report.query_evidence = round(q_ev, 4)
    report.composite_evidence = round(comp, 4)
    report.contradiction_penalty = round(penalty, 4)
    report.composite = label
    report.delta = round(q_ev + comp - penalty, 4)
    supported_n = sum(1 for v in report.concepts.values() if v.state == STATE_SUPPORTED)
    required_n = sum(1 for c in plan.all_concepts() if c.required)
    required_objects = [c for c in plan.objects if c.required]
    # Birden fazla nesne/terim aynı sorguda verildiyse AND mantığı uygulanır.
    # Örn. "gül leopard" yalnızca leopard olan bir deseni geçirmemeli;
    # adayda hem gül hem leopard kanıtı bulunmalıdır. Tek nesneli sorgularda
    # eski davranış korunur.
    if len(required_objects) >= 2:
        object_ok = all(
            (
                (report.concepts.get(c.concept_id) or ConceptEvidence(c.concept_id, c.kind)).state
                == STATE_SUPPORTED
            )
            or (
                float(
                    (
                        report.concepts.get(c.concept_id)
                        or ConceptEvidence(c.concept_id, c.kind)
                    ).channels.get("family_bridge")
                    or 0.0
                ) >= 0.30
                and report.visual_composite >= VISUAL_CHANNEL_MIN
            )
            for c in required_objects
        )
    else:
        object_ok = not required_objects or any(
            (report.concepts.get(c.concept_id) or ConceptEvidence(c.concept_id, c.kind)).state
            == STATE_SUPPORTED
            for c in required_objects
        )
    if required_n and supported_n >= required_n:
        report.confidence = "HIGH"
    elif supported_n:
        report.confidence = "MEDIUM"
    else:
        report.confidence = "LOW"
    final = max(0.0, min(1.0, float(base_score or 0.0) + report.delta))
    if not object_ok:
        final = min(final, 0.48)
        report.confidence = "LOW"
    return final, report


def concept_supported(report: CandidateEvidenceReport, concept_id: str) -> bool:
    ev = report.concepts.get(normalize_turkish(concept_id))
    return bool(ev and ev.state == STATE_SUPPORTED)


def inspector_lines(plan: SearchEvidencePlan, report: CandidateEvidenceReport) -> str:
    lines = ["<b>Query Evidence</b>"]
    lines.append(f"<b>Görsel:</b> {visual_verdict_text(report)}")
    for cid, ev in report.concepts.items():
        ch = " ".join(f"{k}={v:.2f}" for k, v in ev.channels.items()) or "—"
        mark = "✓" if ev.state == STATE_SUPPORTED else ("✗" if ev.state == STATE_CONTRADICTED else "?")
        lines.append(f"{mark} {cid} {ev.state} {ev.score:.2f} [{ch}]")
    if plan.and_groups:
        lines.append(f"COMPOSITE: {report.composite or '—'}")
    if report.conflict:
        lines.append("CONFLICT: true")
    if report.visual_grade:
        lines.append(f"visual_grade: {report.visual_grade}")
    lines.append(
        f"Δ {report.delta:+.3f} (q={report.query_evidence:+.3f} "
        f"comp={report.composite_evidence:+.3f} pen={report.contradiction_penalty:.3f})"
    )
    lines.append(f"CONFIDENCE: {report.confidence}")
    return "<br>".join(lines)
