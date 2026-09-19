"""Pattern Intelligence v2 — object/motif/representation/composite (query-time)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.semantic_pattern_intel import (
    CLIP_FLOOR,
    CLIP_OWN_MIN,
    SUBTYPE_MARGIN,
    PatternIntent,
    parse_pattern_intent,
)
from core.textile_terms import normalize_turkish

# Reserved Pattern DNA keys — never force-filled on existing records.
DNA_V2_KEYS = (
    "objects",
    "motifs",
    "representation",
    "composition",
    "base_motif",
    "overlay_motifs",
    "secondary_motifs",
    "textile_pattern",
)

REP_TEXTILE = "textile_pattern"
REP_PHOTO = "photograph"
REP_ILLUSTRATION = "illustration"
REP_GARMENT = "garment_photo"
REP_LOGO = "logo_monogram"
REP_UNKNOWN = "unknown"

RELATION_OVERLAY = "overlay"
RELATION_MIXED = "mixed"
RELATION_UNKNOWN = "unknown"

COMPOSITE_CLIP_MIN = 0.22
COMPOSITE_MARGIN = 0.015
PURE_OTHER_MAX = 0.24
LEAF_RESULT_CAP = 24
COMPOSITE_RESULT_CAP = 20
# Textile / motif discovery must not be sliced to the object-leaf window.
# Image search retrieves hundreds of FAISS hits; text→CLIP should keep a
# comparable candidate pool. The SearchEngine limit still applies after this.
DISCOVERY_RESULT_CAP = 400
_OBJECT_LEAF_CAP = frozenset(
    {"mouse", "bird", "butterfly", "cat", "dog", "horse", "fish", "human"}
)


def _canon(concept: str) -> str:
    return {
        "monogram": "logo",
        "cicek": "floral",
        "flower": "floral",
    }.get(concept, concept)

# Generic CLIP prompts — not a closed world list.
_CONCEPT_PROMPTS: dict[str, str] = {
    "mouse": "textile fabric print with mouse or mice motif, not a real mouse photograph",
    "bird": "textile fabric print with bird motif, not a real bird photograph",
    "cat": "textile fabric print with cat motif",
    "dog": "textile fabric print with dog motif",
    "horse": "textile fabric print with horse motif",
    "butterfly": "textile fabric print with butterfly motif",
    "leaf": "textile fabric print with leaf foliage motif",
    "fish": "textile fabric print with fish motif",
    "human": "photograph of a person or fashion model",
    "rose": "textile fabric with rose flower motif",
    "daisy": "textile fabric with daisy flower motif",
    "tulip": "textile fabric with tulip flower motif",
    "orchid": "textile fabric with orchid motif",
    "peony": "textile fabric with peony motif",
    "floral": "floral flower print textile fabric",
    "leopard": "leopard print textile fabric pattern, not a real leopard photograph",
    "snake": "snake skin print textile fabric, not a real snake photograph",
    "zebra": "zebra print textile fabric, not a real zebra photograph",
    "tiger": "tiger print textile fabric",
    "geometric": "geometric textile pattern, not floral",
    "polka_dot": "polka dot textile fabric",
    "stripe": "striped textile fabric",
    "logo": "repeating logo monogram textile print",
    "louis_vuitton": "Louis Vuitton LV monogram canvas textile print",
}

_REP_PROMPTS: dict[str, str] = {
    REP_TEXTILE: "allover repeating textile fabric print pattern, seamless fashion textile",
    REP_PHOTO: "photograph of a real animal or real object, not a fabric print",
    REP_GARMENT: "fashion photograph of a person wearing patterned clothing",
    REP_ILLUSTRATION: "illustration drawing vector clipart logo, not fabric texture",
}

_ALIASES: dict[str, str] = {
    "fare": "mouse",
    "mouse": "mouse",
    "mice": "mouse",
    "kus": "bird",
    "kuş": "bird",
    "bird": "bird",
    "kelebek": "butterfly",
    "butterfly": "butterfly",
    "yaprak": "leaf",
    "leaf": "leaf",
    "kedi": "cat",
    "cat": "cat",
    "kopek": "dog",
    "köpek": "dog",
    "dog": "dog",
    "at": "horse",
    "horse": "horse",
    "balik": "fish",
    "balık": "fish",
    "fish": "fish",
    "insan": "human",
    "human": "human",
    "model": "human",
    "cicek": "floral",
    "cicekli": "floral",
    "floral": "floral",
    "flower": "floral",
    "gul": "rose",
    "gül": "rose",
    "rose": "rose",
    "papatya": "daisy",
    "daisy": "daisy",
    "lale": "tulip",
    "tulip": "tulip",
    "orkide": "orchid",
    "orchid": "orchid",
    "sakayik": "peony",
    "peony": "peony",
    "leopar": "leopard",
    "leopard": "leopard",
    "yilan": "snake",
    "snake": "snake",
    "zebra": "zebra",
    "kaplan": "tiger",
    "tiger": "tiger",
    "geometrik": "geometric",
    "geometric": "geometric",
    "logo": "logo",
    "monogram": "logo",
    "puantiye": "polka_dot",
}

_FAMILY = {
    "mouse": "illustration_object",
    "bird": "illustration_object",
    "cat": "illustration_object",
    "dog": "illustration_object",
    "horse": "illustration_object",
    "butterfly": "illustration_object",
    "leaf": "floral",
    "fish": "illustration_object",
    "human": "garment_photo",
    "rose": "floral",
    "daisy": "floral",
    "tulip": "floral",
    "orchid": "floral",
    "peony": "floral",
    "floral": "floral",
    "leopard": "animal_print",
    "snake": "animal_print",
    "zebra": "animal_print",
    "tiger": "animal_print",
    "geometric": "geometric",
    "polka_dot": "geometric",
    "stripe": "stripe",
    "logo": "monogram_logo",
    "louis_vuitton": "monogram_logo",
}

_OVERLAY_MARKERS = ("uzerine", "üstüne", "on top", "over")
_MIXED_MARKERS = ("karisik", "karışık", "mixed", "ve", "and", "+", "ile")
_PHOTO_MARKERS = ("gercek fotograf", "gerçek fotoğraf", "photograph", "photo of")
_TEXTILE_MARKERS = ("kumas", "kumaş", "fabric", "textile", "desen", "baski", "baskı")
_EMBROIDERY_MARKERS = ("nakis", "nakış", "embroidery")


@dataclass
class PatternQueryV2:
    raw: str = ""
    objects: list[str] = field(default_factory=list)
    motifs: list[str] = field(default_factory=list)
    family: str = ""
    color: str = ""
    ground: str = ""
    scale: str = ""
    composition: str = "single"  # single | composite
    relationship: str = RELATION_UNKNOWN
    representation: str = ""
    required: list[str] = field(default_factory=list)
    primary: str = ""
    base_hint: str = ""
    overlay_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternCardV2:
    objects: list[dict[str, Any]] = field(default_factory=list)
    motifs: list[dict[str, Any]] = field(default_factory=list)
    family: str = ""
    representation: str = REP_UNKNOWN
    textile_pattern: bool | None = None
    human: bool | None = None
    garment: bool | None = None
    composition: str = "single"
    base_motif: str = ""
    overlay_motifs: list[str] = field(default_factory=list)
    secondary_motifs: list[str] = field(default_factory=list)
    relationship: str = RELATION_UNKNOWN
    dominant: str = ""
    confidence: float = 0.0
    filename_only: bool = False
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def concept_prompt(concept: str) -> str:
    return _CONCEPT_PROMPTS.get(concept) or f"textile fabric print with {concept} motif"


def representation_prompts() -> dict[str, str]:
    return dict(_REP_PROMPTS)


def _scan_concepts(text: str) -> list[str]:
    norm = normalize_turkish(text or "")
    found: list[str] = []
    for alias in sorted(_ALIASES.keys(), key=len, reverse=True):
        a = normalize_turkish(alias)
        if not a or _ALIASES[alias] in found:
            continue
        if len(a) <= 2:
            hit = bool(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", norm))
        else:
            hit = bool(re.search(rf"(?<![a-z0-9]){re.escape(a)}", norm))
        if hit:
            found.append(_ALIASES[alias])
    return found


def parse_pattern_query_v2(query: str) -> PatternQueryV2:
    raw = (query or "").strip()
    q = PatternQueryV2(raw=raw)
    if not raw:
        return q
    base = parse_pattern_intent(raw)
    q.family = base.family
    q.color = base.color
    q.ground = base.ground
    q.scale = base.scale
    norm = normalize_turkish(raw)
    concepts = _scan_concepts(raw)
    objects = [c for c in concepts if _FAMILY.get(c) in ("illustration_object", "garment_photo") or c in ("mouse", "bird", "butterfly", "cat", "dog", "horse", "fish", "human")]
    motifs = [c for c in concepts if c not in objects]
    if base.motif and base.motif not in motifs and base.motif not in objects:
        if _FAMILY.get(base.motif) == "illustration_object":
            objects.append(base.motif)
        elif base.motif:
            motifs.append(base.motif)
    q.objects = list(dict.fromkeys(_canon(c) for c in objects))
    q.motifs = [m for m in list(dict.fromkeys(_canon(c) for c in motifs)) if m not in q.objects]
    q.required = list(dict.fromkeys(q.motifs + q.objects))
    q.primary = _canon(base.motif) if base.motif else (q.required[0] if q.required else "")
    if q.primary and q.primary not in q.required:
        if _FAMILY.get(q.primary) == "illustration_object":
            q.objects.append(q.primary)
        elif q.primary:
            q.motifs.append(q.primary)
            q.required.append(q.primary)
    q.required = list(dict.fromkeys(_canon(c) for c in q.required))
    if any(m in norm for m in ("uzerine", "üstüne", "on top of", "over ")):
        q.relationship = RELATION_OVERLAY
        parts = None
        for sep in ("uzerine", "on top of", "over"):
            if sep in norm:
                left, _, right = norm.partition(sep)
                left_c = _scan_concepts(left)
                right_c = _scan_concepts(right)
                if left_c:
                    q.base_hint = left_c[0]
                if right_c:
                    q.overlay_hint = right_c[0]
                break
    elif any(m in norm for m in ("karisik", "karışık", "mixed")):
        q.relationship = RELATION_MIXED
    if "fotograf" in norm or "photograph" in norm or "photo of" in norm:
        q.representation = REP_PHOTO
    elif any(m in norm for m in _EMBROIDERY_MARKERS):
        q.representation = "embroidery"
    elif any(m in norm for m in _TEXTILE_MARKERS):
        q.representation = REP_TEXTILE
    distinct_families = {_FAMILY.get(c, c) for c in q.required}
    q.composition = "composite" if len(q.required) >= 2 else "single"
    if q.family:
        q.family = q.family if len(distinct_families) <= 1 else "composite"
    elif len(distinct_families) > 1:
        q.family = "composite"
    return q


def _present(scores: dict[str, float], name: str) -> bool:
    keys = {name, _canon(name)}
    if name == "logo":
        keys.add("monogram")
    need = 0.26 if name in ("mouse", "bird", "butterfly", "cat", "dog", "horse", "fish") else COMPOSITE_CLIP_MIN
    return any(float(scores.get(k, 0.0) or 0.0) >= need for k in keys)


def classify_representation(rep_scores: dict[str, float]) -> tuple[str, bool | None, bool | None, bool | None]:
    if not rep_scores:
        return REP_UNKNOWN, None, None, None
    ranked = sorted(rep_scores.items(), key=lambda kv: float(kv[1] or 0), reverse=True)
    top, top_s = ranked[0]
    second_s = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    if top_s < CLIP_FLOOR or top_s < second_s + SUBTYPE_MARGIN * 0.6:
        return REP_UNKNOWN, None, None, None
    textile = True if top == REP_TEXTILE else False if top == REP_PHOTO else None
    if top == REP_GARMENT:
        textile = None
    human = True if top in (REP_GARMENT, REP_PHOTO) and top == REP_GARMENT else True if top == REP_GARMENT else False if top == REP_TEXTILE else None
    garment = True if top == REP_GARMENT else False if top == REP_TEXTILE else None
    if top == REP_PHOTO:
        human = None
        garment = False
        textile = False
    return top, textile, human, garment


def build_pattern_card(
    query: PatternQueryV2,
    *,
    clip_scores: dict[str, float] | None = None,
    rep_scores: dict[str, float] | None = None,
    filename_hits: list[str] | None = None,
    indexed_type: str = "",
    pattern_family: str = "",
) -> PatternCardV2:
    scores = {k: float(v) for k, v in (clip_scores or {}).items()}
    fname = list(filename_hits or [])
    card = PatternCardV2(family=pattern_family or query.family)
    card.filename_only = bool(fname) and not any(_present(scores, n) for n in set(fname + query.required))

    objects: list[dict[str, Any]] = []
    motifs: list[dict[str, Any]] = []
    for name, sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if sc < COMPOSITE_CLIP_MIN:
            continue
        hit = {
            "id": name,
            "score": round(sc, 4),
            "family": _FAMILY.get(name, ""),
            "evidence": "clip",
            "filename": name in fname,
        }
        # Filename without CLIP is helper only — do not promote to object.
        if _FAMILY.get(name) == "illustration_object" or name in ("mouse", "bird", "butterfly", "cat", "dog", "horse", "fish", "human"):
            objects.append(hit)
        else:
            motifs.append(hit)
    card.objects = objects
    card.motifs = motifs
    card.sources = ["clip"] if scores else []
    if fname:
        card.sources.append("filename_helper")

    visual_ids = [h["id"] for h in objects + motifs]
    if len(visual_ids) >= 2:
        card.composition = "composite"
        ranked = visual_ids
        card.dominant = ranked[0]
        card.base_motif = query.base_hint or ranked[0]
        rest = [x for x in ranked[1:]]
        if query.relationship == RELATION_OVERLAY and query.overlay_hint:
            card.overlay_motifs = [query.overlay_hint]
            card.relationship = RELATION_OVERLAY
            card.base_motif = query.base_hint or ranked[0]
        else:
            card.relationship = RELATION_UNKNOWN
            card.overlay_motifs = []
            card.secondary_motifs = rest
            if query.relationship == RELATION_MIXED:
                card.relationship = RELATION_MIXED
        if card.family in ("", "unknown"):
            card.family = "composite"
    elif len(visual_ids) == 1:
        card.composition = "single"
        card.dominant = visual_ids[0]
        card.base_motif = visual_ids[0]
        fam = _FAMILY.get(visual_ids[0], card.family)
        if fam:
            card.family = fam
    else:
        card.composition = "unknown"
        if indexed_type:
            card.motifs.append(
                {"id": indexed_type, "score": 0.0, "family": _FAMILY.get(indexed_type, ""), "evidence": "index", "filename": False}
            )

    rep, textile, human, garment = classify_representation(rep_scores or {})
    if query.representation == REP_PHOTO:
        # Query asked for a photo; still require visual rep score to confirm.
        if (rep_scores or {}).get(REP_PHOTO, 0) >= CLIP_FLOOR:
            rep, textile = REP_PHOTO, False
    card.representation = rep
    card.textile_pattern = textile
    card.human = human
    card.garment = garment

    vis = [float(scores.get(n, 0)) for n in query.required] if query.required else [max(scores.values(), default=0.0)]
    if query.composition == "composite":
        mins = []
        for n in query.required:
            need = 0.26 if n in ("mouse", "bird", "butterfly", "cat", "dog", "horse", "fish") else COMPOSITE_CLIP_MIN
            mins.append(float(scores.get(n, 0.0) or 0.0) >= need)
        if not all(mins):
            card.confidence = round(min((float(scores.get(n, 0) or 0) for n in query.required), default=0.0), 4)
    card.confidence = round(min(vis) if query.composition == "composite" and vis else max(vis or [0.0]), 4)
    return card


def composite_search_score(
    query: PatternQueryV2,
    card: PatternCardV2,
    *,
    clip_scores: dict[str, float] | None = None,
    v11_score: float = 0.0,
    channel_present: dict[str, bool] | None = None,
) -> tuple[bool, float, int, str]:
    """Return keep, score, rank_bucket (0=best), reason.

    Composite AND prefers all required concepts. Missing CLIP no longer
    hard-drops a candidate that has other-channel evidence; partials rank below full matches.
    """
    scores = clip_scores or {}
    required = query.required
    present_extra = channel_present or {}
    if query.composition == "composite" and len(required) >= 2:
        present = [
            c
            for c in required
            if _present(scores, c) or present_extra.get(c) or present_extra.get(_canon(c))
        ]
        if card.representation == REP_PHOTO and query.representation != REP_PHOTO:
            return False, 0.0, 9, "non_textile_photo"
        if len(present) == len(required):
            combo = min((float(scores.get(c, 0) or 0.0) for c in required), default=0.0)
            text = float(v11_score or 0.0)
            score = max(text, combo)
            score = min(0.96, score + 0.08)
            return True, round(score, 4), 0, "composite_match"
        if present:
            # Yalnızca görsel/CLIP aday taramasında (v11_score yoksa) eski
            # kesinlik filtresini koru. Gerçek metin aramasında v11_score
            # mevcutsa kısmi aday Faz 3'e bırakılır ve tam eşleşmenin altında
            # sıralanır.
            if float(v11_score or 0.0) <= 0.0:
                return False, 0.0, 9, "missing_component:" + ",".join(
                    c for c in required if c not in present
                )
            missing = [c for c in required if c not in present]
            partial = max((float(scores.get(c, 0.0) or 0.0) for c in present), default=0.0)
            partial = max(partial, float(v11_score or 0.0))
            partial = min(0.86, partial * 0.82)
            return True, round(partial, 4), 1, "partial_composite_missing:" + ",".join(missing)
        return False, 0.0, 9, "missing_component:" + ",".join(required)

    primary = query.primary or (required[0] if required else "")
    own = float(scores.get(primary, 0.0) or 0.0)
    obj_ids = {"mouse", "bird", "butterfly", "cat", "dog", "horse", "fish"}
    if primary in obj_ids and own < max(COMPOSITE_CLIP_MIN, 0.26):
        return False, 0.0, 9, "no_visual_object"
    others = [float(v) for k, v in scores.items() if k != primary]
    other_hi = max(others) if others else 0.0
    is_photo = card.representation == REP_PHOTO and card.textile_pattern is False
    if query.representation == REP_PHOTO:
        if not is_photo:
            return False, 0.0, 9, "not_photograph"
        return True, round(max(own, v11_score), 4), 0, "photograph"

    if is_photo and primary in ("leopard", "snake", "zebra", "tiger", "mouse", "bird"):
        return False, 0.0, 9, "real_photo_not_textile"

    if not required:
        return True, float(v11_score or 0), 2, "passthrough"

    visual = own >= COMPOSITE_CLIP_MIN or (card.motifs or card.objects)
    if not visual and v11_score <= 0:
        return False, 0.0, 9, "no_visual_object"

    if primary and own < COMPOSITE_CLIP_MIN and v11_score:
        # v1.1 already accepted (index subtype / CLIP win); keep but don't claim extra objects.
        bucket = 0 if other_hi < PURE_OTHER_MAX else 1
        return True, min(float(v11_score), 0.58 if own < 0.22 else float(v11_score)), bucket, "v11_keep"

    if other_hi >= PURE_OTHER_MAX and card.composition == "composite":
        score = round(0.72 * own + 0.08, 4)
        return True, score, 1, "contains_as_composite"
    from core.semantic_pattern_intel import calibrate_text_clip

    vis = calibrate_text_clip(own) if own else 0.0
    text = float(v11_score or 0.0)
    if own >= COMPOSITE_CLIP_MIN:
        # Visual is the ranking signal. Inflated filename/v11 floors must not
        # become 96–98% for CLIP-confused geometrics.
        score = vis + min(0.05, max(0.0, text - vis) * 0.08)
        score = round(min(0.97, score), 4)
    else:
        score = round(min(text, 0.58), 4)
    return True, score, 0, "pure"


def discovery_result_cap(query: PatternQueryV2) -> int:
    """How many ranked rows v2 may keep.

    Fine-grained illustrated objects stay in a small leaf window. Textile
    motif queries (leopard/floral/geometric/paisley/…) are discovery searches
    and must keep CLIP-visual coverage comparable to image FAISS retrieval.
    """
    if getattr(query, "composition", "") == "composite":
        return COMPOSITE_RESULT_CAP
    primary = str(getattr(query, "primary", "") or "")
    family = str(getattr(query, "family", "") or "")
    if primary in _OBJECT_LEAF_CAP or family in {"illustration_object", "garment_photo"}:
        return LEAF_RESULT_CAP
    return DISCOVERY_RESULT_CAP


def rank_v2_results(
    query: PatternQueryV2,
    rows: list[Any],
    *,
    clip_by_id: dict[int, dict[str, float]] | None = None,
    rep_by_id: dict[int, dict[str, float]] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    clip_by_id = clip_by_id or {}
    rep_by_id = rep_by_id or {}
    kept: list[Any] = []
    dropped = 0
    stats = {
        "v2": True,
        "composition": query.composition,
        "required": query.required,
        "dropped": 0,
        "pure": 0,
        "composite": 0,
        "object_evidence": 0,
        "clip_evidence": 0,
        "dna_evidence": 0,
        "composition_evidence": 0,
    }
    annotated: list[tuple[int, float, Any]] = []
    for row in rows:
        fid = int(getattr(row, "file_id", 0) or 0)
        scores = dict(clip_by_id.get(fid) or {})
        dbg = getattr(row, "debug", {}) or {}
        rivals = dict(dbg.get("rival_clip") or {})
        for k, v in rivals.items():
            scores[k] = max(float(scores.get(k, 0) or 0), float(v or 0))
        obj_ids = {"mouse", "bird", "butterfly", "cat", "dog", "horse", "fish"}
        if dbg.get("clip_score") and query.primary and query.primary not in obj_ids:
            scores[query.primary] = max(float(scores.get(query.primary, 0) or 0), float(dbg.get("clip_score") or 0))
        fname = str(getattr(row, "filename", "") or "")
        fname_hits = [c for c in query.required if normalize_turkish(c) in normalize_turkish(fname)]
        ev = dbg.get("semantic_evidence") or {}
        card = build_pattern_card(
            query,
            clip_scores=scores,
            rep_scores=rep_by_id.get(fid) or {},
            filename_hits=fname_hits,
            indexed_type=str(ev.get("animal_type") or ""),
            pattern_family=str(getattr(row, "pattern_family", "") or ev.get("family") or ""),
        )
        channel_present: dict[str, bool] = {}
        for c in fname_hits:
            channel_present[c] = True
        indexed = str(ev.get("animal_type") or "")
        if indexed:
            channel_present[indexed] = True
        pf = str(getattr(row, "pattern_family", "") or ev.get("family") or "")
        if pf:
            channel_present[pf] = True
        qev = dbg.get("query_evidence_report") or {}
        if not qev:
            bd = getattr(row, "text_score_breakdown", None) or {}
            qev = bd.get("query_evidence_report") or {}
        for cid, evc in (qev.get("concepts") or {}).items():
            if str(evc.get("state") or "") == "SUPPORTED":
                channel_present[str(cid)] = True
        keep, score, bucket, reason = composite_search_score(
            query,
            card,
            clip_scores=scores,
            v11_score=float(getattr(row, "score", 0) or 0),
            channel_present=channel_present,
        )
        if not keep:
            dropped += 1
            continue
        row.score = score
        row.score_percent = round(score * 100, 1)
        row.debug = {
            **dbg,
            "pattern_intel_v2": card.to_dict(),
            "v2_reason": reason,
            "v2_bucket": bucket,
            "v2_clip": {k: round(float(v), 4) for k, v in scores.items() if float(v) >= 0.15},
        }
        if card.objects or card.motifs:
            stats["object_evidence"] += 1
        if any(float(v) >= COMPOSITE_CLIP_MIN for v in scores.values()):
            stats["clip_evidence"] += 1
        if ev.get("subtype_hit") or ev.get("dna_hit"):
            stats["dna_evidence"] += 1
        if card.composition == "composite":
            stats["composite"] += 1
            stats["composition_evidence"] += 1
        else:
            stats["pure"] += 1
        annotated.append((bucket, -score, row))
    annotated.sort(key=lambda t: (t[0], t[1]))
    cap = discovery_result_cap(query)
    kept = [t[2] for t in annotated[:cap]]
    stats["dropped"] = dropped
    stats["kept"] = len(kept)
    return kept, stats
