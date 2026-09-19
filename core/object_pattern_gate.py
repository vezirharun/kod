"""Object ≠ Pattern gate — search/score only.

When query CONTEXT is explicit (deseni/dokusu → textile; hayvanı/fotoğrafı →
object), soft-rank Pattern DNA vs real-object photo evidence.
Bare concept queries stay neutral (no forced demotion).

Does not write index, DNA, or learning. Uses indexed Pattern DNA + Visual
Concept DNA + family fields. User / learned exact evidence is never demoted.
"""
from __future__ import annotations

from typing import Any

from core.textile_terms import normalize_turkish, query_family_hints

# Motif queries where a real photo of the subject is NOT the goal.
_TEXTILE_MOTIF_FAMILIES = frozenset(
    {
        "animal_print",
        "floral",
        "stripe",
        "geometric",
        "camouflage",
        "paisley",
        "polka_dot",
        "lace",
        "baroque",
        "ethnic_print",
        "watercolor",
    }
)

_PHOTO_FAMILIES = frozenset(
    {
        "garment_photo",
        "icon_logo_non_textile",
        "document",
        "unknown",
        "",
    }
)

_OBJECT_CATS = frozenset(
    {
        "animal",
        "hayvan",
        "plant",
        "bitki",
        "fruit",
        "meyve",
        "person",
        "insan",
        "body",
        "yüz",
        "yuz",
    }
)

_ANIMAL_LEMMAS = frozenset(
    {
        "tiger",
        "kaplan",
        "leopard",
        "leopar",
        "zebra",
        "snake",
        "yilan",
        "yılan",
        "lion",
        "aslan",
        "cat",
        "kedi",
        "dog",
        "kopek",
        "köpek",
        "animal",
        "hayvan",
    }
)

_FLORAL_LEMMAS = frozenset(
    {
        "flower",
        "floral",
        "cicek",
        "çiçek",
        "rose",
        "gul",
        "gül",
        "tulip",
        "lale",
        "daisy",
        "papatya",
        "leaf",
        "yaprak",
        "plant",
        "bitki",
    }
)

_PERSON_LEMMAS = frozenset(
    {
        "person",
        "kisi",
        "kişi",
        "human",
        "insan",
        "woman",
        "kadin",
        "kadın",
        "man",
        "erkek",
        "face",
        "yuz",
        "yüz",
    }
)

_MAX_PENALTY = 0.32
_STRONG_TEXTILE = 0.55
_WEAK_TEXTILE = 0.35


def _tm(result: Any) -> dict[str, Any]:
    dbg = getattr(result, "debug", None) or {}
    if isinstance(dbg, dict):
        tm = dbg.get("texture_map")
        if isinstance(tm, dict):
            return tm
    if isinstance(result, dict):
        tm = result.get("texture_map")
        if isinstance(tm, dict):
            return tm
    return {}


def _dbg(result: Any) -> dict[str, Any]:
    d = getattr(result, "debug", None)
    return d if isinstance(d, dict) else {}


def _family(result: Any) -> str:
    fam = (
        getattr(result, "pattern_family", None)
        or _dbg(result).get("pattern_family")
        or _dbg(result).get("result_family")
        or _tm(result).get("pattern_family")
        or ""
    )
    return str(fam or "").strip().lower()


def _pattern_dna(result: Any) -> dict[str, Any]:
    tm = _tm(result)
    dna = tm.get("pattern_dna")
    return dna if isinstance(dna, dict) else {}


def _vcd(result: Any) -> dict[str, Any]:
    try:
        from core.visual_concept_dna import read_visual_concept_dna

        return read_visual_concept_dna(result) or {}
    except Exception:
        tm = _tm(result)
        blob = tm.get("visual_concept_dna")
        return blob if isinstance(blob, dict) else {}


def _user_protected(result: Any) -> bool:
    dbg = _dbg(result)
    if getattr(result, "is_self_match", False):
        return True
    if dbg.get("protected_exact") or dbg.get("user_taught_positive"):
        return True
    if dbg.get("learned_concept_exact"):
        return True
    if dbg.get("manual_user") or dbg.get("category_source") == "manual_user":
        return True
    if dbg.get("user_labeled"):
        return True
    return False


def motif_query_context(
    query_text: str = "",
    *,
    query_family: str = "",
) -> dict[str, Any]:
    """Return textile/object preference from generic query CONTEXT (not concept id).

    Bare concept → no forced preference (natural ranking by visual evidence).
    Explicit deseni/dokusu → textile; hayvanı/fotoğrafı → object/photo.
    """
    fam = str(query_family or "").strip().lower()
    hints: dict[str, str] = {}
    if not fam and query_text:
        try:
            hints = query_family_hints(query_text) or {}
            fam = str(hints.get("pattern_family") or "").strip().lower()
        except Exception:
            hints = {}
    norm = normalize_turkish(query_text or "")
    lemmas: set[str] = set()
    animal = fam == "animal_print" or any(
        t in norm for t in ("kaplan", "tiger", "leopar", "leopard", "zebra", "yilan", "hayvan", "animal")
    )
    floral = fam == "floral" or any(
        t in norm for t in ("cicek", "flower", "floral", "gul", "rose", "lale", "tulip", "papatya", "daisy")
    )
    person = any(t in norm for t in ("figurlu", "figurlu baski", "insan figuru", "kisi baski"))
    logo = fam == "monogram_logo" or any(
        t in norm for t in ("logo desen", "monogram", "logo repeat")
    )
    if animal:
        lemmas |= _ANIMAL_LEMMAS
        if not fam:
            fam = "animal_print"
    if floral:
        lemmas |= _FLORAL_LEMMAS
        if not fam:
            fam = "floral"
    if person:
        lemmas |= _PERSON_LEMMAS
    if hints.get("animal_print_type"):
        lemmas.add(normalize_turkish(hints["animal_print_type"]))

    # Generic CONTEXT extraction (concept-agnostic markers).
    vctx: dict[str, Any] = {}
    try:
        from core.query_attribute_intel import extract_query_visual_context

        vctx = extract_query_visual_context(query_text)
    except Exception:
        vctx = {}
    context = str(vctx.get("context") or "none")
    visual_type = str(vctx.get("visual_type") or "general")
    explicit = bool(vctx.get("explicit"))

    wants_textile = context in {"pattern", "texture"}
    wants_object = context in {"object", "photo"}
    # Logo-print phrasing still implies textile when desen/repeat present.
    if logo and "desen" in norm and not wants_object:
        wants_textile = True
        context = context if context != "none" else "pattern"

    return {
        "wants_textile": wants_textile,
        "wants_object": wants_object,
        "context": context,
        "visual_type": visual_type,
        "explicit": explicit,
        "concept_core": str(vctx.get("concept_core") or ""),
        "family": fam,
        "lemmas": lemmas,
        "animal": animal,
        "floral": floral,
        "person": person,
    }


def textile_strength(result: Any, *, query_family: str = "") -> float:
    """0..1 — how strongly this file looks like a textile pattern."""
    fam = _family(result)
    tm = _tm(result)
    dna = _pattern_dna(result)
    score = 0.0

    if fam and fam not in _PHOTO_FAMILIES and fam != "plain":
        score += 0.28
    if query_family and fam == query_family:
        score += 0.22
    if fam == "animal_print" or fam == "floral":
        score += 0.08

    conf = float(dna.get("pattern_dna_confidence") or dna.get("confidence") or 0)
    if conf >= 0.45:
        score += 0.18
    elif conf >= 0.25:
        score += 0.10

    repeat = str(dna.get("repeat_type") or dna.get("repeat") or dna.get("repeat_class") or "")
    if repeat and repeat.lower() not in {"", "unknown", "random"}:
        score += 0.12
    density = str(dna.get("density") or "")
    if density:
        score += 0.06
    scale = str(dna.get("scale") or "")
    if scale:
        score += 0.04
    motif = str(dna.get("motif") or dna.get("motif_class") or "")
    if motif:
        score += 0.06

    rd = float(tm.get("repeat_density") or 0)
    if rd >= 0.35:
        score += 0.14
    elif rd >= 0.15:
        score += 0.08

    org = float(tm.get("organic_blob_score") or 0)
    stripe = float(tm.get("stripe_score") or 0)
    if org >= 0.35 or stripe >= 0.35:
        score += 0.08

    if fam in {"garment_photo", "icon_logo_non_textile", "document"}:
        score *= 0.35

    return max(0.0, min(1.0, score))


def object_photo_strength(
    result: Any,
    *,
    lemmas: set[str] | None = None,
    animal: bool = False,
    floral: bool = False,
    person: bool = False,
) -> float:
    """0..1 — real-object / photo evidence for the query subject."""
    fam = _family(result)
    dna = _vcd(result)
    lemmas = {normalize_turkish(x) for x in (lemmas or set()) if x}
    score = 0.0
    detected_hit = 0.0
    concept_hit = 0.0

    for row in list(dna.get("objects") or []):
        if not isinstance(row, dict):
            continue
        lemma = normalize_turkish(str(row.get("lemma") or row.get("label") or ""))
        cat = normalize_turkish(str(row.get("category") or row.get("category_tr") or ""))
        conf = float(row.get("confidence") or 0)
        detected = bool(row.get("detected"))
        lab = normalize_turkish(str(row.get("label_tr") or row.get("label") or ""))
        match_lemma = bool(lemmas) and (
            lemma in lemmas or lab in lemmas or any(L in lemma or L in lab for L in lemmas if len(L) >= 4)
        )
        match_cat = cat in _OBJECT_CATS and (
            (animal and cat in {"animal", "hayvan"})
            or (floral and cat in {"plant", "bitki", "fruit", "meyve"})
            or (person and cat in {"person", "insan", "body", "yüz", "yuz"})
            or match_lemma
        )
        if not (match_lemma or match_cat):
            continue
        if detected:
            detected_hit = max(detected_hit, conf)
        else:
            concept_hit = max(concept_hit, conf * 0.55)

    for row in list(dna.get("concepts") or []):
        if not isinstance(row, dict):
            continue
        lemma = normalize_turkish(str(row.get("lemma") or row.get("label") or ""))
        if lemmas and lemma in lemmas:
            # visual_concept alone is weak photo evidence (may be print motif)
            concept_hit = max(concept_hit, float(row.get("confidence") or 0) * 0.35)

    score = max(detected_hit, concept_hit)
    if fam in {"garment_photo", "icon_logo_non_textile"}:
        score = max(score, 0.45 if fam == "garment_photo" else 0.55)
    if fam == "document":
        score = max(score, 0.40)

    dbg = _dbg(result)
    if person and (dbg.get("face_gender_match") or dbg.get("human_semantic_only")):
        # Person-search mode is handled elsewhere; for motif queries this is photo-like.
        score = max(score, 0.65)

    return max(0.0, min(1.0, score))


def adjust_score_for_object_vs_pattern(
    score: float,
    result: Any,
    *,
    query_text: str = "",
    query_family: str = "",
    ctx: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return (new_score, meta). Soft preference only when CONTEXT is explicit."""
    meta: dict[str, Any] = {"applied": False, "penalty": 0.0, "boost": 0.0}
    ctx = ctx or motif_query_context(query_text, query_family=query_family)
    wants_textile = bool(ctx.get("wants_textile"))
    wants_object = bool(ctx.get("wants_object"))
    if not wants_textile and not wants_object:
        meta["reason"] = "bare_concept_neutral"
        meta["context"] = ctx.get("context") or "none"
        return float(score), meta
    if _user_protected(result):
        meta["skipped"] = "user_protected"
        return float(score), meta

    q_fam = str(ctx.get("family") or query_family or "")
    textile = textile_strength(result, query_family=q_fam)
    obj = object_photo_strength(
        result,
        lemmas=set(ctx.get("lemmas") or ()),
        animal=bool(ctx.get("animal")),
        floral=bool(ctx.get("floral")),
        person=bool(ctx.get("person")),
    )
    meta["textile"] = round(textile, 4)
    meta["object"] = round(obj, 4)
    meta["family"] = _family(result)
    meta["context"] = ctx.get("context") or "none"
    meta["visual_type"] = ctx.get("visual_type") or "general"

    # Explicit object/photo CONTEXT → demote pure textile prints.
    if wants_object and not wants_textile:
        if obj >= 0.45 and textile < _WEAK_TEXTILE:
            boost = 0.02 if obj >= 0.7 else 0.0
            if boost:
                score = min(1.0, float(score) + boost)
            meta["applied"] = True
            meta["boost"] = boost
            meta["reason"] = "object_photo_preferred"
            return float(score), meta
        if textile >= _STRONG_TEXTILE and obj < 0.40:
            gap = max(0.0, textile - obj)
            penalty = min(_MAX_PENALTY, 0.08 + 0.22 * gap)
            score = max(0.0, float(score) * (1.0 - penalty))
            meta["applied"] = True
            meta["penalty"] = round(penalty, 4)
            meta["reason"] = "pattern_over_object"
            return float(score), meta
        if textile > obj + 0.20 and obj < 0.45:
            penalty = min(0.12, 0.05 + 0.12 * (textile - obj))
            score = max(0.0, float(score) * (1.0 - penalty))
            meta["applied"] = True
            meta["penalty"] = round(penalty, 4)
            meta["reason"] = "textile_tilted"
            return float(score), meta
        meta["reason"] = "object_neutral"
        return float(score), meta

    # Explicit pattern/texture CONTEXT → demote object photos (existing path).
    if not wants_textile:
        return float(score), meta

    # Strong textile DNA/family → never penalize (print may still mention "tiger").
    if textile >= _STRONG_TEXTILE:
        boost = 0.0
        if textile >= 0.72 and obj < 0.55:
            boost = 0.025
            score = min(1.0, float(score) + boost)
        meta["applied"] = True
        meta["boost"] = boost
        meta["reason"] = "strong_textile"
        return float(score), meta

    # Object/photo-like + weak textile → demote.
    if obj >= 0.45 and textile < _WEAK_TEXTILE:
        gap = max(0.0, obj - textile)
        penalty = min(_MAX_PENALTY, 0.10 + 0.28 * gap)
        # Logo icons slightly softer unless query is logo-print.
        if _family(result) == "icon_logo_non_textile" and "logo" not in normalize_turkish(
            query_text
        ):
            penalty = min(penalty, 0.22)
        score = max(0.0, float(score) * (1.0 - penalty))
        meta["applied"] = True
        meta["penalty"] = round(penalty, 4)
        meta["reason"] = "object_photo_over_pattern"
        return float(score), meta

    # Mid band: mild tilt when object dominates textile.
    if obj > textile + 0.20 and textile < _STRONG_TEXTILE:
        penalty = min(0.12, 0.06 + 0.15 * (obj - textile))
        score = max(0.0, float(score) * (1.0 - penalty))
        meta["applied"] = True
        meta["penalty"] = round(penalty, 4)
        meta["reason"] = "object_tilted"
        return float(score), meta

    meta["reason"] = "neutral"
    return float(score), meta


def apply_object_pattern_gate(
    results: list[Any],
    query_text: str = "",
    *,
    query_family: str = "",
) -> list[Any]:
    """In-place soft re-score when query CONTEXT is explicit (pattern/object)."""
    if not results:
        return results
    ctx = motif_query_context(query_text, query_family=query_family)
    if not ctx.get("wants_textile") and not ctx.get("wants_object"):
        return results
    for rec in results:
        try:
            old = float(getattr(rec, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        new, meta = adjust_score_for_object_vs_pattern(
            old, rec, query_text=query_text, query_family=query_family, ctx=ctx
        )
        if not meta.get("applied"):
            continue
        rec.score = new
        if hasattr(rec, "score_percent"):
            rec.score_percent = round(new * 100, 1)
        dbg = dict(_dbg(rec))
        dbg["object_pattern_gate"] = meta
        rec.debug = dbg
    return results
