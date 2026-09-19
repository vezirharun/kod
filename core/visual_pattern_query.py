"""Visual Pattern Query Priority v1 — DNA/motif over filename/OCR.

Read-only ranking overlay. Does not write patterns.db / FAISS / models.
Reuses existing pattern_family / Pattern DNA / CLIP / DINO channels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.textile_terms import normalize_turkish

# Weakest textual channels must not outrank visual pattern evidence.
_TEXT_ONLY_CAP = 0.46
_VISUAL_NEGATIVE_CAP = 0.42
_NON_FAMILY = frozenset(
    {"", "unknown", "texture_ground", "plain", "document", "garment_photo"}
)

# token (folded) → concept. families are existing Pattern DNA ids.
_PATTERN_LEXICON: dict[str, dict[str, Any]] = {}


def _add(tokens: tuple[str, ...], **meta: Any) -> None:
    for tok in tokens:
        key = normalize_turkish(tok)
        if key:
            _PATTERN_LEXICON[key] = meta


_add(
    ("ekose", "kareli", "tartan", "plaid", "check", "checked", "gingham"),
    concept_id="ekose",
    kind="pattern",
    families=frozenset({"plaid_check"}),
    types=frozenset({"tartan", "plaid", "check", "gingham", "windowpane"}),
)
_add(
    ("cizgili", "cizgi", "stripe", "striped", "stripes"),
    concept_id="stripe",
    kind="pattern",
    families=frozenset({"stripe"}),
    types=frozenset({"stripe", "striped"}),
)
_add(
    ("puantiye", "polka", "polkadot"),
    concept_id="polka",
    kind="pattern",
    families=frozenset({"polka_dot"}),
    types=frozenset({"polka_dot", "polka", "dot"}),
)
_add(
    ("geometrik", "geometric", "geo"),
    concept_id="geometric",
    kind="pattern",
    families=frozenset({"geometric"}),
    types=frozenset({"geometric", "geo"}),
)
_add(
    ("cicek", "floral", "flower", "flowers"),
    concept_id="floral",
    kind="motif",
    families=frozenset({"floral"}),
    types=frozenset({"floral", "flower", "rose"}),
)
_add(
    ("yaprak", "leaf", "leaves", "foliage"),
    concept_id="leaf",
    kind="motif",
    families=frozenset({"floral"}),
    types=frozenset({"leaf", "foliage", "botanical_leaf"}),
)
_add(
    ("leopar", "leopard", "leo"),
    concept_id="leopard",
    kind="pattern",
    families=frozenset({"animal_print"}),
    types=frozenset({"leopard"}),
    animal="leopard",
)
_add(
    ("zebra",),
    concept_id="zebra",
    kind="pattern",
    families=frozenset({"animal_print"}),
    types=frozenset({"zebra"}),
    animal="zebra",
)
_add(
    ("yilan", "snake", "serpent"),
    concept_id="snake",
    kind="pattern",
    families=frozenset({"animal_print"}),
    types=frozenset({"snake", "python"}),
    animal="snake",
)
_add(
    ("kamuflaj", "camouflage", "camo"),
    concept_id="camouflage",
    kind="pattern",
    families=frozenset({"camouflage", "texture_ground"}),
    types=frozenset({"camouflage", "military_camo", "camo"}),
)


@dataclass(frozen=True)
class VisualPatternConcept:
    concept_id: str
    kind: str
    families: frozenset[str]
    types: frozenset[str]
    animal: str = ""
    token: str = ""


def _tokens(text: str) -> list[str]:
    norm = normalize_turkish(text or "")
    return [x for x in re.findall(r"[a-z0-9]+", norm) if len(x) >= 2]


def lookup_pattern_token(token: str) -> VisualPatternConcept | None:
    key = normalize_turkish(token or "")
    meta = _PATTERN_LEXICON.get(key)
    if meta is None and key.endswith("li") and len(key) > 4:
        meta = _PATTERN_LEXICON.get(key[:-2])
    if meta is None:
        return None
    return VisualPatternConcept(
        concept_id=str(meta["concept_id"]),
        kind=str(meta["kind"]),
        families=frozenset(meta["families"]),
        types=frozenset(meta["types"]),
        animal=str(meta.get("animal") or ""),
        token=key,
    )


def parse_visual_pattern_concepts(text: str) -> list[VisualPatternConcept]:
    found: list[VisualPatternConcept] = []
    seen: set[str] = set()
    for tok in _tokens(text):
        hit = lookup_pattern_token(tok)
        if hit is None or hit.concept_id in seen:
            continue
        seen.add(hit.concept_id)
        found.append(hit)
    return found


def is_visual_pattern_query(text: str) -> bool:
    return bool(parse_visual_pattern_concepts(text))


def _snake_clip_beats_leopard(result: Any) -> bool:
    """Indexed leopard DNA is often snakeskin. CLIP snake vs leopard is the live gate."""
    dbg = getattr(result, "debug", {}) or {}
    rivals = dbg.get("rival_clip") if isinstance(dbg.get("rival_clip"), dict) else {}
    if "snake" not in rivals:
        return False
    snake = _f(rivals.get("snake"))
    leopard = max(
        _f(rivals.get("leopard")),
        _f(rivals.get("jaguar")),
        _f(rivals.get("cheetah")),
    )
    return snake >= 0.22 and snake > leopard + 0.02


def _f(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _dna(result: Any) -> dict[str, Any]:
    dbg = getattr(result, "debug", {}) or {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    return dna if isinstance(dna, dict) else {}


def _result_family(result: Any) -> str:
    dna = _dna(result)
    fam = str(
        getattr(result, "pattern_family", "")
        or dna.get("family")
        or (getattr(result, "debug", {}) or {}).get("pattern_family")
        or ""
    ).strip().lower()
    return fam


def _result_type(result: Any) -> str:
    dna = _dna(result)
    dbg = getattr(result, "debug", {}) or {}
    return str(
        dna.get("motif")
        or dna.get("pattern_type")
        or dna.get("type")
        or dbg.get("animal_print_type")
        or dna.get("animal_print_type")
        or ""
    ).strip().lower()


def _text_blob(result: Any) -> str:
    dbg = getattr(result, "debug", {}) or {}
    return " ".join(
        [
            str(getattr(result, "filename", "") or ""),
            str(getattr(result, "path", "") or ""),
            str(dbg.get("ocr_text") or ""),
            str(dbg.get("folder") or ""),
        ]
    ).lower()


def visual_pattern_signals(result: Any, concepts: list[VisualPatternConcept]) -> dict[str, Any]:
    if not concepts:
        return {
            "visual": 0.0,
            "dna": 0.0,
            "family": 0.0,
            "texture": 0.0,
            "dino": 0.0,
            "clip": 0.0,
            "color": 0.0,
            "text": 0.0,
            "positive": False,
            "negative": False,
            "concept_ids": [],
        }
    wanted_fam: set[str] = set()
    wanted_types: set[str] = set()
    wanted_animal = ""
    for c in concepts:
        wanted_fam |= set(c.families)
        wanted_types |= set(c.types)
        if c.animal:
            wanted_animal = c.animal
    bd = getattr(result, "breakdown", {}) or {}
    dbg = getattr(result, "debug", {}) or {}
    dna = _dna(result)
    fam = _result_family(result)
    rtype = _result_type(result)
    dna_s = max(
        _f(bd.get("dna_score")),
        _f(bd.get("pattern_dna_score")),
        _f(dna.get("score")),
        _f(dna.get("confidence")),
    )
    fam_s = _f(bd.get("family_score"))
    tex_s = max(_f(bd.get("texture_score")), _f(bd.get("repeat_score")))
    dino_s = max(_f(dbg.get("dino_score")), _f(bd.get("dino")))
    clip_s = max(_f(dbg.get("clip_score")), _f(bd.get("clip")))
    color_s = max(_f(bd.get("color_score")), _f(bd.get("palette_similarity")))
    text_s = max(_f(bd.get("filename_score")), _f(bd.get("ocr_score")))

    family_hit = bool(fam and fam in wanted_fam)
    if fam == "texture_ground" and "texture_ground" in wanted_fam:
        family_hit = rtype in wanted_types or not rtype
    type_hit = bool(rtype and rtype in wanted_types)
    animal_hit = True
    animal = str(dna.get("animal_print_type") or dbg.get("animal_print_type") or rtype or "")
    if wanted_animal == "snake":
        # Generic animal_print / texture is not snake. Require subtype.
        animal_hit = animal in {"snake", "python"} or (rtype in wanted_types and rtype in {"snake", "python"})
    elif wanted_animal:
        animal_hit = (not animal) or animal == wanted_animal or type_hit

    known = fam not in _NON_FAMILY or (fam == "texture_ground" and rtype in wanted_types)
    mismatch = False
    if fam and fam not in _NON_FAMILY and fam not in wanted_fam:
        mismatch = True
    if wanted_animal:
        typed = str(dna.get("animal_print_type") or dbg.get("animal_print_type") or "")
        if typed and typed != wanted_animal:
            mismatch = True
        if wanted_animal == "snake" and not animal_hit:
            mismatch = True
    if rtype and wanted_types and rtype not in wanted_types and fam in wanted_fam:
        # Contrastive subtypes (snake vs leopard) are negatives; tartan vs
        # unspecified check is not.
        if wanted_animal and rtype != wanted_animal and rtype not in wanted_types:
            mismatch = True
    if wanted_animal == "snake" and _snake_clip_beats_leopard(result):
        animal_hit = True
        mismatch = False
        clip_s = max(clip_s, 0.72)

    if family_hit:
        fam_s = max(fam_s, 0.82)
        dna_s = max(dna_s, 0.78 if type_hit else 0.70)
    if type_hit:
        dna_s = max(dna_s, 0.84)

    visual = max(dna_s, fam_s * 0.95, tex_s * 0.85, dino_s * 0.80, clip_s * 0.70)
    positive = bool((family_hit or type_hit) and animal_hit and not mismatch)
    negative = bool(mismatch and known)

    needles = set()
    for c in concepts:
        needles.add(c.concept_id)
        needles.add(c.token)
        needles.update(c.types)
    blob = _text_blob(result)
    if any(n and len(n) >= 3 and n in blob for n in needles):
        text_s = max(text_s, 0.90)

    return {
        "visual": round(visual, 4),
        "dna": round(dna_s, 4),
        "family": round(fam_s, 4),
        "texture": round(tex_s, 4),
        "dino": round(dino_s, 4),
        "clip": round(clip_s, 4),
        "color": round(color_s, 4),
        "text": round(text_s, 4),
        "positive": positive,
        "negative": negative,
        "family_hit": family_hit,
        "concept_ids": [c.concept_id for c in concepts],
        "result_family": fam,
    }


def apply_visual_pattern_priority(
    results: list[Any],
    text: str,
    *,
    concepts: list[VisualPatternConcept] | None = None,
    reject_unmatched: bool = False,
) -> list[Any]:
    """Additive rerank: Pattern DNA > motif > texture > DINO > CLIP > color > OCR/filename."""
    cons = list(concepts if concepts is not None else parse_visual_pattern_concepts(text))
    if not cons or not results:
        return results
    if (getattr(results[0], "debug", {}) or {}).get("visual_pattern_priority"):
        return results

    scored: list[tuple[tuple, int, Any]] = []
    for pos, row in enumerate(results):
        sig = visual_pattern_signals(row, cons)
        base = _f(getattr(row, "score", 0.0))
        text_s = sig["text"]
        visual_only = sig["positive"]
        filename_only = text_s >= 0.70 and not visual_only and not (
            sig["dna"] >= 0.55 or sig["family"] >= 0.55 or sig["dino"] >= 0.55
        )
        bonus = 0.0
        if visual_only:
            bonus += 0.16 * sig["dna"]
            bonus += 0.12 * sig["family"]
            bonus += 0.08 * sig["texture"]
            bonus += 0.06 * sig["dino"]
            bonus += 0.04 * sig["clip"]
            bonus += 0.02 * sig["color"]
            # filename must not be the reason it sits high
            bonus += 0.01 * min(text_s, 0.3)
        else:
            bonus += 0.05 * sig["dna"]
            bonus += 0.04 * sig["family"]
            bonus += 0.03 * sig["texture"]
            bonus += 0.025 * sig["dino"]
            bonus += 0.02 * sig["clip"]
            bonus += 0.01 * sig["color"]
            bonus += 0.005 * min(text_s, 0.4)

        new_score = min(1.0, base + bonus)
        if visual_only:
            new_score = max(new_score, 0.62 + 0.28 * sig["visual"])
        if sig["negative"]:
            new_score = min(new_score, _VISUAL_NEGATIVE_CAP)
        elif filename_only:
            new_score = min(new_score, _TEXT_ONLY_CAP)

        accept = bool(
            visual_only
            or (
                not sig["negative"]
                and (sig["dna"] >= 0.55 or sig["family"] >= 0.55)
            )
        )
        row.score = new_score
        row.score_percent = round(new_score * 100.0, 1)
        dbg = dict(getattr(row, "debug", {}) or {})
        flags = {
            "visual_pattern_priority": True,
            "pattern_visual_negative": bool(sig["negative"]),
            "pattern_visual_positive": bool(visual_only),
            "pattern_text_only": bool(filename_only),
            "pattern_visual_accepted": accept,
            "pattern_query_concepts": sig["concept_ids"],
            "pattern_visual_signals": sig,
        }
        if sig["negative"]:
            for cid in sig["concept_ids"]:
                flags[f"{cid}_visual_negative"] = True
        dbg.update(flags)
        row.debug = dbg
        bd = dict(getattr(row, "breakdown", {}) or {})
        if sig["negative"] or filename_only:
            bd["filename_score"] = min(_f(bd.get("filename_score")), 0.35)
            bd["ocr_score"] = min(_f(bd.get("ocr_score")), 0.35)
            bd["brand_alias_score"] = min(_f(bd.get("brand_alias_score")), 0.35)
            row.breakdown = bd
        if reject_unmatched and not accept:
            continue
        key = (
            0 if visual_only else 1,
            1 if sig["negative"] else 0,
            1 if filename_only else 0,
            -sig["dna"],
            -sig["family"],
            -sig["texture"],
            -sig["dino"],
            -sig["clip"],
            -sig["color"],
            text_s,  # higher text is weaker tie-break (do not promote)
            -new_score,
            pos,
        )
        scored.append((key, pos, row))
    scored.sort(key=lambda x: x[0])
    return [row for _, _, row in scored]
