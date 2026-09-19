"""Vezir Semantic Pattern Intelligence v1.

Hybrid text→visual ranking helpers. Does not rebuild FAISS, change
index lifecycle, or replace visual DINO/CLIP image search.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.textile_terms import normalize_turkish

BLOCKED_STATUSES = frozenset({"missing", "excluded_internal"})

CLIP_FLOOR = 0.20
CLIP_ONLY_FLOOR = 0.24
CLIP_OWN_MIN = 0.22
CLIP_VISUAL_WIN_MIN = 0.26
SUBTYPE_MARGIN = 0.025
CLIP_BLEND_WEIGHT = 0.22
CLIP_MAX_CONTRIB = 0.22

TIER_EXACT = "Exact"
TIER_VERY_SIMILAR = "Very Similar"
TIER_VARIANT = "Variant"
TIER_SAME_FAMILY = "Same Family"
TIER_SEMANTIC = "Semantic Similar"

TIER_TO_CATEGORY = {
    TIER_EXACT: "exact",
    TIER_VERY_SIMILAR: "near_variant",
    TIER_VARIANT: "similar",
    TIER_SAME_FAMILY: "textile_texture",
    TIER_SEMANTIC: "style",
}

LEAF_SEMANTIC_CAP = 8
PARENT_SEMANTIC_CAP = 16

SHORT_BRAND_ALIASES = frozenset({"lv", "gg", "ff", "cd", "ysl", "mk", "tb", "dg"})

# concept_id → family, motif, specificity, clip prompt
_ONTOLOGY: dict[str, dict[str, str]] = {
    "floral": {
        "family": "floral",
        "motif": "floral",
        "specificity": "parent",
        "prompt": "textile fabric with floral flower print pattern",
    },
    "rose": {
        "family": "floral",
        "motif": "rose",
        "specificity": "leaf",
        "prompt": "textile fabric with rose flower motif, not generic floral",
    },
    "daisy": {
        "family": "floral",
        "motif": "daisy",
        "specificity": "leaf",
        "prompt": "textile fabric with daisy flowers, small daisy print",
    },
    "tulip": {
        "family": "floral",
        "motif": "tulip",
        "specificity": "leaf",
        "prompt": "textile fabric with tulip flower motif",
    },
    "orchid": {
        "family": "floral",
        "motif": "orchid",
        "specificity": "leaf",
        "prompt": "textile fabric with orchid flower motif",
    },
    "peony": {
        "family": "floral",
        "motif": "peony",
        "specificity": "leaf",
        "prompt": "textile fabric with peony flower motif",
    },
    "sunflower": {
        "family": "floral",
        "motif": "sunflower",
        "specificity": "leaf",
        "prompt": "textile fabric with sunflower motif",
    },
    "geometric": {
        "family": "geometric",
        "motif": "geometric",
        "specificity": "parent",
        "prompt": "geometric textile pattern, not floral",
    },
    "polka_dot": {
        "family": "geometric",
        "motif": "polka_dot",
        "specificity": "leaf",
        "prompt": "polka dot textile, regularly spaced round dots on fabric",
    },
    "stripe": {
        "family": "stripe",
        "motif": "stripe",
        "specificity": "leaf",
        "prompt": "striped textile fabric, parallel lines pattern",
    },
    "square": {
        "family": "geometric",
        "motif": "square",
        "specificity": "leaf",
        "prompt": "geometric square check textile pattern",
    },
    "triangle": {
        "family": "geometric",
        "motif": "triangle",
        "specificity": "leaf",
        "prompt": "geometric triangle textile pattern",
    },
    "animal_print": {
        "family": "animal_print",
        "motif": "animal_print",
        "specificity": "parent",
        "prompt": "animal print textile pattern",
    },
    "leopard": {
        "family": "animal_print",
        "motif": "leopard",
        "specificity": "leaf",
        "prompt": "leopard print fabric, not zebra not snake",
    },
    "zebra": {
        "family": "animal_print",
        "motif": "zebra",
        "specificity": "leaf",
        "prompt": "zebra stripe animal print fabric",
    },
    "snake": {
        "family": "animal_print",
        "motif": "snake",
        "specificity": "leaf",
        "prompt": "repeating diamond snakeskin scales, python skin tessellation print, not leopard spots not cheetah rosettes not zebra stripes",
    },
    "tiger": {
        "family": "animal_print",
        "motif": "tiger",
        "specificity": "leaf",
        "prompt": "tiger stripe print fabric",
    },
    "leaf": {
        "family": "floral",
        "motif": "leaf",
        "specificity": "leaf",
        "prompt": "textile fabric with leaf foliage motif",
    },
    "camouflage": {
        "family": "camouflage",
        "motif": "camouflage",
        "specificity": "parent",
        "prompt": "camouflage military textile pattern, not animal print",
    },
    "marble": {
        "family": "marble_abstract",
        "motif": "marble",
        "specificity": "parent",
        "prompt": "marble stone veining texture, not leopard spots",
    },
    "abstract": {
        "family": "marble_abstract",
        "motif": "abstract",
        "specificity": "parent",
        "prompt": "abstract organic blotch textile pattern, not animal print",
    },
    "plaid": {
        "family": "plaid_check",
        "motif": "plaid",
        "specificity": "parent",
        "prompt": "plaid tartan check textile",
    },
    "tartan": {
        "family": "plaid_check",
        "motif": "tartan",
        "specificity": "leaf",
        "prompt": "tartan plaid textile",
    },
    "paisley": {
        "family": "paisley",
        "motif": "paisley",
        "specificity": "parent",
        "prompt": "paisley boteh textile print",
    },
    "monogram": {
        "family": "monogram_logo",
        "motif": "monogram",
        "specificity": "parent",
        "prompt": "repeating logo monogram textile print",
    },
    "louis vuitton": {
        "family": "monogram_logo",
        "motif": "louis_vuitton",
        "specificity": "brand",
        "prompt": "Louis Vuitton LV monogram canvas textile print",
    },
    "gucci": {
        "family": "monogram_logo",
        "motif": "gucci",
        "specificity": "brand",
        "prompt": "Gucci GG interlocking logo textile print",
    },
    "chanel": {
        "family": "monogram_logo",
        "motif": "chanel",
        "specificity": "brand",
        "prompt": "Chanel CC interlocking logo textile print",
    },
    "dior": {
        "family": "monogram_logo",
        "motif": "dior",
        "specificity": "brand",
        "prompt": "Dior logo monogram textile print",
    },
    "ysl": {
        "family": "monogram_logo",
        "motif": "ysl",
        "specificity": "brand",
        "prompt": "YSL Yves Saint Laurent logo textile print",
    },
    "fendi": {
        "family": "monogram_logo",
        "motif": "fendi",
        "specificity": "brand",
        "prompt": "Fendi FF logo textile print",
    },
}

_ALIAS_TO_CONCEPT: dict[str, str] = {}
for _cid, _meta in _ONTOLOGY.items():
    _ALIAS_TO_CONCEPT[normalize_turkish(_cid)] = _cid

_ALIAS_TO_CONCEPT.update(
    {
        "cicek": "floral",
        "cicekli": "floral",
        "flower": "floral",
        "flowers": "floral",
        "gul": "rose",
        "gül": "rose",
        "rose": "rose",
        "roses": "rose",
        "papatya": "daisy",
        "papatyali": "daisy",
        "daisy": "daisy",
        "lale": "tulip",
        "tulip": "tulip",
        "orkide": "orchid",
        "orchid": "orchid",
        "sakayik": "peony",
        "peony": "peony",
        "aycicegi": "sunflower",
        "sunflower": "sunflower",
        "yaprak": "leaf",
        "leaf": "leaf",
        "geometrik": "geometric",
        "geo": "geometric",
        "puantiye": "polka_dot",
        "puantiyeli": "polka_dot",
        "polka": "polka_dot",
        "polka dot": "polka_dot",
        "polka_dot": "polka_dot",
        "dot": "polka_dot",
        "dots": "polka_dot",
        "noktali": "polka_dot",
        "noktalı": "polka_dot",
        "cizgi": "stripe",
        "cizgili": "stripe",
        "stripe": "stripe",
        "striped": "stripe",
        "kare": "square",
        "kareli": "plaid",
        "ucgen": "triangle",
        "triangle": "triangle",
        "leopar": "leopard",
        "leopard": "leopard",
        "leopard print": "leopard",
        "leopar desen": "leopard",
        "leopar print": "leopard",
        "zebra": "zebra",
        "zebra print": "zebra",
        "yilan": "snake",
        "yilan derisi": "snake",
        "snake": "snake",
        "snake print": "snake",
        "snakeskin": "snake",
        "snake skin": "snake",
        "python": "snake",
        "zebra desen": "zebra",
        "kaplan": "tiger",
        "tiger": "tiger",
        "tiger stripe": "tiger",
        "animal print": "animal_print",
        "hayvan deseni": "animal_print",
        "hayvan": "animal_print",
        "ekose": "plaid",
        "plaid": "plaid",
        "check": "plaid",
        "checked": "plaid",
        "tartan": "tartan",
        "paisley": "paisley",
        "sal": "paisley",
        "camouflage": "camouflage",
        "camo": "camouflage",
        "kamuflaj": "camouflage",
        "marble": "marble",
        "mermer": "marble",
        "abstract": "abstract",
        "soyut": "abstract",
        "monogram": "monogram",
        "logo": "monogram",
        "louis vuitton": "louis vuitton",
        "lv": "louis vuitton",
        "vuitton": "louis vuitton",
        "gucci": "gucci",
        "gg": "gucci",
        "chanel": "chanel",
        "dior": "dior",
        "christian dior": "dior",
        "ysl": "ysl",
        "yves saint laurent": "ysl",
        "saint laurent": "ysl",
        "fendi": "fendi",
        "ff": "fendi",
    }
)

_GROUND_PHRASES: dict[str, str] = {
    "beyaz zemin": "white",
    "siyah zemin": "black",
    "lacivert zemin": "navy",
    "navy ground": "navy",
    "white ground": "white",
    "black ground": "black",
    "kirmizi zemin": "red",
    "krem zemin": "cream",
}

_COLOR_OVERRIDE: dict[str, str] = {
    "siyah": "black",
    "black": "black",
    "beyaz": "white",
    "white": "white",
    "lacivert": "navy",
    "navy": "navy",
}

PROTECTED_TERMS = frozenset(
    {
        "lale",
        "tulip",
        "papatya",
        "daisy",
        "gül",
        "gul",
        "orkide",
        "orchid",
        "sakayik",
        "peony",
        "puantiye",
        "noktali",
        "leopar",
        "zebra",
        "yilan",
        "kaplan",
        "snakeskin",
        "yaprak",
        "leaf",
        "karga",
        "crow",
        "leylek",
        "stork",
        "togg",
        "karinca",
        "karınca",
    }
)


@dataclass
class PatternIntent:
    raw: str = ""
    family: str = ""
    motif: str = ""
    specificity: str = "parent"  # parent | leaf | brand
    color: str = ""
    ground: str = ""
    scale: str = ""
    structure: str = ""
    brand: str = ""
    brand_strong: bool = False
    clip_prompts: list[str] = field(default_factory=list)
    concept_id: str = ""

    def as_key(self) -> str:
        return f"{self.family}:{self.motif}:{self.specificity}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternEvidence:
    family: str = ""
    motif: str = ""
    animal_type: str = ""
    filename_hit: bool = False
    ocr_hit: bool = False
    brand_text_hit: bool = False
    brand_strong_hit: bool = False
    subtype_hit: bool = False
    family_hit: bool = False
    dna_hit: bool = False
    clip_score: float = 0.0
    rival_clip: dict[str, float] = field(default_factory=dict)
    clip_margin: float = 0.0
    negative_motifs: list[str] = field(default_factory=list)
    visual_win: bool = False
    animal_score: float = 0.0
    feedback_hit: bool = False
    feedback_labels: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_concept(text: str) -> str:
    norm = normalize_turkish(text or "")
    if not norm:
        return ""
    if norm in _ALIAS_TO_CONCEPT:
        return _ALIAS_TO_CONCEPT[norm]
    parts = re.split(r"\s+", norm)
    # longest phrase first
    for n in range(min(4, len(parts)), 0, -1):
        for i in range(0, len(parts) - n + 1):
            phrase = " ".join(parts[i : i + n])
            if phrase in _ALIAS_TO_CONCEPT:
                return _ALIAS_TO_CONCEPT[phrase]
    return _ALIAS_TO_CONCEPT.get(parts[-1], "") if parts else ""


def competing_leaf_motifs(family: str, motif: str) -> list[str]:
    if not family or not motif:
        return []
    out: list[str] = []
    for meta in _ONTOLOGY.values():
        if meta.get("family") != family or meta.get("specificity") != "leaf":
            continue
        m = str(meta.get("motif") or "")
        if m and m not in out:
            out.append(m)
    return out


# Textile CLIP often confuses animal print with geometric/paisley/marble.
# These are generic family-confusion prompts, not per-keyword special cases.
_CONFUSION_FAMILY_PROMPTS: dict[str, str] = {
    "geometric": "geometric diamond lattice textile pattern, not animal print",
    "paisley": "paisley boteh textile print, not animal print",
    "camouflage": "camouflage military textile pattern, not animal print",
    "marble": "marble stone veining texture, not leopard spots",
    "floral": "floral flower textile print, not animal print",
    "abstract": "abstract organic blotch textile pattern, not animal print",
}


def calibrate_text_clip(clip_score: float) -> float:
    """Monotonic map from OpenCLIP text–image cosine to relevance.

    Text–image cosine lives near 0.20–0.38. Image–image cosine lives near
    0.65–0.90. Mapping 0.38 → 0.96 makes geometric CLIP noise look like a
    96% match. Anchors are quality-aligned with image search (strong ≈ 0.75),
    not a linear stretch of cosine onto percent.
    """
    x = max(0.0, float(clip_score or 0.0))
    anchors = (
        (0.00, 0.00),
        (0.16, 0.10),
        (0.20, 0.22),
        (0.24, 0.36),
        (0.28, 0.48),
        (0.32, 0.60),
        (0.36, 0.70),
        (0.40, 0.78),
        (0.46, 0.86),
        (0.55, 0.90),
        (1.00, 0.93),
    )
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            span = x1 - x0
            t = 0.0 if span <= 0 else (x - x0) / span
            t = max(0.0, min(1.0, t))
            return round(y0 + t * (y1 - y0), 4)
    return 0.93


def competing_subtype_prompts(intent: PatternIntent) -> dict[str, str]:
    """Same-family rivals plus generic textile confusion families."""
    if intent.specificity != "leaf" or not intent.family:
        return {}
    prompts: dict[str, str] = {}
    for meta in _ONTOLOGY.values():
        if meta.get("family") != intent.family or meta.get("specificity") != "leaf":
            continue
        m = str(meta.get("motif") or "")
        p = str(meta.get("prompt") or "")
        if m and p:
            prompts[m] = p
    skip_confusion = intent.family in {
        "illustration_object",
        "garment_photo",
        "monogram_logo",
    }
    if not skip_confusion:
        for name, prompt in _CONFUSION_FAMILY_PROMPTS.items():
            if name != intent.motif and name != intent.family:
                prompts.setdefault(name, prompt)
    return prompts


def parse_pattern_intent(query: str) -> PatternIntent:
    from core.natural_language_query import parse_natural_query
    raw = (query or "").strip()
    intent = PatternIntent(raw=raw)
    if not raw:
        return intent
    norm = normalize_turkish(raw)
    parsed = parse_natural_query(raw)

    for phrase, ground in _GROUND_PHRASES.items():
        if normalize_turkish(phrase) in norm:
            intent.ground = ground
            break

    concept = resolve_concept(raw)
    if not concept and parsed.motif:
        concept = resolve_concept(parsed.motif) or parsed.motif
    meta = _ONTOLOGY.get(concept) or {}
    intent.concept_id = concept
    intent.family = meta.get("family") or ""
    intent.motif = meta.get("motif") or parsed.motif or ""
    intent.specificity = meta.get("specificity") or (
        "leaf" if parsed.motif and parsed.motif not in ("floral", "animal_print") else "parent"
    )
    prompt = meta.get("prompt") or ""
    if prompt:
        intent.clip_prompts = [prompt, raw]
    else:
        intent.clip_prompts = [f"textile fabric pattern {raw}", raw]

    if parsed.color:
        intent.color = parsed.color
    for tok, col in _COLOR_OVERRIDE.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", norm):
            if not intent.ground and not intent.color:
                intent.color = col
    has_black = bool(re.search(r"(?<![a-z0-9])(siyah|black)(?![a-z0-9])", norm))
    has_white = bool(re.search(r"(?<![a-z0-9])(beyaz|white)(?![a-z0-9])", norm))
    if intent.ground == "white" and has_black:
        intent.color = "black"
    elif intent.ground == "black" and has_white:
        intent.color = "white"
    elif intent.ground == "navy" and has_white:
        intent.color = "white"

    intent.scale = parsed.repeat or ""
    if parsed.motif in ("stripe",) or "cizgili" in norm or "striped" in norm:
        intent.structure = "stripe"
    if "geometrik" in norm or "geometric" in norm:
        intent.family = "geometric"
        if intent.motif == "stripe":
            intent.structure = "stripe"
            intent.specificity = "parent"
        elif not intent.motif or intent.motif == "geometric":
            intent.motif = "geometric"
            intent.specificity = "parent"
            intent.clip_prompts = [_ONTOLOGY["geometric"]["prompt"], raw]

    if parsed.brand:
        intent.brand = parsed.brand
        q_tokens = set(re.findall(r"[a-z0-9]+", norm))
        intent.brand_strong = not (q_tokens & SHORT_BRAND_ALIASES) or bool(
            set(normalize_turkish(parsed.brand).split()) & q_tokens
            and len(normalize_turkish(parsed.brand)) >= 4
        )
        # LV / GG alone → weak alias
        if q_tokens <= SHORT_BRAND_ALIASES or q_tokens & SHORT_BRAND_ALIASES and len(q_tokens) <= 2:
            intent.brand_strong = False
        if "louis" in norm or "vuitton" in norm or "gucci" in norm:
            if "lv" not in q_tokens or "louis" in norm:
                intent.brand_strong = len(normalize_turkish(parsed.brand)) >= 5
        intent.specificity = "brand"
        brand_concept = resolve_concept(parsed.brand) or resolve_concept(raw)
        bmeta = _ONTOLOGY.get(brand_concept) or {}
        if bmeta:
            intent.family = bmeta.get("family") or intent.family
            intent.motif = bmeta.get("motif") or intent.motif
            if bmeta.get("prompt"):
                intent.clip_prompts = [bmeta["prompt"], raw]

    if intent.family == "animal_print" and intent.specificity == "leaf":
        intent.clip_prompts = [meta.get("prompt") or intent.clip_prompts[0], raw]
    return intent


def _texture_map(rec: dict[str, Any]) -> dict[str, Any]:
    tm = rec.get("texture_map") or {}
    if isinstance(tm, str):
        try:
            tm = json.loads(tm)
        except json.JSONDecodeError:
            tm = {}
    return tm if isinstance(tm, dict) else {}


def record_is_searchable(rec: dict[str, Any] | None) -> bool:
    if not rec:
        return False
    status = str(rec.get("status") or "")
    if status in BLOCKED_STATUSES:
        return False
    return True


def _blob(rec: dict[str, Any]) -> str:
    return normalize_turkish(
        " ".join(
            str(x)
            for x in (
                rec.get("filename"),
                rec.get("path"),
                rec.get("ocr_text"),
                rec.get("pattern_family"),
                rec.get("pattern_type"),
                rec.get("pattern_subtype"),
            )
            if x
        )
    )


def collect_evidence(
    intent: PatternIntent,
    rec: dict[str, Any],
    *,
    clip_score: float = 0.0,
    rival_clip: dict[str, float] | None = None,
) -> PatternEvidence:
    tm = _texture_map(rec)
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    pf = str(rec.get("pattern_family") or tm.get("pattern_family") or "")
    pt = str(
        rec.get("pattern_type")
        or rec.get("pattern_subtype")
        or tm.get("pattern_type")
        or tm.get("pattern_subtype")
        or ""
    )
    animal = str(tm.get("animal_print_type") or rec.get("animal_print_type") or "")
    blob = _blob(rec)
    motif_n = normalize_turkish(intent.motif)
    family_n = normalize_turkish(intent.family)
    ev = PatternEvidence(
        family=pf,
        motif=pt,
        animal_type=animal,
        clip_score=float(clip_score or 0.0),
        rival_clip={k: float(v) for k, v in (rival_clip or {}).items()},
        animal_score=float(tm.get("animal_score") or 0.0),
    )
    fb = rec.get("feedback_labels") or []
    if isinstance(fb, str):
        fb = [fb]
    ev.feedback_labels = [str(x) for x in fb if x]
    fb_n = normalize_turkish(" ".join(ev.feedback_labels))

    fname_n = normalize_turkish(str(rec.get("filename") or "")).replace("-", " ")
    path_n = normalize_turkish(str(rec.get("path") or "")).replace("-", " ")
    ocr_n = normalize_turkish(str(rec.get("ocr_text") or "")).replace("-", " ")
    needles = {normalize_turkish(intent.raw).replace("-", " "), motif_n}
    if intent.specificity != "leaf":
        needles.add(family_n)
    for alias, cid in _ALIAS_TO_CONCEPT.items():
        meta = _ONTOLOGY.get(cid) or {}
        if cid == intent.concept_id or meta.get("motif") == intent.motif:
            needles.add(alias.replace("-", " "))
    needles = {n for n in needles if n}
    if intent.specificity == "leaf":
        needles.discard(family_n)
        needles -= {
            "floral",
            "cicek",
            "flower",
            "animal",
            "animal_print",
            "animal print",
            "logo",
            "desen",
            "pattern",
        }
    aliases = [a for a in needles if a]
    hay_name = f"{fname_n} {path_n}"
    if any(a and a in hay_name for a in aliases):
        ev.filename_hit = True
        ev.sources.append("filename")
    if any(a and a in ocr_n for a in aliases):
        ev.ocr_hit = True
        ev.sources.append("ocr")

    if intent.family and pf == intent.family:
        ev.family_hit = True
        ev.sources.append("family")
    if (
        intent.family == "geometric"
        and intent.specificity == "parent"
        and pf in ("geometric", "polka_dot")
    ):
        ev.family_hit = True
    if intent.motif == "polka_dot" and pf == "polka_dot":
        ev.family_hit = True
        ev.subtype_hit = True
        ev.sources.append("family")
    if intent.family == "plaid_check" and pf == "plaid_check":
        ev.family_hit = True

    # subtype / animal
    if intent.specificity == "leaf" and intent.family == "animal_print":
        if animal == intent.motif:
            ev.subtype_hit = True
            ev.sources.append("pattern_dna")
        elif animal and animal != intent.motif:
            ev.negative_motifs.append(animal)
    elif intent.specificity == "leaf" and intent.motif:
        rivals = competing_leaf_motifs(intent.family, intent.motif)
        pt_n = normalize_turkish(pt)
        if pt_n == motif_n or (motif_n and motif_n in pt_n):
            ev.subtype_hit = True
            ev.sources.append("pattern_dna")
        elif pt_n and pt_n in rivals and pt_n != motif_n:
            ev.negative_motifs.append(pt_n)
        dna_motif = normalize_turkish(
            str(dna.get("motif_class") or dna.get("motif") or tm.get("motif") or "")
        )
        if motif_n and motif_n in dna_motif:
            ev.subtype_hit = True
            ev.dna_hit = True
            ev.sources.append("pattern_dna")

    if motif_n and motif_n in fb_n:
        ev.feedback_hit = True
        ev.sources.append("feedback")

    if intent.brand:
        brand_n = normalize_turkish(intent.brand).replace("-", " ")
        hay = f"{fname_n} {ocr_n} {path_n}"
        if brand_n and brand_n in hay:
            ev.brand_strong_hit = True
            ev.brand_text_hit = True
            ev.sources.append("filename" if brand_n in fname_n else "ocr")
        if intent.motif == "louis_vuitton" and re.search(r"(?<![a-z0-9])lv(?![a-z0-9])", hay):
            ev.brand_text_hit = True
            ev.sources.append("filename")
        if intent.motif == "gucci" and re.search(r"(?<![a-z0-9])gg(?![a-z0-9])", hay):
            ev.brand_text_hit = True
        if ev.family_hit and ev.clip_score >= CLIP_ONLY_FLOOR:
            ev.sources.append("clip")

    if ev.clip_score >= CLIP_FLOOR:
        ev.sources.append("clip")
    if ev.family_hit and "family" not in ev.sources:
        ev.sources.append("family")
    apply_rival_clip(intent, ev)
    if blob and motif_n and motif_n in blob and not ev.filename_hit:
        pass
    return ev


def apply_rival_clip(intent: PatternIntent, ev: PatternEvidence) -> None:
    rivals = dict(ev.rival_clip or {})
    own = float(rivals.get(intent.motif, ev.clip_score) or 0.0)
    if own > ev.clip_score:
        ev.clip_score = own
    best_rival = 0.0
    negatives: list[str] = list(ev.negative_motifs)
    for name, sc in rivals.items():
        if name == intent.motif:
            continue
        if sc >= CLIP_FLOOR and name not in negatives:
            negatives.append(name)
        if sc > best_rival:
            best_rival = sc
    ev.negative_motifs = negatives
    ev.clip_margin = round(own - best_rival, 4)
    ev.visual_win = bool(
        own >= CLIP_VISUAL_WIN_MIN and own > best_rival + SUBTYPE_MARGIN
    )
    if intent.motif == "snake":
        # Snakeskin is often indexed as leopard; CLIP margin is the live subtype gate.
        ev.visual_win = bool(own >= 0.22 and own > best_rival + 0.02)
    if intent.motif == "leaf":
        # Leaf prints are indexed as floral/flower, not pattern_type=leaf.
        # Bloom rivals otherwise wipe the query. Geometric/paisley CLIP
        # noise must not veto foliage (same trap as empty "eşik üstü").
        bloom_keys = {
            "rose", "daisy", "tulip", "orchid", "peony", "sunflower", "floral",
        }
        bloom = max((float(rivals.get(k) or 0.0) for k in bloom_keys), default=0.0)
        ev.visual_win = bool(own >= CLIP_OWN_MIN and own + 0.04 >= bloom)
    if intent.family == "animal_print" and ev.animal_type:
        if ev.animal_type != intent.motif and ev.animal_type not in ev.negative_motifs:
            ev.negative_motifs.append(ev.animal_type)


def hybrid_score(text_score: float, clip_score: float) -> float:
    """CLIP boosts existing text evidence; cannot dominate filename/OCR."""
    text_score = float(text_score or 0.0)
    clip_score = float(clip_score or 0.0)
    if text_score <= 0.01:
        if clip_score < CLIP_ONLY_FLOOR:
            return 0.0
        return round(min(0.72, 0.48 + (clip_score - CLIP_ONLY_FLOOR) * 1.2), 4)
    contrib = 0.0
    if clip_score >= CLIP_FLOOR:
        contrib = min(CLIP_MAX_CONTRIB, max(0.0, (clip_score - CLIP_FLOOR) / 0.25) * CLIP_MAX_CONTRIB)
    blended = text_score * (1.0 - CLIP_BLEND_WEIGHT) + min(1.0, clip_score / 0.40) * CLIP_BLEND_WEIGHT
    final = max(text_score, text_score * 0.85 + blended * 0.15) + contrib * 0.35
    return round(min(1.0, max(text_score, min(text_score + CLIP_MAX_CONTRIB, final))), 4)


def visual_rank_score(intent: PatternIntent, evidence: PatternEvidence, text_score: float) -> float:
    """Visual first: calibrated CLIP, not a fake 80–98% floor."""
    own = float(evidence.clip_score or 0.0)
    rival = 0.0
    for name, sc in (evidence.rival_clip or {}).items():
        if name != intent.motif:
            rival = max(rival, float(sc or 0.0))
    vis = calibrate_text_clip(own)
    helper = 0.04 if (evidence.filename_hit or evidence.ocr_hit) else 0.0
    subtype = 0.05 if evidence.subtype_hit else 0.0
    penalty = max(0.0, rival - own + 0.01) * 1.6
    if evidence.visual_win:
        return round(min(0.88, max(0.0, vis + helper + subtype - penalty * 0.35)), 4)
    if evidence.subtype_hit:
        return round(min(0.74, vis * 0.65 + 0.18 + helper), 4)
    text_keep = min(0.12, float(text_score or 0.0) * 0.12)
    return round(min(0.52, vis * 0.55 + helper + text_keep), 4)


def subtype_hard_gate(intent: PatternIntent, evidence: PatternEvidence) -> tuple[bool, str]:
    """Family match is not a subtype match.

    UNKNOWN AI labels are not FALSE. Filename/OCR support keeps a candidate
    even when the indexed subtype conflicts; CLIP rival still hard-rejects
    when both sides were actually scored.
    """
    if intent.specificity != "leaf":
        return True, ""
    if intent.motif == "leaf":
        own = float(evidence.clip_score or 0.0)
        bloom = max(
            (
                float((evidence.rival_clip or {}).get(k) or 0.0)
                for k in ("rose", "daisy", "tulip", "orchid", "peony", "sunflower", "floral")
            ),
            default=0.0,
        )
        if evidence.filename_hit or evidence.ocr_hit or evidence.visual_win:
            return True, "leaf_ok"
        if own >= CLIP_OWN_MIN and own + 0.08 >= bloom:
            return True, "leaf_clip"
        rivals = competing_subtype_prompts(intent)
        if not rivals:
            return True, ""
        return False, "leaf_bloom"
    rivals = competing_subtype_prompts(intent)
    if not rivals:
        return True, ""

    text_supported = bool(evidence.filename_hit or evidence.ocr_hit)
    visual_supported = bool(evidence.visual_win)

    if intent.family == "animal_print" and evidence.animal_type:
        if (
            evidence.animal_type != intent.motif
            and not text_supported
            and not visual_supported
        ):
            return False, f"indexed_{evidence.animal_type}"
    if evidence.negative_motifs and intent.motif not in evidence.negative_motifs:
        # indexed rival type already handled; CLIP rivals checked below
        pass

    clip_compared = bool(evidence.rival_clip) or evidence.clip_score >= CLIP_OWN_MIN
    if clip_compared and not evidence.visual_win:
        if evidence.clip_score >= CLIP_FLOOR or any(
            float(v) >= CLIP_FLOOR for k, v in evidence.rival_clip.items() if k != intent.motif
        ):
            top = ""
            best = -1.0
            for k, v in evidence.rival_clip.items():
                if k == intent.motif:
                    continue
                if float(v) > best:
                    best, top = float(v), k
            if not text_supported:
                return False, f"rival_{top or 'subtype'}"

    if intent.family == "animal_print":
        if evidence.subtype_hit and evidence.animal_type == intent.motif:
            if clip_compared and not evidence.visual_win and evidence.clip_score >= CLIP_OWN_MIN:
                if not text_supported:
                    return False, "clip_conflict"
            return True, "index_subtype"
        if evidence.visual_win:
            return True, "clip_win"
        if text_supported:
            return True, "text_supported"
        return False, "no_visual_subtype"

    # Floral / geometric: rival CLIP only when both sides were scored.
    if evidence.rival_clip and not evidence.visual_win:
        own = float(evidence.clip_score or 0.0)
        rival_hi = max(
            (float(v) for k, v in evidence.rival_clip.items() if k != intent.motif),
            default=0.0,
        )
        if own >= CLIP_OWN_MIN or rival_hi >= CLIP_OWN_MIN:
            return False, "rival_subtype"
    return True, "leaf_soft"


def gate_result(
    intent: PatternIntent,
    evidence: PatternEvidence,
    *,
    text_score: float,
) -> tuple[bool, str, float]:
    """Keep? tier, final score. Drops padded family-only hits for leaf queries."""
    ok, reason = subtype_hard_gate(intent, evidence)
    if not ok:
        return False, TIER_SEMANTIC, 0.0

    if intent.specificity == "leaf" and competing_subtype_prompts(intent):
        score = visual_rank_score(intent, evidence, text_score)
    else:
        score = hybrid_score(text_score, evidence.clip_score)

    # Hard blocks
    if (
        intent.specificity == "leaf"
        and intent.family == "geometric"
        and intent.motif not in ("", "geometric", "polka_dot")
        and str(evidence.family or "") in ("polka_dot",)
        and not evidence.subtype_hit
        and not evidence.filename_hit
        and not evidence.ocr_hit
    ):
        return False, TIER_SEMANTIC, 0.0
    if intent.motif == "polka_dot":
        if evidence.animal_score >= 0.55 and not evidence.subtype_hit:
            return False, TIER_SEMANTIC, 0.0
        if evidence.family == "animal_print" and not evidence.subtype_hit:
            return False, TIER_SEMANTIC, 0.0

    if intent.specificity == "leaf" and intent.family == "animal_print":
        if evidence.family_hit and not (
            evidence.subtype_hit
            or evidence.visual_win
            or evidence.filename_hit
            or evidence.ocr_hit
        ):
            return False, TIER_SAME_FAMILY, 0.0

    if intent.specificity == "leaf" and intent.family == "floral":
        if not (
            evidence.subtype_hit
            or evidence.visual_win
            or evidence.feedback_hit
            or (
                not evidence.rival_clip
                and (
                    evidence.filename_hit
                    or evidence.ocr_hit
                    or evidence.clip_score >= CLIP_ONLY_FLOOR
                )
            )
        ):
            return False, TIER_SAME_FAMILY, 0.0

    if intent.family == "geometric" and intent.specificity == "parent":
        if evidence.family in ("floral", "animal_print") and not evidence.filename_hit:
            if evidence.clip_score < 0.28:
                return False, TIER_SEMANTIC, 0.0

    if intent.specificity == "brand":
        if evidence.brand_strong_hit:
            return True, TIER_EXACT, max(score, 0.90)
        if evidence.brand_text_hit:
            return True, TIER_VERY_SIMILAR, max(score, 0.78)
        if evidence.clip_score >= CLIP_ONLY_FLOOR:
            return True, TIER_SEMANTIC, min(0.72, max(score, hybrid_score(0.0, evidence.clip_score)))
        return False, TIER_SEMANTIC, 0.0

    if intent.specificity == "leaf":
        vis = calibrate_text_clip(evidence.clip_score)
        if evidence.visual_win and evidence.subtype_hit:
            shown = max(score, vis)
            if intent.motif == "leaf":
                shown = max(shown, min(0.74, vis + 0.16))
            return True, TIER_EXACT, shown
        if evidence.visual_win:
            shown = max(score, vis)
            if intent.motif == "leaf":
                # CLIP 0.28 → vis 0.48; default slider is 0.60.
                shown = max(shown, min(0.74, vis + 0.16))
            return True, TIER_VERY_SIMILAR, shown
        if evidence.subtype_hit:
            return True, TIER_VERY_SIMILAR, min(0.78, max(score, vis * 0.85))
        if evidence.filename_hit or evidence.ocr_hit:
            if intent.family == "animal_print":
                if evidence.subtype_hit:
                    return True, TIER_VERY_SIMILAR, min(0.78, max(score, 0.70))
                if evidence.negative_motifs:
                    return True, TIER_VARIANT, min(score, 0.55)
                return True, TIER_VARIANT, min(max(score, 0.50), 0.58)
            return True, TIER_VARIANT, min(max(score, vis), 0.70)
        if evidence.clip_score >= CLIP_ONLY_FLOOR:
            shown = min(0.58, vis)
            if intent.motif == "leaf":
                shown = max(0.62, min(0.74, vis + 0.16))
            return True, TIER_SEMANTIC, shown
        return False, TIER_SAME_FAMILY, 0.0

    # parent family query (çiçek, ekose, geometrik, monogram)
    if evidence.filename_hit or evidence.ocr_hit:
        return True, TIER_EXACT if text_score >= 0.90 else TIER_VERY_SIMILAR, max(score, text_score)
    if evidence.family_hit:
        return True, TIER_SAME_FAMILY, min(score, 0.82)
    if evidence.clip_score >= CLIP_ONLY_FLOOR:
        return True, TIER_SEMANTIC, hybrid_score(max(text_score, 0.0), evidence.clip_score)
    if text_score >= 0.70:
        return True, TIER_VARIANT, score
    if text_score >= 0.55:
        return True, TIER_SEMANTIC, score
    return False, TIER_SEMANTIC, 0.0


def cap_confidence_padding(
    intent: PatternIntent,
    rows: list,
    *,
    tier_attr: str = "semantic_tier",
) -> list:
    """Do not pad leaf queries with unbounded CLIP/family hits."""
    primary: list = []
    family: list = []
    semantic: list = []
    for row in rows:
        debug = getattr(row, "debug", None)
        if isinstance(debug, dict):
            tier = str(debug.get(tier_attr) or "")
        elif isinstance(row, dict):
            tier = str(row.get(tier_attr) or row.get("debug", {}).get(tier_attr) or "")
        else:
            tier = ""
        if tier in (TIER_EXACT, TIER_VERY_SIMILAR, TIER_VARIANT):
            primary.append(row)
        elif tier == TIER_SAME_FAMILY:
            family.append(row)
        else:
            semantic.append(row)

    def _clip_key(row) -> float:
        debug = getattr(row, "debug", {}) or {}
        if isinstance(row, dict):
            debug = row.get("debug") or row
        return float((debug or {}).get("clip_score") or getattr(row, "score", 0.0) or 0.0)

    semantic.sort(key=_clip_key, reverse=True)
    sem_cap = LEAF_SEMANTIC_CAP if intent.specificity in ("leaf", "brand") else PARENT_SEMANTIC_CAP
    return primary + family + semantic[:sem_cap]


def feedback_training_hint(rec: dict[str, Any]) -> dict[str, Any]:
    """Ready-to-train structure. Does not start model training."""
    labels = rec.get("feedback_labels") or []
    if isinstance(labels, str):
        labels = [labels]
    return {
        "file_id": rec.get("id") or rec.get("file_id"),
        "labels": list(labels),
        "family_override": rec.get("feedback_family_override") or {},
        "use_for_motif_ranking": True,
        "auto_train": False,
    }


def should_skip_fuzzy(query: str) -> bool:
    norm = normalize_turkish(query or "")
    tokens = set(re.findall(r"[a-z0-9çğıöşü]+", (query or "").lower()))
    tokens.add(norm)
    return bool(tokens & {normalize_turkish(t) for t in PROTECTED_TERMS} or norm in PROTECTED_TERMS)
