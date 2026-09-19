"""Query Attribute Intelligence — soft DNA attribute match (read-only).

Parses scale / density / color / repeat / orientation / style from text and
compares to existing Pattern DNA. Does not write index, DNA, or learning.

Concept match stays more important than attributes (soft ±delta only).
User / learned_exact / manual results are never demoted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.textile_terms import normalize_turkish

# Extensible synonym tables (canonical ← phrases). Independent of ParsedQuery.repeat.
SCALE_SYNONYMS: dict[str, str] = {
    "kucuk": "small",
    "küçük": "small",
    "ufak": "small",
    "mini": "small",
    "minik": "small",
    "petite": "small",
    "fine": "small",
    "small": "small",
    "buyuk": "large",
    "büyük": "large",
    "iri": "large",
    "macro": "large",
    "large": "large",
    "oversized": "large",
}

DENSITY_SYNONYMS: dict[str, str] = {
    "yogun": "dense",
    "yoğun": "dense",
    "sik": "dense",
    "sık": "dense",
    "dense": "dense",
    "packed": "dense",
    "crowded": "dense",
    "seyrek": "sparse",
    "aralikli": "sparse",
    "aralıklı": "sparse",
    "sparse": "sparse",
    "open": "sparse",
    "loose": "sparse",
}

REPEAT_SYNONYMS: dict[str, str] = {
    "allover": "allover",
    "all over": "allover",
    "tekrar": "repeat",
    "tekrarlayan": "repeat",
    "tekrar eden": "repeat",
    "sik tekrar": "dense_repeat",
    "sık tekrar": "dense_repeat",
    "repeat": "repeat",
    "repeated": "repeat",
    "seamless": "allover",
    "metraj": "allover",
    "half drop": "half_drop",
    "halfdrop": "half_drop",
}

ORIENTATION_SYNONYMS: dict[str, str] = {
    "yonlu": "directional",
    "yönlü": "directional",
    "directional": "directional",
    "dikey": "vertical",
    "vertical": "vertical",
    "yatay": "horizontal",
    "horizontal": "horizontal",
    "ince cizgili": "fine_stripe",
    "ince çizgili": "fine_stripe",
    "cizgili": "stripe",
    "çizgili": "stripe",
}

STYLE_SYNONYMS: dict[str, str] = {
    "geometrik": "geometric",
    "geometric": "geometric",
    "organik": "organic",
    "organic": "organic",
    "vintage": "vintage",
    "modern": "modern",
    "minimal": "minimal",
}

# Color tokens beyond single ParsedQuery.color (multi-color queries).
COLOR_TOKEN_SYNONYMS: dict[str, str] = {
    "siyah": "black",
    "black": "black",
    "beyaz": "white",
    "white": "white",
    "krem": "cream",
    "cream": "cream",
    "ivory": "cream",
    "bej": "beige",
    "beige": "beige",
    "kahverengi": "brown",
    "brown": "brown",
    "mavi": "blue",
    "blue": "blue",
    "kirmizi": "red",
    "kırmızı": "red",
    "red": "red",
    "yesil": "green",
    "yeşil": "green",
    "green": "green",
    "gri": "gray",
    "gray": "gray",
    "grey": "gray",
    "gold": "gold",
    "altin": "gold",
    "altın": "gold",
}

_MAX_BONUS = 0.10
_MAX_PENALTY = 0.05
# Attribute must not flip Tiger↔Leopard; keep below typical concept gap.
_ATTR_WEIGHT = {
    "scale": 1.0,
    "density": 1.0,
    "color": 0.85,
    "repeat": 0.7,
    "orientation": 0.55,
    "style": 0.55,
    "pattern_type": 0.6,
}


@dataclass
class QueryAttributes:
    raw: str = ""
    motif: str = ""
    scale: str = ""          # small | large
    density: str = ""        # dense | sparse
    repeat: str = ""         # allover | repeat | half_drop | dense_repeat
    orientation: str = ""
    style: str = ""
    pattern_type: str = ""
    colors: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def has_attributes(self) -> bool:
        return bool(
            self.scale
            or self.density
            or self.repeat
            or self.orientation
            or self.style
            or self.pattern_type
            or self.colors
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hit_phrases(norm: str, mapping: dict[str, str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for phrase in sorted(mapping.keys(), key=len, reverse=True):
        pn = normalize_turkish(phrase)
        if not pn:
            continue
        # word-boundary-ish
        if f" {pn} " in f" {norm} " or norm == pn or norm.startswith(pn + " ") or norm.endswith(" " + pn):
            found.append((phrase, mapping[phrase]))
    # unique by canonical keep first (longest phrase wins via sort)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for phrase, canon in found:
        if canon in seen:
            continue
        seen.add(canon)
        out.append((phrase, canon))
    return out


def attribute_token_keys() -> set[str]:
    """Normalized single-token keys that are attributes, not concept identity."""
    keys: set[str] = set()
    for table in (
        SCALE_SYNONYMS,
        DENSITY_SYNONYMS,
        REPEAT_SYNONYMS,
        ORIENTATION_SYNONYMS,
        STYLE_SYNONYMS,
        COLOR_TOKEN_SYNONYMS,
    ):
        for phrase in table:
            for tok in normalize_turkish(str(phrase)).replace("-", " ").split():
                if len(tok) >= 2:
                    keys.add(tok)
    return keys


# Generic visual CONTEXT markers (not concept identity). Priority: higher wins.
# Applies to ANY concept — never branch on leopard/tiger/etc.
_VISUAL_CONTEXT_MARKERS: tuple[tuple[str, str, int], ...] = (
    # photo / image
    ("fotograf", "photo", 40),
    ("fotografi", "photo", 40),
    ("photo", "photo", 40),
    ("photograph", "photo", 40),
    ("portrait", "photo", 40),
    ("portre", "photo", 40),
    ("selfie", "photo", 40),
    ("resim", "photo", 35),
    ("image", "photo", 35),
    # real-object / animal sense
    ("hayvan", "object", 30),
    ("hayvani", "object", 30),
    ("animal", "object", 30),
    ("gercek", "object", 28),
    ("real", "object", 28),
    # texture / textile material
    ("dokusu", "texture", 22),
    ("dokular", "texture", 22),
    ("dokuyu", "texture", 22),
    ("doku", "texture", 20),
    ("texture", "texture", 20),
    ("textile", "texture", 20),
    # NOTE: bare "tekstil" is NOT a context marker — it collides with customer
    # names like "Ünal Tekstil". Use doku/kumas/fabric for texture intent.
    ("deri", "texture", 18),
    ("derisi", "texture", 18),
    ("skin", "texture", 18),
    ("kumas", "texture", 18),
    ("fabric", "texture", 18),
    # pattern / print (plural + case forms)
    ("desenleri", "pattern", 13),
    ("desenler", "pattern", 12),
    ("deseni", "pattern", 12),
    ("desen", "pattern", 10),
    ("pattern", "pattern", 10),
    ("print", "pattern", 10),
    ("baskilari", "pattern", 11),
    ("baskilar", "pattern", 10),
    ("baski", "pattern", 10),
    ("baskisi", "pattern", 10),
)

_VISUAL_CONTEXT_TOKEN_SET: frozenset[str] = frozenset(
    m for m, _c, _p in _VISUAL_CONTEXT_MARKERS
)

_CONTEXT_TO_VISUAL_TYPE = {
    "none": "general",
    "pattern": "pattern",
    "texture": "texture",
    "object": "object",
    "photo": "photo",
}


def visual_context_token_keys() -> set[str]:
    """Normalized tokens that mark visual CONTEXT, not CONCEPT identity."""
    return set(_VISUAL_CONTEXT_TOKEN_SET)


def extract_query_visual_context(query: str) -> dict[str, Any]:
    """Split CONCEPT vs CONTEXT vs VISUAL TYPE (generic, concept-agnostic).

    Bare concept → context=none (no forced textile/object preference).
    Explicit markers only: deseni/dokusu/hayvanı/fotoğrafı/…
    """
    raw = str(query or "").strip()
    out: dict[str, Any] = {
        "raw": raw,
        "concept_core": "",
        "context": "none",
        "visual_type": "general",
        "explicit": False,
        "marker": "",
    }
    if not raw:
        return out

    norm = normalize_turkish(raw).replace("-", " ")
    tokens = [t for t in norm.split() if t]
    # Exact token → context (no startswith — avoids doku⊂doktora false hits).
    marker_map = {m: (c, p) for m, c, p in _VISUAL_CONTEXT_MARKERS}
    best_ctx = "none"
    best_pri = -1
    best_marker = ""
    for tok in tokens:
        hit = marker_map.get(tok)
        if hit is None:
            continue
        ctx, pri = hit
        if pri > best_pri:
            best_pri = pri
            best_ctx = ctx
            best_marker = tok

    attr = attribute_token_keys()
    ctx_keys = visual_context_token_keys()
    kept = [t for t in tokens if t not in attr and t not in ctx_keys]

    core = " ".join(kept).strip()
    if not core:
        try:
            attrs = extract_query_attributes(raw)
            core = str(attrs.motif or "").strip()
        except Exception:
            core = ""
    if not core:
        core = raw

    out["concept_core"] = core
    out["context"] = best_ctx
    out["visual_type"] = _CONTEXT_TO_VISUAL_TYPE.get(best_ctx, "general")
    out["explicit"] = best_ctx != "none"
    out["marker"] = best_marker
    return out


def concept_core_text(query: str) -> str:
    """Motif / remaining tokens after attribute + context strip — concept identity only."""
    raw = (query or "").strip()
    if not raw:
        return ""
    # Prefer visual-context strip so "leopard dokusu" → "leopard".
    try:
        vctx = extract_query_visual_context(raw)
        if vctx.get("explicit") and vctx.get("concept_core"):
            return str(vctx["concept_core"]).strip()
    except Exception:
        pass
    attrs = extract_query_attributes(raw)
    if attrs.motif:
        return str(attrs.motif).strip()
    norm = normalize_turkish(raw).replace("-", " ")
    attr = attribute_token_keys()
    ctx = visual_context_token_keys()
    kept = [t for t in norm.split() if t and t not in attr and t not in ctx]
    return " ".join(kept).strip()


def extract_query_attributes(query: str) -> QueryAttributes:
    """Parse visual/textile attributes; reuses NL motif/color when present."""
    raw = (query or "").strip()
    attrs = QueryAttributes(raw=raw)
    if not raw:
        return attrs
    norm = normalize_turkish(raw)

    try:
        from core.natural_language_query import parse_natural_query

        parsed = parse_natural_query(raw)
        if parsed.motif:
            attrs.motif = str(parsed.motif)
            attrs.sources["motif"] = "natural_language_query"
        # NL packs scale+density into repeat — split below; keep color seed.
        if parsed.color:
            attrs.colors.append(_color_canon(parsed.color))
            attrs.sources["color"] = "natural_language_query"
        if parsed.style and not attrs.style:
            attrs.style = str(parsed.style)
        if parsed.texture == "geometric" or parsed.motif == "geometric":
            attrs.style = attrs.style or "geometric"
            attrs.pattern_type = attrs.pattern_type or "geometric"
    except Exception:
        pass

    for phrase, canon in _hit_phrases(norm, SCALE_SYNONYMS):
        attrs.scale = canon
        attrs.sources["scale"] = phrase
        break
    for phrase, canon in _hit_phrases(norm, DENSITY_SYNONYMS):
        attrs.density = canon
        attrs.sources["density"] = phrase
        break
    for phrase, canon in _hit_phrases(norm, REPEAT_SYNONYMS):
        # Don't let "sık" alone steal density — dense_repeat is explicit.
        if canon == "dense_repeat":
            attrs.density = attrs.density or "dense"
            attrs.repeat = "repeat"
            attrs.sources["repeat"] = phrase
        else:
            attrs.repeat = canon
            attrs.sources["repeat"] = phrase
        break
    for phrase, canon in _hit_phrases(norm, ORIENTATION_SYNONYMS):
        attrs.orientation = canon
        attrs.sources["orientation"] = phrase
        if canon in {"stripe", "fine_stripe"}:
            attrs.pattern_type = attrs.pattern_type or "stripe"
        break
    for phrase, canon in _hit_phrases(norm, STYLE_SYNONYMS):
        attrs.style = canon
        attrs.sources["style"] = phrase
        break

    # Multi-color: collect all color tokens present.
    colors: list[str] = list(attrs.colors)
    for phrase, canon in _hit_phrases(norm, COLOR_TOKEN_SYNONYMS):
        c = _color_canon(canon)
        if c and c not in colors:
            colors.append(c)
            attrs.sources.setdefault("color", phrase)
    attrs.colors = colors

    # NL repeat may be small/large/dense — promote if we missed synonyms.
    try:
        from core.natural_language_query import parse_natural_query

        pr = parse_natural_query(raw).repeat
        if pr == "small" and not attrs.scale:
            attrs.scale = "small"
            attrs.sources.setdefault("scale", "nl_repeat")
        elif pr == "large" and not attrs.scale:
            attrs.scale = "large"
            attrs.sources.setdefault("scale", "nl_repeat")
        elif pr == "dense" and not attrs.density:
            attrs.density = "dense"
            attrs.sources.setdefault("density", "nl_repeat")
        elif pr in {"allover", "repeat"} and not attrs.repeat:
            attrs.repeat = pr
            attrs.sources.setdefault("repeat", "nl_repeat")
    except Exception:
        pass

    return attrs


def _color_canon(value: str) -> str:
    v = normalize_turkish(str(value or ""))
    # Map color_family ids → simple tokens
    if v in {"black_white", "black"}:
        return "black" if "white" not in v or v == "black" else "black"
    if "cream" in v or v == "krem":
        return "cream"
    if "beige" in v or v == "bej":
        return "beige"
    if "brown" in v or "tan" in v:
        return "brown"
    if "blue" in v or "navy" in v:
        return "blue"
    if "red" in v or "pink" in v:
        return "red"
    if "green" in v:
        return "green"
    if "gray" in v or "grey" in v or "grayscale" in v:
        return "gray"
    if "gold" in v:
        return "gold"
    if "white" in v:
        return "white"
    if "black" in v:
        return "black"
    return COLOR_TOKEN_SYNONYMS.get(v, v)


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


def _dna(result: Any) -> dict[str, Any]:
    tm = _tm(result)
    raw = tm.get("pattern_dna")
    return raw if isinstance(raw, dict) else {}


def _band_norm(value: str) -> str:
    v = normalize_turkish(str(value or ""))
    if v in {"high", "yuksek", "yüksek", "large", "buyuk", "büyük", "macro", "dense", "sik", "sık"}:
        return "high"
    if v in {"low", "dusuk", "düşük", "small", "kucuk", "küçük", "fine", "mini", "sparse", "seyrek"}:
        return "low"
    if v in {"medium", "med", "orta"}:
        return "medium"
    return v


def _user_protected(result: Any) -> bool:
    dbg = getattr(result, "debug", None) or {}
    if not isinstance(dbg, dict):
        return False
    if getattr(result, "is_self_match", False):
        return True
    if dbg.get("protected_exact") or dbg.get("user_taught_positive"):
        return True
    if dbg.get("learned_concept_exact"):
        return True
    if dbg.get("user_labeled") or dbg.get("category_source") == "manual_user":
        return True
    return False


def _concept_aligned(result: Any, attrs: QueryAttributes) -> bool:
    """True when result already looks like the queried motif/family."""
    motif = normalize_turkish(attrs.motif or "")
    if not motif:
        return True  # attribute-only tilt allowed
    dbg = getattr(result, "debug", None) or {}
    if isinstance(dbg, dict):
        if dbg.get("learned_concept") or dbg.get("learned_concept_exact"):
            label = normalize_turkish(
                str(dbg.get("learned_canonical") or dbg.get("learned_concept_label") or "")
            )
            if motif and label and (motif in label or label in motif):
                return True
            # TR↔EN leaf (Tiger↔kaplan) — never sibling bags (Tiger↛Leopard).
            if motif and label:
                try:
                    from core.concept_query_normalize import leaf_translation_keys

                    if leaf_translation_keys(motif) & leaf_translation_keys(label):
                        return True
                except Exception:
                    pass
            # parent animal_print / floral still aligned
            if motif in {"animal_print", "floral"} and dbg.get("learned_concept"):
                return True
    fam = normalize_turkish(
        str(
            getattr(result, "pattern_family", "")
            or (dbg.get("pattern_family") if isinstance(dbg, dict) else "")
            or _tm(result).get("pattern_family")
            or ""
        )
    )
    sub = normalize_turkish(
        str(
            getattr(result, "animal_print_type", "")
            or _tm(result).get("animal_print_type")
            or _tm(result).get("pattern_subtype")
            or ""
        )
    )
    dna = _dna(result)
    motif_dna = normalize_turkish(str(dna.get("motif") or dna.get("motif_class") or ""))
    if motif in {"tiger", "kaplan"} and ("tiger" in sub or "kaplan" in sub or "tiger" in motif_dna):
        return True
    if motif in {"leopard", "leopar"} and ("leopard" in sub or "leopar" in sub or "leopard" in motif_dna):
        return True
    if motif in {"zebra"} and "zebra" in (sub + motif_dna):
        return True
    if motif in {"snake", "yilan"} and ("snake" in sub or "yilan" in sub or "snake" in motif_dna):
        return True
    if motif in {"floral", "flower", "cicek", "rose", "gul", "tulip", "lale", "daisy", "papatya"}:
        if fam == "floral" or motif in motif_dna or motif in sub:
            return True
    if motif in {"animal_print", "animal", "hayvan"} and fam == "animal_print":
        return True
    if motif in {"geometric", "geometrik"} and (
        fam in {"geometric", "stripe", "plaid_check"} or "geometric" in motif_dna
    ):
        return True
    if motif in {"stripe", "cizgili"} and (fam == "stripe" or "stripe" in motif_dna):
        return True
    # Leaf animal types must not align via parent family alone (Tiger ≠ Leopard).
    return False


def _match_scale(want: str, dna: dict[str, Any], tm: dict[str, Any]) -> float:
    """1 match, 0 unknown, -1 mismatch."""
    if not want:
        return 0.0
    band = _band_norm(str(dna.get("scale") or ""))
    scale_score = float(tm.get("scale_pattern_score") or 0)
    if want == "small":
        if band == "low" or band in {"small", "fine", "mini"}:
            return 1.0
        if band == "high" or band in {"large", "macro"}:
            return -1.0
        if scale_score and scale_score < 0.28:
            return 0.7
        if scale_score and scale_score >= 0.50:
            return -0.8
        return 0.0
    if want == "large":
        if band == "high" or band in {"large", "macro"}:
            return 1.0
        if band == "low" or band in {"small", "fine"}:
            return -1.0
        if scale_score and scale_score >= 0.48:
            return 0.7
        if scale_score and scale_score < 0.22:
            return -0.8
        return 0.0
    return 0.0


def _match_density(want: str, dna: dict[str, Any], tm: dict[str, Any]) -> float:
    if not want:
        return 0.0
    band = _band_norm(str(dna.get("density") or ""))
    rd = float(tm.get("repeat_density") or 0)
    if want == "dense":
        if band == "high":
            return 1.0
        if band == "low":
            return -1.0
        if rd >= 0.45:
            return 0.75
        if rd and rd < 0.15:
            return -0.75
        return 0.0
    if want == "sparse":
        if band == "low":
            return 1.0
        if band == "high":
            return -1.0
        if rd and rd < 0.18:
            return 0.75
        if rd >= 0.45:
            return -0.75
        return 0.0
    return 0.0


def _match_repeat(want: str, dna: dict[str, Any], tm: dict[str, Any]) -> float:
    if not want:
        return 0.0
    rep = normalize_turkish(
        str(dna.get("repeat_type") or dna.get("repeat_class") or dna.get("repeat") or "")
    )
    rd = float(tm.get("repeat_density") or 0)
    if want in {"allover", "repeat", "dense_repeat"}:
        if "all over" in rep or "allover" in rep or rep == "all over":
            return 1.0
        if "half" in rep and want == "allover":
            return -0.4
        if rd >= 0.30:
            return 0.6
        return 0.0
    if want == "half_drop":
        if "half" in rep:
            return 1.0
        return 0.0
    return 0.0


def _match_orientation(want: str, dna: dict[str, Any], tm: dict[str, Any]) -> float:
    if not want:
        return 0.0
    ori = normalize_turkish(str(dna.get("orientation") or ""))
    geo = normalize_turkish(str(dna.get("geometry") or ""))
    stripe = float(tm.get("stripe_score") or 0)
    if want in {"directional", "vertical", "horizontal", "stripe", "fine_stripe"}:
        if "directional" in ori or stripe >= 0.42:
            return 0.9
        if want in {"stripe", "fine_stripe"} and ("stripe" in geo or stripe >= 0.35):
            return 0.85
        if ori == "random" and stripe < 0.2:
            return -0.6
        return 0.0
    return 0.0


def _match_style(want: str, dna: dict[str, Any], tm: dict[str, Any]) -> float:
    if not want:
        return 0.0
    style = normalize_turkish(str(dna.get("style") or dna.get("designer_style") or ""))
    geo = normalize_turkish(str(dna.get("geometry") or ""))
    fam = normalize_turkish(str(tm.get("pattern_family") or ""))
    if want == "geometric":
        if "geometric" in geo or fam in {"geometric", "plaid_check", "stripe"}:
            return 1.0
        if "organic" in geo:
            return -0.5
        return 0.0
    if want == "organic":
        if "organic" in geo or float(tm.get("organic_blob_score") or 0) >= 0.35:
            return 0.9
        return 0.0
    if want and want in style:
        return 0.8
    return 0.0


def _match_colors(want: list[str], dna: dict[str, Any], tm: dict[str, Any], result: Any) -> float:
    if not want:
        return 0.0
    # Stage 6: semantic names from RGB clusters (USER > AI). Unknown → 0 (no penalty).
    have_named: list[str] = []
    to_en = lambda x: normalize_turkish(str(x or ""))  # noqa: E731
    try:
        from core.color_evidence import (
            effective_detected_colors,
            ensure_color_evidence,
            to_en_color,
        )

        to_en = to_en_color
        tm2 = ensure_color_evidence(tm)
        have_named = effective_detected_colors(tm2)
    except Exception:
        have_named = []
        tm2 = tm

    blob_parts = [
        str(dna.get("color_family") or ""),
        " ".join(str(x) for x in (dna.get("dominant_colors") or [])),
        str(tm2.get("color_family") or ""),
        str(getattr(result, "color_family", "") or ""),
        " ".join(str(x) for x in (tm2.get("dominant_palette") or [])),
        " ".join(str(x) for x in have_named),
    ]
    ev = tm2.get("color_evidence") if isinstance(tm2.get("color_evidence"), dict) else {}
    if ev.get("detected_colors"):
        blob_parts.append(" ".join(str(x) for x in ev["detected_colors"]))
    blob = normalize_turkish(" ".join(blob_parts))
    if not blob.strip() and not have_named:
        return 0.0
    hits = 0
    misses = 0
    have_set = {normalize_turkish(to_en(x) or x) for x in have_named}
    for c in want:
        cn = _color_canon(c)
        aliases = {
            "black": ("black", "siyah", "black_white"),
            "white": ("white", "beyaz", "black_white"),
            "cream": ("cream", "krem", "ivory"),
            "beige": ("beige", "bej"),
            "brown": ("brown", "tan", "kahve", "dark_brown"),
            "blue": ("blue", "navy", "mavi", "dark_blue"),
            "red": ("red", "pink", "kirmizi", "dark_red", "burgundy"),
            "green": ("green", "yesil", "dark_green"),
            "gray": ("gray", "grey", "gri", "grayscale", "light_gray"),
            "gold": ("gold", "altin"),
            "navy": ("navy", "lacivert", "blue"),
            "purple": ("purple", "mor"),
            "turquoise": ("turquoise", "turkuaz"),
            "pink": ("pink", "pembe"),
            "orange": ("orange", "turuncu"),
            "yellow": ("yellow", "sari"),
            "burgundy": ("burgundy", "bordo"),
        }.get(cn, (cn,))
        if cn in have_set or any(a in have_set for a in aliases if a):
            hits += 1
        elif any(a in blob for a in aliases if a):
            hits += 1
        else:
            misses += 1
    if hits == 0 and misses:
        return -0.7
    if hits and not misses:
        return 1.0
    if hits:
        return 0.45
    return 0.0


def score_attributes_against_dna(
    attrs: QueryAttributes,
    result: Any,
) -> dict[str, Any]:
    """Return match vector and soft delta (not a hard filter)."""
    empty = {
        "delta": 0.0,
        "matches": {},
        "aligned": False,
        "applied": False,
        "reason": "no_attrs",
    }
    if not attrs.has_attributes:
        return empty
    tm = _tm(result)
    dna = _dna(result)
    aligned = _concept_aligned(result, attrs)
    parts: dict[str, float] = {}
    if attrs.scale:
        parts["scale"] = _match_scale(attrs.scale, dna, tm)
    if attrs.density:
        parts["density"] = _match_density(attrs.density, dna, tm)
    if attrs.repeat:
        parts["repeat"] = _match_repeat(attrs.repeat, dna, tm)
    if attrs.orientation:
        parts["orientation"] = _match_orientation(attrs.orientation, dna, tm)
    if attrs.style:
        parts["style"] = _match_style(attrs.style, dna, tm)
    if attrs.colors:
        parts["color"] = _match_colors(attrs.colors, dna, tm, result)

    if not parts:
        return empty

    weighted = 0.0
    wsum = 0.0
    for key, val in parts.items():
        w = float(_ATTR_WEIGHT.get(key, 0.5))
        weighted += w * val
        wsum += w
    mean = weighted / max(wsum, 1e-6)

    # Soft delta; shrink when concept not aligned so Leopard+scale can't beat Tiger.
    if aligned:
        if mean >= 0:
            delta = min(_MAX_BONUS, 0.045 + 0.055 * mean)
        else:
            delta = max(-_MAX_PENALTY, -0.035 + 0.045 * mean)
    else:
        # Wrong concept: attributes never promote a rival sibling.
        delta = 0.0

    return {
        "delta": round(delta, 4),
        "matches": {k: round(v, 3) for k, v in parts.items()},
        "aligned": aligned,
        "applied": True,
        "mean": round(mean, 4),
        "reason": "aligned" if aligned else "weak_concept",
    }


def adjust_score_for_attributes(
    score: float,
    result: Any,
    attrs: QueryAttributes,
) -> tuple[float, dict[str, Any]]:
    if not attrs.has_attributes:
        return float(score), {"applied": False, "reason": "no_attrs"}
    if _user_protected(result):
        return float(score), {"applied": False, "skipped": "user_protected"}
    meta = score_attributes_against_dna(attrs, result)
    if not meta.get("applied"):
        return float(score), meta
    new = max(0.0, min(1.0, float(score) + float(meta.get("delta") or 0)))
    meta["before"] = round(float(score), 4)
    meta["after"] = round(new, 4)
    return new, meta


def apply_query_attribute_intel(
    results: list[Any],
    query_text: str,
    *,
    attrs: QueryAttributes | None = None,
) -> list[Any]:
    """Soft re-score by Pattern DNA attributes. Never hard-filters."""
    if not results:
        return results
    attrs = attrs or extract_query_attributes(query_text)
    if not attrs.has_attributes:
        return results
    for rec in results:
        try:
            old = float(getattr(rec, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        new, meta = adjust_score_for_attributes(old, rec, attrs)
        if not meta.get("applied") and not meta.get("skipped"):
            continue
        if meta.get("applied"):
            rec.score = new
            if hasattr(rec, "score_percent"):
                rec.score_percent = round(new * 100, 1)
        dbg = dict(getattr(rec, "debug", None) or {})
        dbg["query_attribute_intel"] = {
            **meta,
            "attrs": {
                "scale": attrs.scale,
                "density": attrs.density,
                "repeat": attrs.repeat,
                "colors": list(attrs.colors),
                "orientation": attrs.orientation,
                "style": attrs.style,
                "motif": attrs.motif,
            },
        }
        rec.debug = dbg
    return results
