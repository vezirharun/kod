"""Doğal dil metin arama — sorguyu marka/renk/motif/repeat/stil/doku bileşenlerine ayırır."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from core.brand_aliases import BRAND_ALIASES, brand_match_score, expand_brand_terms
from core.textile_terms import normalize_turkish, tokenize

# Renk ifadesi → color_family (texture_profile ile uyumlu)
COLOR_PHRASES: dict[str, str] = {
    "kahverengi": "brown_tan",
    "brown": "brown_tan",
    "bronz": "brown_tan",
    "bej": "beige",
    "beige": "beige",
    "tan": "brown_tan",
    "krem": "cream",
    "cream": "cream",
    "ivory": "cream",
    "camel": "camel",
    "deve": "camel",
    "bordo": "burgundy",
    "burgundy": "burgundy",
    "wine": "burgundy",
    "sarap": "burgundy",
    "haki": "khaki",
    "khaki": "khaki",
    "olive": "khaki",
    "zeytin": "khaki",
    "siyah": "black_white",
    "black": "black_white",
    "beyaz": "black_white",
    "white": "black_white",
    "mavi": "blue",
    "blue": "blue",
    "lacivert": "blue",
    "navy": "blue",
    "kirmizi": "red_pink",
    "kırmızı": "red_pink",
    "red": "red_pink",
    "pembe": "red_pink",
    "pink": "red_pink",
    "yesil": "green",
    "yeşil": "green",
    "green": "green",
    "sari": "yellow",
    "sarı": "yellow",
    "yellow": "yellow",
    "altin": "gold",
    "altın": "gold",
    "gold": "gold",
    "golden": "gold",
    "turuncu": "orange",
    "orange": "orange",
    "gri": "grayscale",
    "gray": "grayscale",
    "grey": "grayscale",
    "neon": "neon_multicolor",
    "renkli": "neon_multicolor",
    "multicolor": "neon_multicolor",
}

# Motif ifadesi → canonical motif / pattern hint
MOTIF_PHRASES: dict[str, str] = {
    "leopar": "leopard",
    "leopard": "leopard",
    "leo": "leopard",
    "zebra": "zebra",
    "snake": "snake",
    "yilan": "snake",
    "yılan": "snake",
    "tiger": "tiger",
    "kaplan": "tiger",
    "cicek": "floral",
    "çiçek": "floral",
    "cicekli": "floral",
    "çiçekli": "floral",
    "floral": "floral",
    "flower": "floral",
    "gul": "rose",
    "gül": "rose",
    "rose": "rose",
    "papatya": "daisy",
    "papatyali": "daisy",
    "papatyalı": "daisy",
    "daisy": "daisy",
    "lale": "tulip",
    "tulip": "tulip",
    "orkide": "orchid",
    "orchid": "orchid",
    "sakayik": "peony",
    "şakayık": "peony",
    "peony": "peony",
    "geometrik": "geometric",
    "geometric": "geometric",
    "kareli": "plaid",
    "kare": "square",
    "logo": "logo",
    "monogram": "logo",
    "arma": "logo",
    "amblem": "logo",
    "ekose": "plaid",
    "plaid": "plaid",
    "check": "plaid",
    "cizgili": "stripe",
    "çizgili": "stripe",
    "stripe": "stripe",
    "striped": "stripe",
    "paisley": "paisley",
    "sal": "paisley",
    "şal": "paisley",
    "puantiye": "polka_dot",
    "puantiyeli": "polka_dot",
    "noktali": "polka_dot",
    "noktalı": "polka_dot",
    "polka": "polka_dot",
    "polka dot": "polka_dot",
    "dot": "polka_dot",
    "hayvan": "animal_print",
    "animal": "animal_print",
    "leopard print": "leopard",
    "animal print": "animal_print",
}

# Repeat / ölçek
REPEAT_PHRASES: dict[str, str] = {
    "kucuk": "small",
    "küçük": "small",
    "mini": "small",
    "minik": "small",
    "small": "small",
    "buyuk": "large",
    "büyük": "large",
    "large": "large",
    "macro": "large",
    "metraj": "allover",
    "allover": "allover",
    "tekrar": "repeat",
    "repeat": "repeat",
    "repeated": "repeat",
    "seamless": "allover",
    "dense": "dense",
    "sik": "dense",
    "sık": "dense",
}

# Stil
STYLE_PHRASES: dict[str, str] = {
    "yazili": "typography",
    "yazı": "typography",
    "yazi": "typography",
    "text": "typography",
    "lettering": "typography",
    "barok": "baroque",
    "baroque": "baroque",
    "luxury": "luxury",
    "luks": "luxury",
    "lüks": "luxury",
    "vintage": "vintage",
    "modern": "modern",
    "minimal": "minimal",
    "cocuk": "kids",
    "çocuk": "kids",
    "cocuk deseni": "kids",
    "çocuk deseni": "kids",
}

# Doku
TEXTURE_PHRASES: dict[str, str] = {
    "jakar": "jacquard",
    "jacquard": "jacquard",
    "dantel": "lace",
    "lace": "lace",
    "orgu": "knit",
    "örgü": "knit",
    "knit": "knit",
    "denim": "denim",
    "kot": "denim",
    "suede": "suede",
    "velvet": "velvet",
    "kadife": "velvet",
    "mermer": "marble",
    "marble": "marble",
}

GENERIC_SKIP = frozenset(
    {"desen", "pattern", "print", "motif", "tekstil", "kumas", "kumaş", "fabric"}
)

_BRAND_KEYS = sorted(BRAND_ALIASES.keys(), key=len, reverse=True)
_PHRASE_GROUPS: tuple[tuple[str, dict[str, str]], ...] = (
    ("brand", {}),  # brand handled separately
    ("color", COLOR_PHRASES),
    ("motif", MOTIF_PHRASES),
    ("repeat", REPEAT_PHRASES),
    ("style", STYLE_PHRASES),
    ("texture", TEXTURE_PHRASES),
)


@dataclass
class ParsedQuery:
    raw: str = ""
    brand: str = ""
    brand_tokens: list[str] = field(default_factory=list)
    color: str = ""
    color_tokens: list[str] = field(default_factory=list)
    motif: str = ""
    motif_tokens: list[str] = field(default_factory=list)
    repeat: str = ""
    repeat_tokens: list[str] = field(default_factory=list)
    style: str = ""
    style_tokens: list[str] = field(default_factory=list)
    texture: str = ""
    texture_tokens: list[str] = field(default_factory=list)
    residual: str = ""
    components: dict[str, str] = field(default_factory=dict)
    # NL 2.0 optional fields (filled by bridge / SI — not by phrase extract alone)
    customer: str = ""
    scale: str = ""
    intent: str = ""
    search_text: str = ""

    @property
    def has_components(self) -> bool:
        return bool(self.components) or bool(self.customer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "brand": self.brand,
            "color": self.color,
            "motif": self.motif,
            "repeat": self.repeat,
            "style": self.style,
            "texture": self.texture,
            "residual": self.residual,
            "components": dict(self.components),
            "customer": self.customer,
            "scale": self.scale,
            "intent": self.intent,
            "search_text": self.search_text or self.raw,
        }


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    esc = re.escape(normalize_turkish(phrase))
    return re.compile(rf"(?<![a-z0-9]){esc}(?![a-z0-9])")


def _extract_phrases(text: str, mapping: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    remaining = text
    found: list[tuple[str, str]] = []
    for phrase in sorted(mapping.keys(), key=len, reverse=True):
        pat = _phrase_pattern(phrase)
        if pat.search(remaining):
            found.append((phrase, mapping[phrase]))
            remaining = pat.sub(" ", remaining)
    remaining = re.sub(r"\s+", " ", remaining).strip()
    return remaining, found


def _extract_brand(text: str) -> tuple[str, str, list[str]]:
    norm = normalize_turkish(text)
    for key in _BRAND_KEYS:
        pat = _phrase_pattern(key)
        if pat.search(norm):
            brand = BRAND_ALIASES[key]
            tokens = expand_brand_terms(key) + expand_brand_terms(brand)
            cleaned = pat.sub(" ", norm)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned, brand, list(dict.fromkeys(tokens))
    return norm, "", []


@lru_cache(maxsize=256)
def parse_natural_query(query: str) -> ParsedQuery:
    raw = (query or "").strip()
    if not raw:
        return ParsedQuery()
    text = normalize_turkish(raw)
    parsed = ParsedQuery(raw=raw)

    text, brand, brand_tokens = _extract_brand(text)
    if brand:
        parsed.brand = brand
        parsed.brand_tokens = brand_tokens
        parsed.components["brand"] = brand

    for field_name, mapping in _PHRASE_GROUPS[1:]:
        text, hits = _extract_phrases(text, mapping)
        if not hits:
            continue
        canonical = hits[0][1]
        tokens = [h[0] for h in hits] + [canonical]
        setattr(parsed, field_name, canonical)
        setattr(parsed, f"{field_name}_tokens", list(dict.fromkeys(tokens)))
        parsed.components[field_name] = canonical

    # Kalan tokenler
    residual_tokens = [
        t for t in tokenize(text) if t not in GENERIC_SKIP and len(t) >= 2
    ]
    parsed.residual = " ".join(residual_tokens)

    # Tek kelime motif/renk fallback (ör. "leopard" tek başına)
    if not parsed.motif and parsed.residual:
        for tok in residual_tokens:
            if tok in MOTIF_PHRASES:
                parsed.motif = MOTIF_PHRASES[tok]
                parsed.motif_tokens = [tok, parsed.motif]
                parsed.components["motif"] = parsed.motif
                residual_tokens = [t for t in residual_tokens if t != tok]
                break
    if not parsed.color and parsed.residual:
        for tok in residual_tokens:
            if tok in COLOR_PHRASES:
                parsed.color = COLOR_PHRASES[tok]
                parsed.color_tokens = [tok, parsed.color]
                parsed.components["color"] = parsed.color
                residual_tokens = [t for t in residual_tokens if t != tok]
                break
    parsed.residual = " ".join(residual_tokens)
    return parsed


def expand_parsed_terms(parsed: ParsedQuery) -> list[str]:
    """FTS aday genişletmesi."""
    out: list[str] = []
    if parsed.brand:
        out.extend(parsed.brand_tokens or expand_brand_terms(parsed.brand))
        out.append(parsed.brand)
    if parsed.color:
        out.extend(parsed.color_tokens)
        out.append(parsed.color)
        for k, v in COLOR_PHRASES.items():
            if v == parsed.color:
                out.append(k)
    if parsed.motif:
        out.extend(parsed.motif_tokens)
        out.append(parsed.motif)
    if parsed.repeat:
        out.extend(parsed.repeat_tokens)
        out.append(parsed.repeat)
    if parsed.style:
        out.extend(parsed.style_tokens)
        out.append(parsed.style)
    if parsed.texture:
        out.extend(parsed.texture_tokens)
        out.append(parsed.texture)
    if parsed.residual:
        out.append(parsed.residual)
    return list(dict.fromkeys(normalize_turkish(t) for t in out if t))


def _norm_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    an, bn = normalize_turkish(a), normalize_turkish(b)
    return an == bn or an in bn or bn in an


def _color_match(query_color: str, cand_color: str, blob_n: str, color_index: dict | None = None) -> float:
    if not query_color:
        return 0.0
    from core.color_index import color_index_match_score

    ci_score = color_index_match_score(query_color, color_index)
    if ci_score > 0:
        return ci_score
    if _norm_match(query_color, cand_color):
        return 0.88
    for tok in COLOR_PHRASES:
        if COLOR_PHRASES[tok] == query_color and normalize_turkish(tok) in blob_n:
            return 0.72
    return 0.0


def _motif_match(
    query_motif: str,
    *,
    pf: str,
    pt: str,
    dna: dict,
    semantic: dict,
    blob_n: str,
) -> float:
    if not query_motif:
        return 0.0
    motif_class = str(dna.get("motif_class") or dna.get("motif") or "")
    sem_motif = str(semantic.get("motif") or semantic.get("motifs") or "")
    checks = [
        (pf, 0.86),
        (pt, 0.84),
        (motif_class, 0.82),
        (sem_motif, 0.80),
    ]
    motif_map = {
        "leopard": ("leopard", "leo"),
        "floral": ("floral", "flower", "cicek", "botanical"),
        "rose": ("rose", "gul"),
        "daisy": ("daisy", "papatya"),
        "tulip": ("tulip", "lale"),
        "orchid": ("orchid", "orkide"),
        "peony": ("peony", "sakayik"),
        "logo": ("monogram_logo", "logo", "monogram"),
        "plaid": ("plaid_check", "plaid", "check", "ekose"),
        "stripe": ("stripe", "striped", "cizgi"),
        "zebra": ("zebra",),
        "snake": ("snake", "python", "yilan"),
        "tiger": ("tiger", "kaplan"),
        "animal_print": ("animal", "animal_print"),
        "paisley": ("paisley", "scarf"),
        "polka_dot": ("polka", "polka_dot", "puantiye", "noktali"),
        "geometric": ("geometric", "geometrik"),
    }
    aliases = motif_map.get(query_motif, (query_motif,))
    for val, base in checks:
        vn = normalize_turkish(val)
        if any(normalize_turkish(a) in vn or vn in normalize_turkish(a) for a in aliases if a):
            return base
    if any(normalize_turkish(a) in blob_n for a in aliases):
        return 0.70
    return 0.0


def _repeat_match(query_repeat: str, dna: dict, texture_map: dict) -> float:
    if not query_repeat:
        return 0.0
    scale = normalize_turkish(str(dna.get("scale") or ""))
    repeat_class = normalize_turkish(
        str(dna.get("repeat_class") or dna.get("repeat_type") or "")
    )
    density = float(texture_map.get("repeat_density", 0) or 0)
    scale_score = float(texture_map.get("scale_pattern_score", 0) or 0)
    if query_repeat == "small":
        if scale in ("small", "fine", "mini", "kucuk"):
            return 0.84
        if scale_score < 0.35 and density >= 0.2:
            return 0.72
    if query_repeat == "large":
        if scale in ("large", "macro", "buyuk"):
            return 0.84
        if scale_score >= 0.45:
            return 0.76
    if query_repeat in ("allover", "repeat", "dense"):
        if query_repeat in repeat_class or repeat_class in query_repeat:
            return 0.80
        if density >= 0.25:
            return 0.70
    return 0.0


def _style_match(query_style: str, dna: dict, pf: str, blob_n: str, texture_map: dict | None = None) -> float:
    if not query_style:
        return 0.0
    if query_style == "kids":
        from core.auto_tags import flatten_auto_tags

        if "Kids" in flatten_auto_tags(texture_map or {}):
            return 0.88
    designer = normalize_turkish(
        str(dna.get("designer_style") or dna.get("style") or dna.get("brand_style") or "")
    )
    if query_style == "typography" and pf == "typography_text":
        return 0.86
    if query_style in designer or designer in query_style:
        return 0.82
    if query_style in blob_n:
        return 0.68
    return 0.0


def _texture_match(query_texture: str, tf: str, dna: dict, blob_n: str) -> float:
    if not query_texture:
        return 0.0
    tex_class = normalize_turkish(str(dna.get("texture_class") or ""))
    if _norm_match(query_texture, tf) or _norm_match(query_texture, tex_class):
        return 0.82
    if normalize_turkish(query_texture) in blob_n:
        return 0.68
    return 0.0


def natural_language_score(
    parsed: ParsedQuery,
    rec: dict[str, Any],
    *,
    texture_map: dict[str, Any],
    blob_n: str,
    ocr: str,
    fname: str,
) -> tuple[float, dict[str, float], list[str]]:
    """Bileşen bazlı doğal dil skoru."""
    if not parsed.has_components:
        return 0.0, {}, []

    from core.pattern_dna import PatternDNA

    dna_raw = texture_map.get("pattern_dna") or {}
    dna = PatternDNA.from_dict(dna_raw).to_dict() if isinstance(dna_raw, dict) else {}
    semantic = texture_map.get("semantic_tags") or {}
    if not isinstance(semantic, dict):
        semantic = {}

    pf = str(rec.get("pattern_family") or texture_map.get("pattern_family") or "")
    pt = str(rec.get("pattern_type") or texture_map.get("animal_print_type") or "")
    tf = str(rec.get("texture_family") or texture_map.get("texture_family") or "")
    cand_color = str(
        dna.get("color_family")
        or texture_map.get("color_family")
        or rec.get("color_family")
        or ""
    )

    part_scores: dict[str, float] = {}
    reasons: list[str] = []

    if parsed.brand:
        sem_flat = []
        for v in semantic.values():
            if isinstance(v, list):
                sem_flat.extend(str(x) for x in v)
            elif v:
                sem_flat.append(str(v))
        bs = brand_match_score(parsed.brand, [ocr, fname, blob_n, " ".join(sem_flat)])
        if bs <= 0:
            for ref in semantic.get("brand_references") or []:
                if brand_match_score(parsed.brand, [str(ref)]):
                    bs = 0.82
                    break
        if bs <= 0 and _norm_match(parsed.brand, str(dna.get("designer_style") or "")):
            bs = 0.78
        if bs > 0:
            part_scores["nl_brand"] = bs
            reasons.append(f"Marka: {parsed.brand}")

    if parsed.color:
        cs = _color_match(
            parsed.color,
            cand_color,
            blob_n,
            texture_map.get("color_index") if isinstance(texture_map, dict) else None,
        )
        if cs > 0:
            part_scores["nl_color"] = cs
            reasons.append(f"Renk: {parsed.color_tokens[0] if parsed.color_tokens else parsed.color}")

    if parsed.motif:
        ms = _motif_match(
            parsed.motif,
            pf=pf,
            pt=pt,
            dna=dna,
            semantic=semantic,
            blob_n=blob_n,
        )
        if ms > 0:
            part_scores["nl_motif"] = ms
            reasons.append(f"Motif: {parsed.motif}")

    if parsed.repeat:
        rs = _repeat_match(parsed.repeat, dna, texture_map)
        if rs > 0:
            part_scores["nl_repeat"] = rs
            reasons.append(f"Repeat/ölçek: {parsed.repeat}")

    if parsed.style:
        ss = _style_match(parsed.style, dna, pf, blob_n, texture_map)
        if ss > 0:
            part_scores["nl_style"] = ss
            reasons.append(f"Stil: {parsed.style}")

    if parsed.texture:
        ts = _texture_match(parsed.texture, tf, dna, blob_n)
        if ts > 0:
            part_scores["nl_texture"] = ts
            reasons.append(f"Doku: {parsed.texture}")

    requested = len(parsed.components)
    if not part_scores:
        return 0.0, {}, []

    matched_avg = sum(part_scores.values()) / requested
    coverage = len(part_scores) / max(requested, 1)
    final = matched_avg * (0.55 + 0.45 * coverage)
    if len(part_scores) >= 2:
        final = min(1.0, final + 0.04 * (len(part_scores) - 1))

    breakdown = {k: round(v, 4) for k, v in part_scores.items()}
    breakdown["nl_score"] = round(final, 4)
    breakdown["nl_coverage"] = round(coverage, 4)
    return final, breakdown, reasons
