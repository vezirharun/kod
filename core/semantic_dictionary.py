"""Multilingual textile concepts used to expand natural-language queries."""

from __future__ import annotations

import re


SEMANTIC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "yazili": ("typography", "text", "letter", "word", "logo", "monogram"),
    "yazi": ("typography", "text", "letter", "word", "repeated text"),
    "harf": ("letter", "typography", "text", "monogram"),
    "logo": ("logo", "emblem", "symbol", "monogram", "brand pattern", "logo repeat", "repeated logo"),
    "amblem": ("emblem", "logo", "symbol", "arma", "brand pattern"),
    "arma": ("emblem", "logo", "symbol", "badge", "patch", "brand pattern"),
    "patch": ("patch", "badge", "emblem", "arma", "symbol"),
    "monogram": ("monogram", "logo", "letter", "repeat", "luxury", "logo repeat", "repeated logo"),
    "logo repeat": ("monogram", "logo", "repeated logo", "allover repeat", "luxury"),
    "repeated logo": ("monogram", "logo", "logo repeat", "allover repeat", "luxury"),
    "lv": ("lv", "louis vuitton", "monogram", "logo", "luxury", "brand pattern", "logo repeat"),
    "luks": ("luxury", "monogram", "logo", "baroque", "ornament", "brand pattern"),
    "luxury": ("luxury", "luks", "monogram", "logo", "brand style", "brand pattern"),
    "brand style": ("brand style", "luxury", "logo", "monogram", "brand pattern"),
    "jakar": ("jacquard", "jakar", "woven", "fabric texture", "luxury"),
    "jacquard": ("jacquard", "jakar", "woven", "fabric texture", "luxury"),
    "metraj": ("allover", "repeat", "tekrar", "seamless", "pattern"),
    "tekrar": ("repeat", "repeated", "allover", "metraj", "seamless"),
    "allover": ("allover", "repeat", "metraj", "seamless"),
    "askeri": (
        "military", "army", "combat", "camouflage", "camo", "digital camo",
        "woodland", "desert camo", "urban camo", "forest camo",
    ),
    "kamuflaj": (
        "camouflage", "camo", "military", "woodland", "digital camo",
        "desert camo", "urban camo", "forest camo",
    ),
    "cicek": ("floral", "flower", "rose", "tulip", "leaf", "botanical", "bloom"),
    "botanik": ("botanical", "floral", "flower", "leaf", "garden"),
    "hayvan": (
        "animal", "animal print", "leopard", "tiger", "snake", "zebra", "cow",
        "giraffe", "crocodile", "octopus", "butterfly",
    ),
    "geometrik": ("geometric", "geometry", "repeat", "mosaic", "circle", "square"),
    "zincir": ("chain", "link", "gold chain", "ornament"),
    "puantiye": ("polka dot", "dot", "dotted", "circle repeat"),
    "orgu": ("knit", "knitted", "weave", "fabric texture"),
    "denim": ("denim", "jean", "fabric texture"),
    "kot": ("denim", "jean", "fabric texture"),
    "mermer": ("marble", "stone", "fluid art", "abstract"),
    "gucci": (
        "gg", "double g", "interlocking g", "monogram", "luxury", "logo",
        "letter", "typography", "brand pattern",
    ),
    "dolce gabbana": (
        "dg", "majolica", "baroque", "crown", "rose", "floral", "luxury",
        "monogram", "brand pattern",
    ),
    "amiri": (
        "amiri", "amiri los angeles", "luxury", "streetwear", "rock",
        "brand", "logo", "typography",
    ),

    "d&g": ("dg", "baroque", "crown", "rose", "floral", "monogram"),
    "louis vuitton": (
        "lv", "monogram", "flower", "floral", "luxury", "logo", "brand pattern",
    ),
    "dior": ("oblique", "cd", "monogram", "logo", "letter", "luxury"),
    "burberry": ("check", "nova check", "plaid", "tartan", "luxury"),
    "versace": (
        "baroque", "medusa", "greek key", "gold chain", "ornament", "luxury",
    ),
}

BRAND_ALIASES: dict[str, str] = {
    "lv": "louis_vuitton",
    "gucci": "gucci",
    "dolce gabbana": "dolce_gabbana",
    "dolce & gabbana": "dolce_gabbana",
    "d&g": "dolce_gabbana",
    "louis vuitton": "louis_vuitton",
    "dior": "dior",
    "burberry": "burberry",
    "versace": "versace",
    "amiri": "amiri",
    "amiri los angeles": "amiri",
}


def expand_semantic_query(normalized_query: str) -> list[str]:
    """Return concepts for complete words/phrases found in a normalized query."""
    query = f" {normalized_query.strip()} "
    expanded: list[str] = []
    for key, concepts in SEMANTIC_CONCEPTS.items():
        pattern = r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
        if re.search(pattern, query):
            expanded.extend((key, *concepts))
    return list(dict.fromkeys(expanded))


def detect_brand_references(normalized_text: str) -> list[str]:
    found: list[str] = []
    for alias, brand_id in BRAND_ALIASES.items():
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, normalized_text):
            found.append(brand_id)
    return list(dict.fromkeys(found))
