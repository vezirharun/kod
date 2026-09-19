"""Query Intent Router — search-channel routing (not motif detection).

This is a lightweight query classifier. It never replaces DINO/FAISS/Pattern DNA;
it only labels which search channel a query belongs to.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from core.textile_terms import normalize_turkish


@dataclass(frozen=True)
class ConceptChannel:
    channel: str
    value: str
    label: str
    token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "value": self.value,
            "label": self.label,
            "token": self.token,
        }


@dataclass(frozen=True)
class QueryIntent:
    kind: str = "pattern"          # pattern | person | person_name | gender | hybrid
    value: str = ""
    tokens: tuple[str, ...] = ()
    explicit: bool = False
    channel: str = "pattern"       # brand | category | material_color | motif_object | texture_style | pattern | person
    label: str = ""                # human route label
    channels: tuple[ConceptChannel, ...] = ()
    unknown_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "tokens": list(self.tokens),
            "explicit": self.explicit,
            "channel": self.channel,
            "label": self.label,
            "channels": [c.to_dict() for c in self.channels],
            "unknown_tokens": list(self.unknown_tokens),
            "compound": len(self.channels) >= 2,
        }


_GENDER = {
    "kadın": "FEMALE", "kadin": "FEMALE", "woman": "FEMALE",
    "women": "FEMALE", "female": "FEMALE", "bayan": "FEMALE",
    "erkek": "MALE", "man": "MALE", "men": "MALE", "male": "MALE",
}

_PERSON_WORDS = {"kişi", "kisi", "person"}
_BLOCKED_PERSONISH = {
    "desen", "pattern", "çiçek", "cicek", "leopar", "leopard", "gül", "gul",
    "elbise", "etek", "pantolon", "gömlek", "gomlek", "çanta", "canta",
    "ayakkabı", "ayakkabi", "kadın", "kadin", "erkek", "mavi", "kırmızı",
    "kirmizi", "siyah", "beyaz", "altın", "altin", "gümüş", "gumus",
    "marka", "logo", "aksesuar", "doku", "repeat", "motif", "kamuflaj",
    "barok", "baroque", "paisley", "geometrik", "geometric", "ekose", "çizgi",
    "cizgi", "zebra", "kaplan", "yılan", "yilan", "tropikal", "botanik",
    "vintage", "modern", "boho", "romantik", "safari", "lüks", "luks",
    "gucci", "dior", "versace", "burberry", "chanel", "prada", "armani",
    "amiri", "dolce", "gabbana", "louis", "vuitton",
}

# Query-intent lexicons (not detectors).
_MOTIF_OBJECT = {
    "kiraz": "kiraz", "cherry": "kiraz",
    "yaprak": "yaprak", "leaf": "yaprak", "leaves": "yaprak",
    "kelebek": "kelebek", "butterfly": "kelebek",
    "kanat": "kanat", "wing": "kanat", "wings": "kanat",
    "maske": "maske", "mask": "maske",
    "kafatasi": "kafatasi", "skull": "kafatasi",
}
_MATERIAL_COLOR = {
    "gold": "gold", "golden": "gold", "altin": "gold",
    "gumus": "gumus", "silver": "gumus",
}
_TEXTURE_STYLE = {
    "grunge": ("texture_style", "texture/style"),
    "popart": ("texture_style", "style"),
    "pop": ("texture_style", "style"),  # only with art
    "dantel": ("texture_style", "pattern/texture"),
    "lace": ("texture_style", "pattern/texture"),
}
_PRODUCT_CATEGORY = {
    "aksesuar": ("Aksesuar", "Aksesuar"),
    "accessory": ("Aksesuar", "Aksesuar"),
    "accessories": ("Aksesuar", "Aksesuar"),
    "kolye": ("Aksesuar", "Aksesuar → kolye"),
    "necklace": ("Aksesuar", "Aksesuar → kolye"),
}

_CHANNEL_FROM_CONCEPT = {
    "brand": ("brand", "Marka"),
    "marka": ("brand", "Marka"),
    "category": ("category", "category"),
    "material_color": ("material_color", "material/color"),
    "material": ("material_color", "material/color"),
    "color": ("material_color", "material/color"),
    "motif_object": ("motif_object", "motif/object"),
    "motif": ("motif_object", "motif/object"),
    "object": ("motif_object", "motif/object"),
    "texture_style": ("texture_style", "texture/style"),
    "style": ("texture_style", "style"),
    "pattern_texture": ("texture_style", "pattern/texture"),
    "pattern": ("pattern", "pattern"),
    "object": ("object", "object"),
    "visual_concept": ("learned", "Öğrenilmiş kavram"),
    "learned": ("learned", "Öğrenilmiş kavram"),
}

# Animal-print / pattern-family tokens (not motif-object lexicon).
_PATTERN_PRINT = {
    "leopar": "leopard",
    "leopard": "leopard",
    "leo": "leopard",
    "zebra": "zebra",
    "snake": "snake",
    "yilan": "snake",
    "ekose": "tartan",
    "kareli": "tartan",
    "tartan": "tartan",
    "plaid": "tartan",
    "check": "tartan",
    "cizgili": "stripe",
    "cizgi": "stripe",
    "stripe": "stripe",
    "striped": "stripe",
    "puantiye": "polka",
    "polka": "polka",
    "geometrik": "geometric",
    "geometric": "geometric",
    "kamuflaj": "camouflage",
    "camouflage": "camouflage",
    "camo": "camouflage",
}
# Floral / print motifs that are not the V1 object lexicon.
_MOTIF_PRINT = {
    "cicek": "floral",
    "floral": "floral",
    "flower": "floral",
    "flowers": "floral",
}

_SKIP_TOKENS = {
    "ve", "ile", "bir", "the", "a", "an", "desen", "pattern", "print",
    "kumas", "kumaş", "fabric", "deseni", "uzerine", "ustune",
}

_OBJECT_REAL = {
    "kiraz": ("cherry", "object"),
    "cilek": ("strawberry", "object"),
    "elma": ("apple", "object"),
    "kaplan": ("tiger", "object"),
    "tiger": ("tiger", "object"),
    "bisiklet": ("bicycle", "object"),
    "bicycle": ("bicycle", "object"),
    "tavsan": ("rabbit", "object"),
    "kus": ("bird", "object"),
    "kopek": ("dog", "object"),
    "kedi": ("cat", "object"),
    "araba": ("car", "object"),
    "ayakkabi": ("shoe", "object"),
    "canta": ("handbag", "object"),
    "cocuk": ("child", "person"),
}

_CHANNEL_PRI = {
    "brand": 0,
    "category": 1,
    "material_color": 2,
    "object": 3,
    "motif_object": 4,
    "texture_style": 5,
    "pattern": 6,
    "person": 7,
}


def _tokens(text: str) -> list[str]:
    norm = normalize_turkish(text or "")
    return [x for x in re.findall(r"[a-z0-9ğüşöçıİĞÜŞÖÇ]+", norm) if len(x) >= 2]


def _intent(
    kind,
    value="",
    tokens=(),
    explicit=False,
    channel="pattern",
    label="",
    channels=(),
    unknown_tokens=(),
):
    return QueryIntent(
        kind=kind,
        value=value,
        tokens=tuple(tokens),
        explicit=explicit,
        channel=channel,
        label=label or channel,
        channels=tuple(channels),
        unknown_tokens=tuple(unknown_tokens),
    )


def _brand_for_token(tok: str, db_path: str | None) -> str:
    try:
        from core.brand_aliases import BRAND_ALIASES, normalize_brand_key

        key = normalize_brand_key(tok)
        if key and key in BRAND_ALIASES:
            return BRAND_ALIASES[key]
        if db_path and key:
            from core.search_memory import load_brand_memory

            mem = load_brand_memory(db_path)
            if key in mem:
                return mem[key]
    except Exception:
        pass
    return ""


def _registry_token(tok: str, db_path: str | None) -> dict[str, Any] | None:
    if not db_path or len(tok) < 2:
        return None
    try:
        from core.concept_registry import find

        hits = find(db_path, tok, limit=8)
    except Exception:
        return None
    for hit in hits:
        can = normalize_turkish(str(hit.get("canonical") or ""))
        if can == tok and float(hit.get("score") or 0) >= 0.92:
            return hit
    return None


def _channels_from_product_token(tok: str) -> list[ConceptChannel]:
    parent, lab = _PRODUCT_CATEGORY[tok]
    out = [ConceptChannel("category", parent, "Aksesuar", tok)]
    if tok not in {"aksesuar", "accessory", "accessories"}:
        out.append(ConceptChannel("category", f"{parent}/kolye", lab, tok))
    return out


def _classify_token(tok: str, db_path: str | None) -> list[ConceptChannel]:
    if tok in _SKIP_TOKENS:
        return []
    if tok in _GENDER:
        return [ConceptChannel("person", _GENDER[tok], "person", tok)]
    if tok in _PRODUCT_CATEGORY:
        return _channels_from_product_token(tok)
    if tok in _MATERIAL_COLOR:
        return [ConceptChannel("material_color", _MATERIAL_COLOR[tok], "material/color", tok)]
    if tok in _TEXTURE_STYLE and tok != "pop":
        ch, lab = _TEXTURE_STYLE[tok]
        return [ConceptChannel(ch, tok, lab, tok)]
    if tok in _MOTIF_OBJECT:
        return [ConceptChannel("motif_object", _MOTIF_OBJECT[tok], "motif/object", tok)]
    if tok in _OBJECT_REAL:
        val, lab = _OBJECT_REAL[tok]
        ch = "person" if lab == "person" else "object"
        return [ConceptChannel(ch, val, lab, tok)]
    try:
        from core.geometric_concepts import resolve_geometric_query

        geo = resolve_geometric_query(tok)
        if geo:
            return [ConceptChannel("pattern", geo.concept_id, geo.label, tok)]
    except Exception:
        pass
    if tok in _PATTERN_PRINT:
        return [ConceptChannel("pattern", _PATTERN_PRINT[tok], "pattern", tok)]
    if tok in _MOTIF_PRINT:
        return [ConceptChannel("motif_object", _MOTIF_PRINT[tok], "motif/object", tok)]
    brand = _brand_for_token(tok, db_path)
    if brand:
        return [ConceptChannel("brand", brand, "Marka", tok)]
    hit = _registry_token(tok, db_path)
    if hit:
        ctype = str(hit.get("concept_type") or "").lower()
        mapped = _CHANNEL_FROM_CONCEPT.get(ctype) or ("learned", "Öğrenilmiş kavram")
        ch, lab = mapped
        parent = str(hit.get("parent") or "")
        can = str(hit.get("canonical") or tok)
        if parent and ch == "category":
            lab = f"{parent} → {can}"
        return [ConceptChannel(ch, can, lab, tok)]
    return []


def intent_retrieval_needles(intent: QueryIntent) -> list[str]:
    """Needles for existing FTS/terms expansion — not a new ranker."""
    out: list[str] = []
    for ch in intent.channels:
        for piece in (ch.token, ch.value):
            raw = str(piece or "").strip()
            if not raw:
                continue
            out.append(raw)
            if "/" in raw:
                for part in raw.split("/"):
                    p = part.strip()
                    if len(p) >= 2:
                        out.append(p)
        if ch.channel == "material_color" and ch.value == "gold":
            out.extend(["gold", "altin", "altın", "golden"])
        if ch.channel == "material_color" and ch.value == "gumus":
            out.extend(["gumus", "gümüş", "silver"])
        if ch.channel == "category":
            out.extend(["aksesuar", "accessory", "kolye", "necklace"])
        if ch.channel == "texture_style" and ch.value in {"dantel", "lace"}:
            out.extend(["dantel", "lace"])
        if ch.channel == "motif_object" and ch.value == "floral":
            out.extend(["cicek", "çiçek", "floral", "flower"])
        if ch.channel == "pattern":
            try:
                from core.geometric_concepts import concept_by_id, resolve_geometric_query, retrieval_needles

                geo = concept_by_id(str(ch.value or "")) or resolve_geometric_query(
                    str(ch.token or ch.value or "")
                )
                out.extend(retrieval_needles(geo))
            except Exception:
                pass
        if ch.channel == "pattern" and ch.value == "leopard":
            out.extend(["leopar", "leopard"])
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        k = normalize_turkish(x)
        if not k or k in seen or k in _SKIP_TOKENS:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def classify_query(text: str, db_path: str | None = None) -> QueryIntent:
    raw = " ".join((text or "").strip().split())
    norm = normalize_turkish(raw)
    if not norm:
        return QueryIntent()
    toks = _tokens(raw)

    if norm in _GENDER:
        return _intent("gender", _GENDER[norm], toks, True, "person", "gender")

    m = re.fullmatch(r"(?:kişi|kisi|person)[ _-]?(\d{1,6})", norm)
    if m:
        return _intent("person", f"person_{int(m.group(1)):04d}", toks, True, "person", "person")

    m = re.fullmatch(r"(?:kişi|kisi|person)[: ]+(.+)", norm)
    if m:
        name = m.group(1).strip()
        return _intent("person_name", name, _tokens(name), True, "person", "person")

    joined = " ".join(toks)
    if joined in {"pop art", "pop-art"} or (len(toks) == 2 and toks[0] == "pop" and toks[1] == "art"):
        ch = ConceptChannel("texture_style", "popart", "style", "popart")
        return _intent("pattern", "popart", toks, True, "texture_style", "style", channels=(ch,))

    if len(toks) == 1:
        brand = (
            _brand_for_token(raw, db_path)
            or _brand_for_token(norm, db_path)
            or _brand_for_token(toks[0], db_path)
        )
        if brand:
            ch = ConceptChannel("brand", brand, "Marka", toks[0])
            return _intent("pattern", brand, toks, True, "brand", "Marka", channels=(ch,))

    parsed: list[ConceptChannel] = []
    unknown: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    for tok in toks:
        parts = _classify_token(tok, db_path)
        if not parts:
            if tok not in _SKIP_TOKENS:
                unknown.append(tok)
            continue
        for ch in parts:
            key = (ch.channel, normalize_turkish(ch.value))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            parsed.append(ch)

    if parsed:
        best_pri = min(_CHANNEL_PRI.get(c.channel, 99) for c in parsed)
        same = [c for c in parsed if _CHANNEL_PRI.get(c.channel, 99) == best_pri]
        primary = max(same, key=lambda c: (len(c.value), len(c.label)))
        return _intent(
            "pattern",
            primary.value,
            toks,
            True,
            primary.channel,
            primary.label,
            channels=tuple(parsed),
            unknown_tokens=tuple(dict.fromkeys(unknown)),
        )

    if db_path and len(norm) >= 2:
        try:
            from core.concept_registry import find

            hits = find(db_path, raw, limit=1)
            if hits and float(hits[0].get("score") or 0) >= 0.92:
                can = normalize_turkish(str(hits[0].get("canonical") or ""))
                if can == norm:
                    ctype = str(hits[0].get("concept_type") or "").lower()
                    mapped = _CHANNEL_FROM_CONCEPT.get(ctype) or (
                        "learned",
                        "Öğrenilmiş kavram",
                    )
                    ch, lab = mapped
                    parent = str(hits[0].get("parent") or "")
                    if parent and ch == "category":
                        lab = f"{parent} → {hits[0].get('canonical')}"
                    val = str(hits[0].get("canonical") or "")
                    one = ConceptChannel(ch, val, lab, toks[0] if toks else val)
                    return _intent("pattern", val, toks, True, ch, lab, channels=(one,))
        except Exception:
            pass

    if len(toks) >= 2 and not any(t in _BLOCKED_PERSONISH for t in toks):
        return _intent(
            "person_name",
            norm,
            toks,
            False,
            "person",
            "person",
            unknown_tokens=tuple(dict.fromkeys(unknown)),
        )

    return _intent(
        "pattern",
        "",
        toks,
        False,
        "pattern",
        "pattern",
        unknown_tokens=tuple(dict.fromkeys(unknown)),
    )
