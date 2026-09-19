"""Marka alias sözlüğü — OCR kısa logolarını marka adına çevirir."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

# Kısa logo / kısaltma → marka adı (küçük harf anahtar)
BRAND_ALIASES: dict[str, str] = {
    # Louis Vuitton
    "lv": "louis vuitton",
    "louis": "louis vuitton",
    "louise": "louis vuitton",
    "vuitton": "louis vuitton",
    "louis vuitton": "louis vuitton",
    "lv monogram": "louis vuitton",
    "louis vuitton monogram": "louis vuitton",
    # Gucci
    "gg": "gucci",
    "gucci": "gucci",
    "gg supreme": "gucci",
    "gucci gg": "gucci",
    "double g": "gucci",
    "interlocking g": "gucci",
    # Yves Saint Laurent
    "ysl": "yves saint laurent",
    "saint": "yves saint laurent",
    "laurent": "yves saint laurent",
    "saint laurent": "yves saint laurent",
    "yves saint laurent": "yves saint laurent",
    # Christian Dior
    "cd": "christian dior",
    "dior": "christian dior",
    "christian dior": "christian dior",
    "dior oblique": "christian dior",
    # Fendi
    "ff": "fendi",
    "fendi": "fendi",
    "fendi ff": "fendi",
    # Burberry
    "tb": "burberry",
    "burberry": "burberry",
    "nova check": "burberry",
    # Michael Kors
    "mk": "michael kors",
    "michael kors": "michael kors",
    "michael": "michael kors",
    # Diğer
    "dg": "dolce gabbana",
    "dolce": "dolce gabbana",
    "gabbana": "dolce gabbana",
    "dolce gabbana": "dolce gabbana",
    "celine": "celine",
    "céline": "celine",
    "paris": "paris",
    "chanel": "chanel",
    "hermes": "hermes",
    "prada": "prada",
    "versace": "versace",
    "valentino": "valentino",
    "balenciaga": "balenciaga",
    "givenchy": "givenchy",
    "armani": "armani",
    "giorgio armani": "armani",
    "emporio armani": "armani",
    "moncler": "moncler",
    "moschino": "moschino",
    "ferragamo": "ferragamo",
    "salvatore ferragamo": "ferragamo",
    "balmain": "balmain",
    "alexander mcqueen": "alexander mcqueen",
    "mcqueen": "alexander mcqueen",
    "ralph lauren": "ralph lauren",
    "tommy hilfiger": "tommy hilfiger",
    "calvin klein": "calvin klein",
    "ck": "calvin klein",
    # Amiri
    "amiri": "amiri",
    "amiri los angeles": "amiri",
    "nike": "nike",
    "adidas": "adidas",
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9çğıöşü]+", re.IGNORECASE)
SHORT_BRAND_ALIASES = frozenset({"lv", "gg", "ff", "cd", "ysl", "mk", "tb", "dg"})


def normalize_brand_key(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = (
        raw.replace("ç", "c")
        .replace("ğ", "g")
        .replace("ı", "i")
        .replace("ö", "o")
        .replace("ş", "s")
        .replace("ü", "u")
        .replace("é", "e")
    )
    return re.sub(r"\s+", " ", raw).strip()


def canonical_brand_display(text: str, db_path: str | None = None) -> str:
    """Case-insensitive display form: Amiri/amiri/AMIRI → Amiri (tree/alias)."""
    raw = " ".join(str(text or "").strip().split())
    if raw.lower().startswith("marka/"):
        raw = raw.split("/", 1)[1].strip()
    key = normalize_brand_key(raw)
    if not key:
        return ""
    try:
        from core.category_tree import CATEGORY_TREE

        for name in (CATEGORY_TREE.get("Marka", {}) or {}):
            if normalize_brand_key(str(name)) == key:
                return str(name)
    except Exception:
        pass
    alias_can = BRAND_ALIASES.get(key)
    if alias_can:
        for name in BRAND_ALIASES.values():
            if normalize_brand_key(name) == normalize_brand_key(alias_can):
                try:
                    from core.category_tree import CATEGORY_TREE

                    for tree_name in (CATEGORY_TREE.get("Marka", {}) or {}):
                        if normalize_brand_key(str(tree_name)) == normalize_brand_key(alias_can):
                            return str(tree_name)
                except Exception:
                    pass
                break
        return " ".join(w.capitalize() for w in normalize_brand_key(alias_can).split())
    if db_path:
        dyn = _dynamic_brand_aliases(db_path)
        hit = dyn.get(key)
        if hit:
            hk = normalize_brand_key(hit)
            if hk == key:
                return " ".join(w.capitalize() for w in hk.split())
            return canonical_brand_display(hit, None) or str(hit)
    return " ".join(w.capitalize() for w in key.split())


def tokenize_brand_text(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if len(t) >= 2]


def _dynamic_brand_aliases(db_path: str | None = None) -> dict[str, str]:
    """Kalıcı kullanıcı marka hafızası: search_memory.db + legacy read-only."""
    if not db_path:
        return {}
    try:
        from core.search_memory import load_brand_memory

        out = dict(load_brand_memory(db_path))
    except Exception:
        out = {}
    try:
        from core.category_memory import all_dynamic

        for row in all_dynamic(db_path):
            parent = str(row.get("parent") or "")
            if parent.casefold() != "marka":
                continue
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            key = normalize_brand_key(label)
            if key and key not in out:
                out[key] = label
            for alias in row.get("aliases") or []:
                ak = normalize_brand_key(str(alias))
                if ak and ak not in out:
                    out[ak] = label
    except Exception:
        pass
    return out


def register_brand_alias(db_path: str, alias: str, canonical: str) -> bool:
    """Kullanıcı marka eşlemesi — search_memory.db, Pattern Index değil."""
    try:
        from core.search_memory import register_brand_memory

        return register_brand_memory(db_path, alias, canonical)
    except Exception:
        return False


def resolve_brand_alias(token: str, db_path: str | None = None) -> str | None:
    key = normalize_brand_key(token)
    if not key:
        return None
    dynamic = _dynamic_brand_aliases(db_path)
    return dynamic.get(key) or BRAND_ALIASES.get(key)


def expand_brand_terms(text: str, db_path: str | None = None) -> list[str]:
    """OCR/sorgu metninden marka alias genişletmeleri üret."""
    out: list[str] = []
    seen: set[str] = set()
    norm = normalize_brand_key(text)
    if not norm:
        return out

    brand = resolve_brand_alias(norm, db_path)
    if brand and brand not in seen:
        seen.add(brand)
        out.append(brand)

    tokens = tokenize_brand_text(norm)
    for tok in tokens:
        brand = resolve_brand_alias(tok, db_path)
        if brand and brand not in seen:
            seen.add(brand)
            out.append(brand)
    for i, tok in enumerate(tokens):
        for other in tokens[i + 1 :]:
            pair = f"{tok} {other}"
            brand = resolve_brand_alias(pair, db_path)
            if brand and brand not in seen:
                seen.add(brand)
                out.append(brand)
    return out



def brand_names(db_path: str | None = None) -> list[str]:
    """Bilinen + kullanıcı tarafından öğrenilmiş kanonik marka adlarını döndür."""
    by_key: dict[str, str] = {}
    def _add(name: str) -> None:
        display = canonical_brand_display(name, db_path)
        key = normalize_brand_key(display or name)
        if not key:
            return
        prev = by_key.get(key)
        if prev is None or (display and display != display.lower() and prev == prev.lower()):
            by_key[key] = display or name

    for v in BRAND_ALIASES.values():
        if str(v).strip():
            _add(str(v))
    for v in _dynamic_brand_aliases(db_path).values():
        if str(v).strip():
            _add(str(v))
    try:
        from core.category_tree import CATEGORY_TREE
        for k in (CATEGORY_TREE.get("Marka", {}) or {}):
            if str(k).strip():
                _add(str(k))
    except Exception:
        pass
    return sorted(by_key.values(), key=lambda x: (normalize_brand_key(x).replace(" ", ""), x.lower()))


def brand_suggestions(query: str, *, limit: int = 8, db_path: str | None = None) -> list[tuple[str, float]]:
    """Yazım hatalı/kısaltılmış marka sorguları için seçim yapılabilir adaylar.

    Örn. ``loui`` ve ``louise`` -> ``Louis Vuitton``.
    Belirsiz sonuçlarda otomatik seçim yapmaz; kullanıcıya adayları gösterir.
    """
    q = normalize_brand_key(query)
    if len(q) < 2:
        return []
    candidates = brand_names(db_path)
    scored: list[tuple[str, float]] = []
    for name in candidates:
        nkey = normalize_brand_key(name)
        compact = nkey.replace(" ", "")
        ratio = SequenceMatcher(None, q.replace(" ", ""), compact).ratio()
        q_tokens = q.split()
        n_tokens = nkey.split()
        prefix = 0.0
        for qt in q_tokens:
            if len(qt) >= 3 and any(nt.startswith(qt) or qt.startswith(nt[:max(3, min(5, len(nt)))]) for nt in n_tokens):
                prefix = max(prefix, 0.94)
        if compact.startswith(q.replace(" ", "")) or any(t.startswith(q) for t in n_tokens if len(q) >= 3):
            prefix = max(prefix, 0.98)
        score = max(ratio, prefix)
        if score >= (0.56 if len(q) <= 5 else 0.62):
            scored.append((name, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:max(1, int(limit))]

def brand_alias_keys_for(brand: str) -> list[str]:
    """Bir marka için tüm kısa alias anahtarları."""
    bn = normalize_brand_key(brand)
    return [k for k, v in BRAND_ALIASES.items() if normalize_brand_key(v) == bn]


def query_brand_needles(query: str, db_path: str | None = None) -> list[str]:
    """Sorgu için bir kez üretilen marka arama iğneleri (normalize)."""
    q_norm = normalize_brand_key(query)
    brands = set(expand_brand_terms(query, db_path))
    q_brand = resolve_brand_alias(q_norm, db_path)
    if q_brand:
        brands.add(q_brand)
    needles: list[str] = []
    seen: set[str] = set()
    for brand in brands:
        bn = normalize_brand_key(brand)
        if bn and bn not in seen:
            seen.add(bn)
            needles.append(bn)
        for key in brand_alias_keys_for(brand):
            kn = normalize_brand_key(key)
            if kn and kn not in seen:
                seen.add(kn)
                needles.append(kn)
    if q_norm and q_norm not in seen:
        needles.append(q_norm)
    return needles


def enrich_ocr_text(ocr_text: str) -> str:
    """Index blob için OCR metnine marka alias'larını ekle."""
    raw = (ocr_text or "").strip()
    if not raw:
        return ""
    slim: list[str] = [raw]
    brands = expand_brand_terms(raw)
    for brand in brands:
        slim.append(brand)
        slim.extend(brand_alias_keys_for(brand))
    return " ".join(dict.fromkeys(slim))


def brand_match_score(
    query: str,
    haystacks: Iterable[str],
    *,
    needles: list[str] | None = None,
) -> float:
    """Sorgu marka alias ile OCR/filename haystack'inde eşleşirse skor.

    Haystack üzerinde expand_brand_terms çağırma — OCR metninde O(n²) patlıyordu.
    """
    if needles is None:
        needles = query_brand_needles(query)
    if not needles:
        return 0.0

    joined = " ".join(h or "" for h in haystacks)
    joined_n = normalize_brand_key(joined)
    if not joined_n:
        return 0.0

    padded = f" {joined_n} "
    best = 0.0
    for kn in needles:
        if not kn:
            continue
        hit = kn == joined_n or f" {kn} " in padded or kn in joined_n
        if not hit:
            continue
        short = kn in SHORT_BRAND_ALIASES or len(kn) <= 2
        best = max(best, 0.42 if short else 0.90)
        if not short and best >= 0.90:
            return 0.90
    return best
