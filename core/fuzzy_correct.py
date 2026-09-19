"""Tekstil sorgularında yazım hatası düzeltme.

Özellikler:
- TERM_SYNONYMS + TERM_FAMILY_HINTS sözlüğüne karşı difflib eşleştirme
- Türkçe normalize (leoaprd → leopard)
- Minimum benzerlik eşiği (0.72) — gereksiz düzeltme yok
- Sık yazım hataları tablosu (sabit mapping — hız için)
- Sonuç: (düzeltilmiş_sorgu, güven, orijinal_sorgu)
"""

from __future__ import annotations

import difflib
from functools import lru_cache
from typing import NamedTuple

_TYPO_MAP: dict[str, str] = {
    # Hayvan
    "leoaprd": "leopard",
    "leoapr": "leopard",
    "leaoprd": "leopard",
    "leoperd": "leopard",
    "lepoard": "leopard",
    "leoprad": "leopard",
    "leoopard": "leopard",
    "leaopard": "leopard",
    "leaoprd": "leopard",
    "leopar": "leopar",          # doğru Türkçe form
    "leapar": "leopar",
    "leoaр": "leopar",
    "lepard": "leopard",
    "zebrra": "zebra",
    "zbera": "zebra",
    "zebera": "zebra",
    "tigar": "tiger",
    "tiiger": "tiger",
    "kaplan": "kaplan",
    "jaquar": "jaguar",
    "jaguaar": "jaguar",
    "jaguar":  "jaguar",   # doğru form — difflib jakar'a yönlendirmesin
    "cheetah": "cheetah",
    "panther": "panther",
    "ocelot":  "ocelot",
    "giraffe": "giraffe",
    "cheeta": "cheetah",
    "cheetaah": "cheetah",
    "cheetagh": "cheetah",
    "sneke": "snake",
    "snaek": "snake",
    "yılan": "yılan",
    # Çiçek
    "florla": "floral",
    "florl": "floral",
    "floerl": "floral",
    "çiçekl": "çiçek",
    "cicekl": "cicek",
    "cicek": "cicek",
    "foral": "floral",
    # Marka
    "louıs": "louis",
    "loius": "louis",
    "loius vuitton": "louis vuitton",
    "louıs vuitton": "louis vuitton",
    "louis vuıtton": "louis vuitton",
    "louis vutton": "louis vuitton",
    "louis viutton": "louis vuitton",
    "lvuitton": "louis vuitton",
    "guci": "gucci",
    "guccci": "gucci",
    "guchhi": "gucci",
    "givenchy": "givenchy",
    "burburry": "burberry",
    "burbery": "burberry",
    "feendi": "fendi",
    "fendie": "fendi",
    "versacce": "versace",
    "versaace": "versace",
    "balenciga": "balenciaga",
    "balencaga": "balenciaga",
    # Desen
    "pasiley": "paisley",
    "pailey": "paisley",
    "pasley": "paisley",
    "geometrik": "geometrik",   # doğru
    "geometrick": "geometrik",
    "geomtric": "geometric",
    "merble": "marble",
    "marbel": "marble",
    "mrable": "marble",
    "abstact": "abstract",
    "abtract": "abstract",
    "abstrat": "abstract",
    "dantle": "dantel",
    "dantelle": "dantel",
    "lase": "lace",
    "laace": "lace",
    "lale": "lale",
    "tulip": "tulip",
    "papatya": "papatya",
    "daisy": "daisy",
    "puantye": "puantiye",
    "puantıye": "puantiye",
    "puantye": "puantiye",
    "ekosee": "ekose",
    "ekoze": "ekose",
    "stripee": "stripe",
    "strıpe": "stripe",
    "çizgı": "çizgi",
    "jacuard": "jacquard",
    "jaquard": "jacquard",
    "jakard": "jakar",
    "monogramm": "monogram",
    "barroque": "baroque",
    "barock": "baroque",
    "boroque": "baroque",
}

_SIMILARITY_THRESHOLD = 0.72
_VOCABULARY: list[str] = []

# Desen ontolojisi — lale/lace gibi yanlış normalize yasak
_PROTECTED_TERMS = frozenset(
    {
        "lale",
        "tulip",
        "papatya",
        "daisy",
        "gül",
        "gul",
        "orkide",
        "orchid",
        "şakayık",
        "sakayik",
        "peony",
        "puantiye",
        "noktalı",
        "noktali",
        "çiçek",
        "cicek",
        "çilek",
        "cilek",
        "strawberry",
        "leopar",
        "zebra",
        "yılan",
        "yilan",
        "karga",
        "leylek",
        "togg",
        "karınca",
        "karinca",
    }
)


class CorrectionResult(NamedTuple):
    corrected: str
    confidence: float   # 1.0 = sabit map, 0.72-1.0 = difflib
    original: str
    was_corrected: bool


def _build_vocabulary() -> list[str]:
    """TERM_SYNONYMS + TERM_FAMILY_HINTS anahtarlarından sözcük listesi."""
    try:
        from core.textile_terms import TERM_FAMILY_HINTS, TERM_SYNONYMS
        vocab: set[str] = set()
        for k in TERM_SYNONYMS:
            vocab.add(k)
            for v in TERM_SYNONYMS[k]:
                vocab.add(v)
        vocab.update(TERM_FAMILY_HINTS.keys())
        # Marka sözlüğü
        try:
            from core.brand_aliases import BRAND_ALIASES
            vocab.update(BRAND_ALIASES.values())
        except Exception:
            pass
        return sorted(vocab)
    except Exception:
        return []


def _vocabulary() -> list[str]:
    global _VOCABULARY
    if not _VOCABULARY:
        _VOCABULARY = _build_vocabulary()
    return _VOCABULARY


@lru_cache(maxsize=4096)
def correct_term(term: str) -> CorrectionResult:
    """Tek bir terimi düzelt."""
    raw = (term or "").strip()
    if not raw or len(raw) < 3:
        return CorrectionResult(raw, 1.0, raw, False)

    try:
        from core.concept_query_normalize import turkish_casefold

        low = turkish_casefold(raw)
    except Exception:
        low = raw.lower()
    if low in _PROTECTED_TERMS:
        return CorrectionResult(low, 1.0, raw, False)

    # 1. Sabit yazım hatası tablosu
    if low in _TYPO_MAP:
        fixed = _TYPO_MAP[low]
        return CorrectionResult(fixed, 1.0, raw, fixed != low)
    try:
        from core.textile_terms import normalize_turkish

        nlow = normalize_turkish(low)
        if nlow in _TYPO_MAP:
            fixed = _TYPO_MAP[nlow]
            return CorrectionResult(fixed, 1.0, raw, fixed != low)
    except Exception:
        pass

    # 2. Tam eşleşme (zaten doğru)
    vocab = _vocabulary()
    if low in vocab:
        return CorrectionResult(low, 1.0, raw, False)

    # 3. difflib fuzzy — kısa terimler için minimum 3 karakter
    if len(raw) < 4:
        return CorrectionResult(low, 1.0, raw, False)

    matches = difflib.get_close_matches(low, vocab, n=2, cutoff=_SIMILARITY_THRESHOLD)
    if matches:
        best = matches[0]
        ratio = difflib.SequenceMatcher(None, low, best).ratio()
        # Ambiguous: second candidate close → do not auto-correct
        if len(matches) > 1:
            ratio2 = difflib.SequenceMatcher(None, low, matches[1]).ratio()
            if ratio2 >= ratio - 0.04 and ratio < 0.92:
                return CorrectionResult(low, round(ratio, 3), raw, False)
        return CorrectionResult(best, round(ratio, 3), raw, best != low)

    return CorrectionResult(low, 1.0, raw, False)


def correct_query(query: str) -> CorrectionResult:
    """Tam sorguyu düzelt.

    Önce sorgunun tamamını dene, yoksa tek tek kelimeleri düzelt.
    """
    raw = (query or "").strip()
    if not raw:
        return CorrectionResult("", 1.0, raw, False)

    # Tüm sorgu
    whole = correct_term(raw)
    if whole.was_corrected:
        return whole

    # Kelime kelime
    words = raw.lower().split()
    corrected_words: list[str] = []
    any_fixed = False
    min_conf = 1.0
    for w in words:
        res = correct_term(w)
        corrected_words.append(res.corrected)
        if res.was_corrected:
            any_fixed = True
        if res.confidence < min_conf:
            min_conf = res.confidence

    corrected = " ".join(corrected_words)
    return CorrectionResult(corrected, min_conf, raw, any_fixed)
