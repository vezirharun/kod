"""Tekstil desen terimleri — Türkçe/İngilizce eş anlamlılar ve aile eşlemesi."""

from __future__ import annotations

import re
import unicodedata

_TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")

# Ana terim → eş anlamlılar (küçük harf)
TERM_SYNONYMS: dict[str, list[str]] = {
    "leopar": ["leopard", "leo", "leopar", "jaguar print"],
    "leopard": ["leopar", "leo", "leopard", "cheetah print"],
    "animal print": [
        "leopar",
        "leopard",
        "zebra",
        "snake",
        "hayvan deseni",
        "animal",
        "hayvan",
    ],
    "animal": ["animal print", "hayvan", "hayvan deseni"],
    "hayvan": ["animal", "animal print", "hayvan deseni"],
    "çiçek": ["flower", "cicek", "çiçek", "bloom"],
    "cicek": ["çiçek", "flower", "cicek"],
    "flower": ["çiçek", "cicek", "flower", "bloom"],
    "floral": ["botanical", "floral"],
    "küçük çiçek": [
        "small flower",
        "mini floral",
        "minik çiçek",
        "kucuk cicek",
        "petite floral",
        "ditsy",
        "çıtır çiçek",
    ],
    "çıtır çiçek": ["küçük çiçek", "ditsy floral", "mini floral", "small flower"],
    "minik çiçek": ["küçük çiçek", "small flower", "mini floral", "minik cicek"],
    "small flower": ["küçük çiçek", "minik çiçek", "mini floral", "small flower"],
    "rose": ["gül", "gul", "rose", "roses"],
    "gül": ["rose", "gul", "gül"],
    "dudak": ["lips", "lip", "dudak", "dudaklar"],
    "dudaklar": ["dudak", "lips", "lip"],
    "lips": ["dudak", "lip", "lips", "dudaklar"],
    "lip": ["lips", "dudak", "lip"],
    "kalp": ["heart", "kalp", "hearts"],
    "heart": ["kalp", "heart", "hearts"],
    "hearts": ["heart", "kalp"],
    "kelebek": ["butterfly", "kelebek", "butterflies"],
    "butterfly": ["kelebek", "butterfly", "butterflies"],
    "butterflies": ["butterfly", "kelebek"],
    "papatya": ["daisy", "papatya"],
    "daisy": ["papatya", "daisy"],
    "lale": ["tulip", "lale"],
    "tulip": ["lale", "tulip"],
    "orchid": ["orkide", "orchid"],
    "orkide": ["orchid", "orkide"],
    "şakayık": ["peony", "sakayik", "şakayık"],
    "sakayik": ["peony", "şakayık", "sakayik"],
    "peony": ["şakayık", "sakayik", "peony"],
    "ayçiçeği": ["sunflower", "aycicegi"],
    "sunflower": ["ayçiçeği", "aycicegi", "sunflower"],
    "puantiye": ["polka", "polka dot", "dot", "noktalı", "noktali", "benek", "puantiye"],
    "polka": ["puantiye", "polka dot", "dot", "noktalı", "noktali"],
    "polka dot": ["puantiye", "polka", "dot", "noktalı"],
    "noktalı": ["puantiye", "polka", "polka dot", "dot", "noktali"],
    "noktali": ["puantiye", "polka", "dot", "noktalı"],
    "dot": ["puantiye", "polka", "polka dot", "noktalı", "dot"],
    "mermer": ["marble", "mermer", "stone"],
    "marble": ["mermer", "marble"],
    "soyut": ["abstract", "soyut", "fluid", "watercolor"],
    "abstract": ["soyut", "abstract", "fluid art"],
    "ekose": ["plaid", "check", "checked", "ekose", "tartan", "gingham", "kareli"],
    "plaid": ["ekose", "check", "plaid", "tartan", "kareli"],
    "kareli": ["ekose", "plaid", "check", "tartan", "kareli"],
    "tartan": ["ekose", "plaid", "check", "tartan"],
    "check": ["ekose", "plaid", "tartan", "check", "kareli"],
    "çizgi": ["stripe", "striped", "çizgi", "line", "çizgili"],
    "çizgili": ["stripe", "striped", "çizgi", "çizgili"],
    "stripe": ["çizgi", "stripe", "striped", "çizgili"],
    "zebra": ["zebra", "zebra desen"],
    "snake": ["yılan", "snake", "python", "serpent", "yilan"],
    "yılan": ["snake", "python", "yilan", "yılan"],
    "yılan dokusu": ["snake skin", "snakeskin", "yilan dokusu"],
    "snake skin": ["yılan dokusu", "snakeskin", "snake skin"],
    "snakeskin": ["snake skin", "yılan dokusu"],
    "tiger": ["kaplan", "tiger"],
    "kaplan": ["tiger", "kaplan"],
    "paisley": ["şal", "paisley", "şal desen", "sal desen"],
    "şal": ["paisley", "şal desen", "sal", "scarf"],
    "dantel": ["lace", "dantel", "openwork", "crochet lace"],
    "lace": ["dantel", "lace", "openwork", "guipure"],
    "bordür": ["border", "bordür", "bordur", "scarf border"],
    "zincir": ["chain", "zincir", "chain print"],
    "barok": ["baroque", "versace", "barok", "baroque print"],
    "monogram": [
        "logo", "monogram", "lv", "louis vuitton", "logo repeat",
        "repeated logo", "luxury", "brand style",
    ],
    "logo": ["monogram", "logo repeat", "repeated logo", "emblem", "arma", "amblem"],
    "logo repeat": ["monogram", "logo", "repeated logo", "allover", "repeat"],
    "repeated logo": ["monogram", "logo", "logo repeat", "allover", "repeat"],
    "lv": ["louis vuitton", "louis", "vuitton", "monogram", "logo", "luxury", "logo repeat"],
    "louis": ["lv", "louis vuitton", "vuitton", "monogram"],
    "vuitton": ["lv", "louis vuitton", "louis", "monogram"],
    "louis vuitton": ["lv", "louis", "vuitton", "monogram"],
    "gg": ["gucci", "gg supreme"],
    "gucci": ["gg", "gg supreme"],
    "gg supreme": ["gg", "gucci"],
    "ysl": ["yves saint laurent", "saint laurent", "saint", "laurent"],
    "saint": ["ysl", "yves saint laurent", "saint laurent", "laurent"],
    "laurent": ["ysl", "yves saint laurent", "saint laurent", "saint"],
    "yves saint laurent": ["ysl", "saint", "laurent", "saint laurent"],
    "cd": ["christian dior", "dior"],
    "dior": ["cd", "christian dior"],
    "christian dior": ["cd", "dior"],
    "dg": ["dolce gabbana", "dolce", "gabbana"],
    "dolce": ["dg", "dolce gabbana", "gabbana"],
    "gabbana": ["dg", "dolce gabbana", "dolce"],
    "dolce gabbana": ["dg", "dolce", "gabbana"],
    "amiri": ["amiri", "amiri los angeles"],
    "ff": ["fendi"],
    "fendi": ["ff"],
    "mk": ["michael kors", "michael"],
    "michael kors": ["mk", "michael"],
    "michael": ["mk", "michael kors"],
    "tb": ["burberry", "nova check"],
    "burberry": ["tb", "nova check"],
    "nova check": ["tb", "burberry"],
    "celine": ["céline", "paris"],
    "paris": ["celine"],
    "yazili": ["yazi", "yazı", "text", "typography", "logo", "lettering", "text pattern"],
    "yazı": ["yazili", "yazi", "text", "typography", "lettering"],
    "yazi": ["yazili", "yazı", "text", "typography", "lettering"],
    "luxury": ["luks", "monogram", "logo", "brand style"],
    "luks": ["luxury", "monogram", "logo", "brand style"],
    "arma": ["emblem", "logo", "amblem", "patch", "symbol"],
    "amblem": ["emblem", "logo", "arma", "symbol"],
    "patch": ["arma", "emblem", "badge", "symbol"],
    "jacquard": ["jakar", "jacquard", "woven", "fabric texture"],
    "jakar": ["jacquard", "jakar", "woven", "fabric texture"],
    "metraj": ["allover", "repeat", "tekrar", "seamless"],
    "tekrar": ["repeat", "repeated", "allover", "metraj"],
    "kahverengi leopar": ["brown leopard", "tan leopard", "kahverengi leopard"],
    "brown leopard": ["kahverengi leopar", "tan leopard", "brown leopard"],
    "desen": ["pattern", "desen", "motif", "print"],
    "pattern": ["desen", "pattern", "motif"],
    "geometric": ["geometrik", "geo", "mosaic"],
    "geometrik": ["geometric", "geo"],
    "houndstooth": ["kazayağı", "houndstooth"],
    "jaguar": ["jaguar", "leopard", "animal print"],
    "cheetah": ["cheetah", "leopard", "animal print"],
    "panther": ["panther", "leopard", "animal print"],
    "ocelot": ["ocelot", "animal print"],
    "giraffe": ["giraffe", "animal print"],
    "crocodile": ["timsah", "crocodile", "animal print"],
    "timsah": ["timsah", "crocodile", "animal print"],
}

# Parent-group eşanlam kümeleri (hiyerarşik arama; tür adlarını karıştırmaz)
PARENT_SYNONYM_GROUPS: list[set[str]] = [
    {
        "cartoon",
        "çizgi film",
        "cizgi film",
        "çizgi film karakteri",
        "cizgi film karakteri",
        "cartoon character",
        "cartoon karakter",
    },
    {"flower", "floral", "çiçek", "cicek"},
    {"animal", "animal print", "hayvan", "hayvan deseni"},
]


def parent_synonym_keys(label: str) -> set[str]:
    """Parent etiketinin sıkı eşanlam anahtarları (normalize)."""
    raw = str(label or "").strip()
    if not raw:
        return set()
    want = normalize_turkish(raw)
    keys = {want}
    for group in PARENT_SYNONYM_GROUPS:
        norms = {normalize_turkish(x) for x in group}
        if want in norms or raw.lower() in {x.lower() for x in group}:
            keys |= norms
    return {k for k in keys if len(k) >= 2}

# Sorgu teriminden beklenen pattern_family / animal_print_type
TERM_FAMILY_HINTS: dict[str, dict[str, str]] = {
    "leopar": {"pattern_family": "animal_print", "animal_print_type": "leopard"},
    "leopard": {"pattern_family": "animal_print", "animal_print_type": "leopard"},
    "leo": {"pattern_family": "animal_print", "animal_print_type": "leopard"},
    "animal print": {"pattern_family": "animal_print"},
    "çiçek": {"pattern_family": "floral"},
    "cicek": {"pattern_family": "floral"},
    "flower": {"pattern_family": "floral"},
    "floral": {"pattern_family": "floral"},
    "küçük çiçek": {"pattern_family": "floral", "pattern_type": "small_floral"},
    "çıtır çiçek": {"pattern_family": "floral", "pattern_type": "ditsy_floral"},
    "minik çiçek": {"pattern_family": "floral", "pattern_type": "small_floral"},
    "small flower": {"pattern_family": "floral", "pattern_type": "small_floral"},
    "rose": {"pattern_family": "floral", "pattern_type": "rose"},
    "gül": {"pattern_family": "floral", "pattern_type": "rose"},
    "papatya": {"pattern_family": "floral", "pattern_type": "daisy"},
    "daisy": {"pattern_family": "floral", "pattern_type": "daisy"},
    "lale": {"pattern_family": "floral", "pattern_type": "tulip"},
    "tulip": {"pattern_family": "floral", "pattern_type": "tulip"},
    "orkide": {"pattern_family": "floral", "pattern_type": "orchid"},
    "orchid": {"pattern_family": "floral", "pattern_type": "orchid"},
    "şakayık": {"pattern_family": "floral", "pattern_type": "peony"},
    "sakayik": {"pattern_family": "floral", "pattern_type": "peony"},
    "peony": {"pattern_family": "floral", "pattern_type": "peony"},
    "ayçiçeği": {"pattern_family": "floral", "pattern_type": "sunflower"},
    "sunflower": {"pattern_family": "floral", "pattern_type": "sunflower"},
    "puantiye": {"pattern_family": "polka_dot", "pattern_type": "polka_dot"},
    "polka": {"pattern_family": "polka_dot", "pattern_type": "polka_dot"},
    "polka dot": {"pattern_family": "polka_dot", "pattern_type": "polka_dot"},
    "noktalı": {"pattern_family": "polka_dot", "pattern_type": "polka_dot"},
    "noktali": {"pattern_family": "polka_dot", "pattern_type": "polka_dot"},
    "dot": {"pattern_family": "polka_dot", "pattern_type": "polka_dot"},
    "mermer": {"pattern_family": "marble_abstract", "pattern_type": "marble"},
    "marble": {"pattern_family": "marble_abstract", "pattern_type": "marble"},
    "soyut": {"pattern_family": "marble_abstract", "pattern_type": "abstract_paint"},
    "abstract": {"pattern_family": "marble_abstract", "pattern_type": "abstract_paint"},
    "ekose": {"pattern_family": "plaid_check"},
    "plaid": {"pattern_family": "plaid_check"},
    "kareli": {"pattern_family": "plaid_check"},
    "tartan": {"pattern_family": "plaid_check"},
    "check": {"pattern_family": "plaid_check"},
    "çizgi": {"pattern_family": "stripe"},
    "çizgili": {"pattern_family": "stripe"},
    "stripe": {"pattern_family": "stripe"},
    "zebra": {"pattern_family": "animal_print", "animal_print_type": "zebra"},
    "snake": {"pattern_family": "animal_print", "animal_print_type": "snake"},
    "yılan": {"pattern_family": "animal_print", "animal_print_type": "snake"},
    "yılan dokusu": {"pattern_family": "animal_print", "animal_print_type": "snake"},
    "yılan derisi": {"pattern_family": "animal_print", "animal_print_type": "snake"},
    "tiger": {"pattern_family": "animal_print", "animal_print_type": "tiger"},
    "kaplan": {"pattern_family": "animal_print", "animal_print_type": "tiger"},
    "paisley": {"pattern_family": "paisley"},
    "şal": {"pattern_family": "paisley"},
    "dantel": {"pattern_family": "lace"},
    "lace": {"pattern_family": "lace"},
    "bordür": {"pattern_family": "scarf_border"},
    "zincir": {"pattern_family": "chain"},
    "barok": {"pattern_family": "baroque"},
    "monogram": {"pattern_family": "monogram_logo"},
    "logo repeat": {"pattern_family": "monogram_logo"},
    "repeated logo": {"pattern_family": "monogram_logo"},
    "lv": {"pattern_family": "monogram_logo"},
    "logo desen": {"pattern_family": "monogram_logo"},
    "arma": {"pattern_family": "monogram_logo"},
    "amblem": {"pattern_family": "monogram_logo"},
    "emblem": {"pattern_family": "monogram_logo"},
    "patch": {"pattern_family": "monogram_logo"},
    "yazi deseni": {"pattern_family": "typography_text"},
    "yazı deseni": {"pattern_family": "typography_text"},
    "text pattern": {"pattern_family": "typography_text"},
    "geometric": {"pattern_family": "geometric"},
    "geometrik": {"pattern_family": "geometric"},
    "jaguar":    {"pattern_family": "animal_print", "animal_print_type": "jaguar"},
    "cheetah":   {"pattern_family": "animal_print", "animal_print_type": "cheetah"},
    "panther":   {"pattern_family": "animal_print", "animal_print_type": "leopard"},
    "ocelot":    {"pattern_family": "animal_print", "animal_print_type": "ocelot"},
    "giraffe":   {"pattern_family": "animal_print", "animal_print_type": "giraffe"},
    "crocodile": {"pattern_family": "animal_print", "animal_print_type": "crocodile"},
    "timsah":    {"pattern_family": "animal_print", "animal_print_type": "crocodile"},
    "ahtapot": {"pattern_family": "unknown", "pattern_type": "illustration"},
    "octopus": {"pattern_family": "unknown", "pattern_type": "illustration"},
    "çizim": {"pattern_family": "unknown", "pattern_type": "illustration"},
    "illustration": {"pattern_family": "unknown", "pattern_type": "illustration"},
    "illüstrasyon": {"pattern_family": "unknown", "pattern_type": "illustration"},
    "kamuflaj": {"pattern_family": "texture_ground", "pattern_type": "camouflage"},
    "camo": {"pattern_family": "texture_ground", "pattern_type": "camouflage"},
    "camouflage": {"pattern_family": "texture_ground", "pattern_type": "camouflage"},
    "askeri camo": {
        "pattern_family": "texture_ground",
        "pattern_type": "military_camo",
    },
    "orman camo": {"pattern_family": "texture_ground", "pattern_type": "military_camo"},
    "zemin doku": {"pattern_family": "texture_ground", "pattern_type": "ground"},
    "ground": {"pattern_family": "texture_ground", "pattern_type": "ground"},
    "yazili": {"pattern_family": "typography_text", "pattern_type": "text_pattern"},
    "yazi": {"pattern_family": "typography_text", "pattern_type": "text_pattern"},
    "logo": {"pattern_family": "monogram_logo"},
    "luks": {"pattern_family": "monogram_logo"},
    "luxury": {"pattern_family": "monogram_logo"},
    "brand style": {"pattern_family": "monogram_logo"},
    "jacquard": {"pattern_family": "texture_ground", "pattern_type": "jacquard"},
    "jakar": {"pattern_family": "texture_ground", "pattern_type": "jacquard"},
    "askeri": {"pattern_family": "texture_ground", "pattern_type": "military_camo"},
    "gucci": {"pattern_family": "monogram_logo"},
    "louis vuitton": {"pattern_family": "monogram_logo"},
    "dior": {"pattern_family": "monogram_logo"},
    "burberry": {"pattern_family": "plaid_check"},
    "versace": {"pattern_family": "baroque"},
    "dolce gabbana": {"pattern_family": "baroque"},
    "amiri": {"pattern_family": "typography_text"},
}

# Hızlı seçim butonları — (görünen ad, pattern_family id)
FAMILY_UI_CHOICES: list[tuple[str, str]] = [
    ("Ekose / Kare", "plaid_check"),
    ("Çiçek (Floral)", "floral"),
    ("Leopard / Animal", "animal_print"),
    ("Çizgi (Stripe)", "stripe"),
    ("Geometrik", "geometric"),
    ("Paisley / Şal", "paisley"),
    ("Dantel", "lace"),
    ("Soyut / Mermer", "marble_abstract"),
    ("Çizim / İllüstrasyon", "unknown"),
    ("Düz / Plain", "plain"),
]

# Çelişen aileler — sorgu bir aileyi hedeflerken diğerini cezalandır
CONFLICTING_FAMILIES: dict[str, set[str]] = {
    "animal_print": {"floral", "plaid_check", "geometric", "plain", "marble_abstract", "polka_dot"},
    "floral": {"animal_print", "geometric", "polka_dot"},
    "geometric": {"floral", "animal_print", "marble_abstract"},
    "polka_dot": {"animal_print", "floral", "marble_abstract"},
    "plaid_check": {"floral", "animal_print"},
    "marble_abstract": {"animal_print", "floral"},
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def normalize_turkish(text: str) -> str:
    """Türkçe karakterleri ASCII'ye indir, küçük harf."""
    if not text:
        return ""
    lowered = text.strip().lower()
    lowered = lowered.translate(_TR_MAP)
    lowered = unicodedata.normalize("NFKD", lowered)
    return "".join(c for c in lowered if not unicodedata.combining(c))


def fuzzy_correct_query(query: str) -> tuple[str, bool]:
    """Yazım düzeltmesi yap; (düzeltilmiş, düzeltildi_mi) döndür."""
    try:
        from core.fuzzy_correct import correct_query
        res = correct_query(query)
        return res.corrected, res.was_corrected
    except Exception:
        return query, False


def expand_query_terms(query: str, *, include_semantic: bool = False) -> list[str]:
    """Sorguyu normalize edip eş anlamlılarla genişlet (yazım düzeltme dahil)."""
    base = (query or "").strip().lower()
    if not base:
        return []
    # Yazım düzeltme — corrected farklıysa her ikisini de kullan
    try:
        from core.fuzzy_correct import correct_query
        cr = correct_query(base)
        corrected_base = cr.corrected if cr.was_corrected else base
    except Exception:
        corrected_base = base
    norm = normalize_turkish(base)
    terms: set[str] = {base, norm}
    if corrected_base != base:
        terms.add(corrected_base)
        terms.add(normalize_turkish(corrected_base))
    if base in TERM_SYNONYMS:
        terms.update(TERM_SYNONYMS[base])
    if norm in TERM_SYNONYMS:
        terms.update(TERM_SYNONYMS[norm])
    if corrected_base in TERM_SYNONYMS:
        terms.update(TERM_SYNONYMS[corrected_base])
    for key, alts in TERM_SYNONYMS.items():
        nkey = normalize_turkish(key)
        if key in base or base in key or nkey in norm or norm in nkey:
            terms.add(key)
            terms.update(alts)
            for alt in alts:
                terms.add(normalize_turkish(alt))
    # Marka alias (lv ↔ louis vuitton, gg ↔ gucci, …)
    try:
        from core.brand_aliases import brand_alias_keys_for, expand_brand_terms

        for brand in expand_brand_terms(base):
            terms.add(brand)
            terms.add(normalize_turkish(brand))
            terms.update(brand_alias_keys_for(brand))
    except Exception:
        pass
    try:
        from core.auto_tags import expand_auto_tag_query

        for tag in expand_auto_tag_query(base):
            terms.add(tag)
            terms.add(normalize_turkish(tag))
            terms.add(tag.lower())
    except Exception:
        pass
    if include_semantic:
        from core.semantic_dictionary import expand_semantic_query
        for concept in expand_semantic_query(norm):
            terms.add(concept)
            terms.add(normalize_turkish(concept))
    try:
        from core.textile_knowledge_base import expand_knowledge_terms

        for t in expand_knowledge_terms(base):
            terms.add(t)
            terms.add(normalize_turkish(t))
        for t in expand_knowledge_terms(norm):
            terms.add(t)
            terms.add(normalize_turkish(t))
    except Exception:
        pass
    return sorted(terms, key=len, reverse=True)


def query_family_hints(query: str, *, include_semantic: bool = False) -> dict[str, str]:
    """Sorgudan beklenen desen ailesi ipuçları (typo düzeltmeli)."""
    base = (query or "").strip().lower()
    norm = normalize_turkish(base)
    # Yazım düzeltme: leoaprd → leopard → animal_print
    try:
        from core.fuzzy_correct import correct_query
        cr = correct_query(base)
        corrected = cr.corrected if cr.was_corrected else base
    except Exception:
        corrected = base
    corrected_n = normalize_turkish(corrected)
    try:
        from core.geometric_concepts import family_hints_for_query

        geo_hints = family_hints_for_query(base) or family_hints_for_query(norm) or family_hints_for_query(corrected)
        if geo_hints:
            return dict(geo_hints)
    except Exception:
        pass
    semantic_only = {
        "yazili", "yazi", "logo", "luks", "askeri", "gucci",
        "louis vuitton", "dior", "burberry", "versace", "dolce gabbana",
    }
    for candidate in dict.fromkeys([base, norm, corrected, corrected_n]):
        if candidate in TERM_FAMILY_HINTS and (include_semantic or candidate not in semantic_only):
            return dict(TERM_FAMILY_HINTS[candidate])
    for key, hints in TERM_FAMILY_HINTS.items():
        nkey = normalize_turkish(key)
        if (include_semantic or key not in semantic_only) and (
            key in base or nkey in norm or key in corrected or nkey in corrected_n
        ):
            return dict(hints)
    try:
        from core.textile_knowledge_base import knowledge_family_hints

        kb_hints = knowledge_family_hints(base) or knowledge_family_hints(norm)
        if kb_hints:
            return dict(kb_hints)
    except Exception:
        pass
    return {}


def resolve_user_family_label(text: str) -> dict[str, str]:
    """
    Kullanıcı metnini pattern_family + etiket bilgisine çevir.
    Örn: 'ekose' → plaid_check, 'ahtapot' → unknown + illustration tag.
    """
    raw = (text or "").strip()
    if not raw:
        return {}
    hints = query_family_hints(raw)
    out: dict[str, str] = {"tag": raw}
    if hints.get("pattern_family"):
        out["pattern_family"] = hints["pattern_family"]
    if hints.get("animal_print_type"):
        out["animal_print_type"] = hints["animal_print_type"]
    if hints.get("pattern_type"):
        out["pattern_subtype"] = hints["pattern_type"]
    return out


def tokenize(text: str) -> set[str]:
    norm = normalize_turkish(text)
    return {t for t in _TOKEN_SPLIT.split(norm) if len(t) >= 2}


def families_conflict(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return False
    return b in CONFLICTING_FAMILIES.get(a, set()) or a in CONFLICTING_FAMILIES.get(
        b, set()
    )
