"""Tasarımcı odaklı etiketler, rozetler ve benzerlik ikonları."""

from __future__ import annotations

from typing import Any

# Display provenance — search ranking / learning stores are never mutated here.
SRC_USER_TAUGHT = "USER-TAUGHT"
SRC_AI_DETECTED = "AI-DETECTED"
SRC_DNA = "DNA"
SRC_CATEGORY = "CATEGORY"
SRC_QUERY_MATCH = "QUERY-MATCH"
SRC_COLOR = "COLOR"

# Progressive / search-stage display (internal English keys → designer Turkish)
STAGE_UI_LABELS: dict[str, str] = {
    "Exact": "Tam Eşleşme",
    "Pattern Family": "Aynı Desen Ailesi",
    "Semantic": "Benzer Stil",
    "Texture": "Benzer Doku",
    "Deep Search": "Benzer Stil",
    "Hızlı Sonuç": "Tam Eşleşme",
    "Tam Analiz": "Benzer Stil",
    "AI ile Güncellendi": "Benzer Stil",
}

# Similarity reason → (icon, short label)
REASON_ICONS: list[tuple[str, tuple[str, ...]]] = [
    ("◆", ("tam eşleşme", "exact", "aynı dosya", "kişisel", "protected")),
    ("▣", ("desen ailesi", "pattern family", "aile", "family", "animal")),
    ("≈", ("doku", "texture", "patch", "repeat", "motif")),
    ("🎨", ("renk", "color", "palette", "ton")),
    ("◇", ("stil", "style", "semantic", "dna", "tasarım")),
]

FAMILY_LABELS_TR: dict[str, str] = {
    "animal_print": "Hayvan Deseni",
    "floral": "Floral",
    "geometric": "Geometrik",
    "stripe": "Çizgi",
    "polka_dot": "Puantiyé",
    "abstract": "Soyut",
    "marble_abstract": "Mermer / Soyut",
    "monogram_logo": "Monogram",
    "plaid_check": "Ekose",
    "paisley": "Şal",
    "unknown": "",
}

COLOR_LABELS_TR: dict[str, str] = {
    "brown_tan": "Kahve / Bej",
    "beige": "Bej",
    "cream": "Krem",
    "camel": "Camel",
    "black_white": "Siyah-Beyaz",
    "grayscale": "Gri",
    "navy_blue": "Lacivert",
    "blue": "Mavi",
    "red": "Kırmızı",
    "red_pink": "Kırmızı / Pembe",
    "burgundy": "Bordo",
    "green": "Yeşil",
    "khaki": "Haki",
    "yellow": "Sarı",
    "gold": "Altın",
    "orange": "Turuncu",
    "multicolor": "Çok Renkli",
    "neon_multicolor": "Çok Renkli",
    "pastel": "Pastel",
    "neutral": "Nötr",
    "unknown": "",
}


def stage_label(stage: str) -> str:
    key = str(stage or "").strip()
    return STAGE_UI_LABELS.get(key, key or "Tam Eşleşme")


def family_badge_text(pattern_family: str, animal_print_type: str = "") -> str:
    pf = str(pattern_family or "").strip()
    if not pf or pf == "unknown":
        return ""
    if pf == "animal_print" and animal_print_type:
        return str(animal_print_type).replace("_", " ").title()
    return FAMILY_LABELS_TR.get(pf, pf.replace("_", " ").title())


_MAIN_AUTO_FAMILIES = {
    "animal_print",
    "floral",
    "geometric",
    "plaid_check",
    "paisley",
    "abstract",
    "marble_abstract",
    "monogram_logo",
}


def _pretty_taught(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:]


def resolve_display_query_text(result: Any = None, query_text: str = "") -> str:
    """Kart/inspector için aktif sorgu metni (arg > debug.query_meaning)."""
    q = str(query_text or "").strip()
    if q:
        return q
    dbg = getattr(result, "debug", None) or {}
    if not isinstance(dbg, dict):
        return ""
    qm = dbg.get("query_meaning")
    if isinstance(qm, dict):
        for key in ("normalized", "concept_core"):
            val = str(qm.get(key) or "").strip()
            if val:
                return val
    return str(dbg.get("text_query") or dbg.get("query_text") or "").strip()


def _norm_token_set(text: str) -> set[str]:
    from core.textile_terms import expand_query_terms, normalize_turkish, tokenize

    raw = str(text or "").strip()
    if not raw:
        return set()
    out: set[str] = set(tokenize(raw))
    out.add(normalize_turkish(raw))
    for part in raw.replace("/", " ").replace("-", " ").split():
        part = part.strip()
        if len(part) < 2:
            continue
        out.add(normalize_turkish(part))
        try:
            for syn in expand_query_terms(part):
                n = normalize_turkish(syn)
                if len(n) >= 2:
                    out.add(n)
        except Exception:
            pass
    return {t for t in out if len(t) >= 2}


def label_aligned_with_query(label: str, query: str) -> bool:
    """Genel eşleşme: TR/EN alias + aile ipucu. Özel blacklist yok."""
    q = str(query or "").strip()
    if not q:
        return True
    lab = str(label or "").strip()
    if not lab:
        return False
    if _norm_token_set(q) & _norm_token_set(lab):
        return True
    try:
        from core.textile_terms import families_conflict, query_family_hints

        qh = query_family_hints(q)
        lh = query_family_hints(lab)
        qa = str(qh.get("animal_print_type") or "").strip()
        la = str(lh.get("animal_print_type") or "").strip()
        if qa and la:
            return qa == la
        qf = str(qh.get("pattern_family") or "").strip()
        lf = str(lh.get("pattern_family") or "").strip()
        if qf and lf:
            if families_conflict(qf, lf):
                return False
            return qf == lf
    except Exception:
        pass
    return False


def _raw_taught_concept(result: Any) -> str:
    """Dosyada duran öğretilmiş/kanıtlı kavram — sorgu filtresi yok."""
    dbg = getattr(result, "debug", None) or {}
    if not isinstance(dbg, dict):
        return ""
    for key in ("learned_canonical", "learned_concept_label"):
        val = str(dbg.get(key) or "").strip()
        if val:
            return _pretty_taught(val)
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    if dbg.get("learned_concept_exact") or dbg.get("learned_concept") or dbg.get(
        "user_labeled"
    ) or tm.get("user_labeled"):
        path = str(
            dbg.get("manual_category_path")
            or dbg.get("category_path")
            or tm.get("manual_category_path")
            or tm.get("category_path")
            or ""
        )
        child = path.split("/")[-1].strip() if path else ""
        if child:
            return _pretty_taught(child)
    return ""


def taught_concept_label(result: Any, query_text: str = "") -> str:
    """Öğreti etiketi; sorgu varken yalnızca sorguyla hizalıysa göster."""
    taught = _raw_taught_concept(result)
    if not taught:
        return ""
    query = resolve_display_query_text(result, query_text)
    if query and not label_aligned_with_query(taught, query):
        return ""
    return taught


def _auto_family_label(result: Any) -> str:
    return family_badge_text(
        getattr(result, "pattern_family", "") or "",
        getattr(result, "animal_print_type", "") or "",
    )


def _auto_family_aligned(result: Any, query: str) -> bool:
    if not query:
        return True
    auto = _auto_family_label(result)
    if auto and label_aligned_with_query(auto, query):
        return True
    try:
        from core.textile_terms import query_family_hints

        hints = query_family_hints(query)
    except Exception:
        hints = {}
    pf = str(getattr(result, "pattern_family", "") or "").strip()
    at = str(getattr(result, "animal_print_type", "") or "").strip()
    qf = str(hints.get("pattern_family") or "").strip()
    qa = str(hints.get("animal_print_type") or "").strip()
    if qf and pf == qf:
        if qf == "animal_print" and qa:
            return (not at) or at == qa or label_aligned_with_query(at, qa)
        return True
    if qa and at and (at == qa or label_aligned_with_query(at, qa)):
        return True
    return False


def _query_hit_signal(result: Any, query: str) -> bool:
    """Bu satır mevcut sorgu için gerçekten aday mı (ranking değil, display)."""
    if not query:
        return False
    dbg = getattr(result, "debug", None) or {}
    if not isinstance(dbg, dict):
        dbg = {}
    if dbg.get("learned_concept_exact") or dbg.get("learned_concept"):
        can = str(dbg.get("learned_canonical") or "").strip()
        return (not can) or label_aligned_with_query(can, query)
    if dbg.get("object_index_hit") or dbg.get("visual_concept_detector"):
        return True
    if dbg.get("open_vocab_object"):
        return True
    if getattr(result, "same_pattern_family", False) or getattr(
        result, "same_animal_family", False
    ):
        return True
    bd = getattr(result, "breakdown", None) or {}
    if not isinstance(bd, dict):
        bd = {}
    for key in ("learned_concept", "nl_score", "text", "fts", "concept"):
        if float(bd.get(key, 0) or 0) >= 0.45:
            return True
    if _auto_family_aligned(result, query):
        return True
    return float(getattr(result, "score", 0) or 0) >= 0.55


def _query_match_display_label(result: Any, query: str) -> str:
    if not query or not _query_hit_signal(result, query):
        return ""
    dbg = getattr(result, "debug", None) or {}
    if isinstance(dbg, dict):
        can = str(dbg.get("learned_canonical") or "").strip()
        if can and label_aligned_with_query(can, query):
            return _pretty_taught(can)
        qm = dbg.get("query_meaning") if isinstance(dbg.get("query_meaning"), dict) else {}
        core = str(qm.get("concept_core") or "").strip()
        if core and label_aligned_with_query(core, query):
            return _pretty_taught(core)
    try:
        from core.textile_terms import query_family_hints

        hints = query_family_hints(query)
    except Exception:
        hints = {}
    qf = str(hints.get("pattern_family") or "").strip()
    if qf:
        return family_badge_text(qf, str(hints.get("animal_print_type") or ""))
    # Serbest kavram (dudak/gül/kalp): sorgunun anlamlı gövdesi
    parts = [p for p in query.replace("/", " ").split() if len(p) >= 2]
    # Renk kelimelerini atla; kavramı bul
    colorish = {
        "siyah",
        "beyaz",
        "krem",
        "kahve",
        "kirmizi",
        "kırmızı",
        "mavi",
        "yesil",
        "yeşil",
        "black",
        "white",
        "cream",
        "red",
        "blue",
        "green",
    }
    from core.textile_terms import normalize_turkish

    for p in reversed(parts):
        if normalize_turkish(p) in colorish:
            continue
        return _pretty_taught(p)
    return _pretty_taught(parts[-1]) if parts else ""


def family_badge_for_result(result: Any, query_text: str = "") -> str:
    """Ana kart etiketi — sorgu bağlamında provenance filtresi uygulanır."""
    query = resolve_display_query_text(result, query_text)
    taught = taught_concept_label(result, query)
    if taught:
        return taught
    auto = _auto_family_label(result)
    if not query:
        # Geriye uyum: sorgu yoksa öğreti > DNA (mevcut teach-authority testleri)
        raw = _raw_taught_concept(result)
        if raw:
            return raw
        return auto
    if auto and _auto_family_aligned(result, query):
        return auto
    return _query_match_display_label(result, query)


def independent_feature_badge(result: Any, query_text: str = "") -> str:
    """Öğretiyle çelişmeyen ikincil özellik (ör. Çizgi). Ana aile/Ekose değil."""
    query = resolve_display_query_text(result, query_text)
    taught = taught_concept_label(result, query) or (
        _raw_taught_concept(result) if not query else ""
    )
    if not taught:
        return ""
    pf = str(getattr(result, "pattern_family", "") or "").strip()
    if pf in _MAIN_AUTO_FAMILIES or not pf:
        return ""
    auto = family_badge_text(pf, getattr(result, "animal_print_type", "") or "")
    if not auto:
        return ""
    if auto.casefold() in taught.casefold() or taught.casefold() in auto.casefold():
        return ""
    if query and not label_aligned_with_query(auto, query):
        # İkincil özellik sorguyla çelişmesin; çizgi gibi nötr aileler serbest
        if pf not in ("stripe", "polka_dot"):
            return ""
    return auto


def color_badge_text(color_family: str) -> str:
    cf = str(color_family or "").strip()
    if not cf or cf == "unknown":
        return ""
    return COLOR_LABELS_TR.get(cf, cf.replace("_", " ").title())


def color_badge_for_result(result: Any, query_text: str = "") -> str:
    """Renk rozeti: TextureProfile/DNA color_family; free-text sızıntı yok."""
    _ = query_text  # reserved — renk kanıtı sorgudan bağımsız gösterilebilir
    cf = str(getattr(result, "color_family", "") or "").strip()
    if not cf or cf == "unknown":
        dbg = getattr(result, "debug", None) or {}
        if isinstance(dbg, dict):
            cf = str(dbg.get("color_family") or "").strip()
            tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
            if (not cf or cf == "unknown") and tm:
                cf = str(tm.get("color_family") or "").strip()
    return color_badge_text(cf)


def filter_explanations_for_query(
    explanations: list[str] | None,
    result: Any,
    query_text: str = "",
) -> list[str]:
    """'Neden benzer?' satırlarından sorguyla ilgisiz learned sızıntısını çıkar."""
    query = resolve_display_query_text(result, query_text)
    rows = [str(x).strip() for x in (explanations or []) if str(x).strip()]
    if not query:
        return rows
    taught_raw = _raw_taught_concept(result)
    taught_ok = bool(taught_raw and label_aligned_with_query(taught_raw, query))
    out: list[str] = []
    for line in rows:
        low = line.casefold()
        if "öğrenilmiş kavram" in low or "ogrenilmis kavram" in low:
            if not taught_ok:
                continue
        # Satırda geçen yabancı kavram adı (Leopard vb.) sorguyla hizasızsa at
        if taught_raw and not taught_ok and taught_raw.casefold() in low:
            continue
        out.append(line)
    return out


def build_result_display_labels(
    result: Any,
    query_text: str = "",
) -> dict[str, Any]:
    """Merkezi display enrichment — her etiket için provenance."""
    query = resolve_display_query_text(result, query_text)
    labels: list[dict[str, Any]] = []

    fam = family_badge_for_result(result, query)
    if fam:
        raw = _raw_taught_concept(result)
        if raw and label_aligned_with_query(raw, query or raw):
            src = SRC_USER_TAUGHT
        elif _auto_family_aligned(result, query) if query else bool(_auto_family_label(result)):
            src = SRC_DNA
        else:
            src = SRC_QUERY_MATCH
        labels.append(
            {
                "label": fam,
                "source": src,
                "confidence": 1.0 if src == SRC_USER_TAUGHT else 0.85,
                "matched_query": bool(query and label_aligned_with_query(fam, query)),
                "evidence": src,
            }
        )

    extra = independent_feature_badge(result, query)
    if extra:
        labels.append(
            {
                "label": extra,
                "source": SRC_DNA,
                "confidence": 0.7,
                "matched_query": False,
                "evidence": "independent_feature",
            }
        )

    color = color_badge_for_result(result, query)
    if color:
        labels.append(
            {
                "label": color,
                "source": SRC_COLOR,
                "confidence": 0.8,
                "matched_query": False,
                "evidence": "color_family",
            }
        )

    brand = brand_badge_text(result)
    if brand:
        labels.append(
            {
                "label": brand,
                "source": SRC_AI_DETECTED,
                "confidence": 0.6,
                "matched_query": False,
                "evidence": "brand",
            }
        )

    return {
        "query": query,
        "family": fam,
        "extra": extra,
        "color": color,
        "brand": brand,
        "labels": labels,
        "display_labels": [x["label"] for x in labels],
    }


def brand_badge_text(result: Any) -> str:
    dbg = getattr(result, "debug", None) or {}
    tm = dbg.get("texture_map") if isinstance(dbg, dict) else {}
    if not isinstance(tm, dict):
        tm = {}
    for key in (
        "brand",
        "detected_brand",
        "brand_name",
        "ocr_brand",
        "brand_style",
        "designer_style",
    ):
        val = tm.get(key) or (dbg.get(key) if isinstance(dbg, dict) else None)
        if val and str(val).strip() and str(val).strip().lower() not in ("unknown", "none"):
            return str(val).strip()[:28]
    tree = tm.get("pattern_family_tree")
    if isinstance(tree, dict):
        root = str(tree.get("root") or "").strip()
        if root and root.lower() not in ("unknown", "floral", "animal_print"):
            return root[:28]
    return ""


def reason_chips(explanations: list[str], *, limit: int = 4) -> list[tuple[str, str]]:
    """Return list of (icon, short_text) for designer-facing reason chips."""
    chips: list[tuple[str, str]] = []
    for raw in explanations or []:
        text = str(raw or "").strip()
        if not text:
            continue
        low = text.lower()
        icon = "•"
        for candidate_icon, keys in REASON_ICONS:
            if any(k in low for k in keys):
                icon = candidate_icon
                break
        short = text if len(text) <= 36 else text[:33] + "…"
        chips.append((icon, short))
        if len(chips) >= limit:
            break
    return chips


def similarity_tier_label(result: Any) -> str:
    """Map result strength to designer tier label."""
    if getattr(result, "is_self_match", False) or (
        isinstance(getattr(result, "debug", None), dict)
        and result.debug.get("protected_exact")
    ):
        return "Tam Eşleşme"
    if getattr(result, "same_pattern_family", False) or getattr(
        result, "same_animal_family", False
    ):
        return "Aynı Desen Ailesi"
    score = float(getattr(result, "score", 0) or 0)
    tex = float((getattr(result, "breakdown", None) or {}).get("texture", 0) or 0)
    if tex >= 0.45 or float(getattr(result, "texture_family_score", 0) or 0) >= 0.45:
        return "Benzer Doku"
    if score >= 0.55:
        return "Benzer Stil"
    return "Benzer Stil"
