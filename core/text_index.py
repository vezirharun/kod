"""Metin arama blob oluşturma ve skorlama."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.category_tree import is_category_descendant, resolve_category_query
from core.textile_terms import (
    expand_query_terms,
    families_conflict,
    normalize_turkish,
    query_family_hints,
    tokenize,
)
from core.texture_profile import TextureProfile


def build_text_search_blob(
    *,
    filename: str = "",
    path: str = "",
    customer: str = "",
    source_name: str = "",
    ocr_text: str = "",
    texture_map: dict[str, Any] | None = None,
    feedback_labels: list[str] | None = None,
    group_label: str = "",
    pattern_family: str = "",
    pattern_type: str = "",
    texture_family: str = "",
    pattern_subtype: str = "",
    category_path: str = "",
    category_aliases: list[str] | None = None,
    semantic_enabled: bool = False,
) -> str:
    """Indexleme sırasında normalize edilmiş arama blob'u."""
    prof = TextureProfile.from_dict(texture_map or {})
    pf = pattern_family or prof.pattern_family or ""
    pt = pattern_type or prof.animal_print_type or prof.texture_family or ""
    sub = pattern_subtype or prof.pattern_subtype or ""
    tf = texture_family or prof.texture_family or ""
    from core.semantic_tags import flatten_semantic_tags
    semantic_terms = (
        flatten_semantic_tags((texture_map or {}).get("semantic_tags"))
        if semantic_enabled else []
    )
    from core.pattern_dna import flatten_pattern_dna
    dna_terms = flatten_pattern_dna((texture_map or {}).get("pattern_dna"))
    from core.brand_aliases import enrich_ocr_text
    from core.color_index import flatten_color_index
    from core.auto_tags import flatten_auto_tags
    ocr_enriched = enrich_ocr_text(ocr_text) if ocr_text else ""
    color_terms = flatten_color_index((texture_map or {}).get("color_index"))
    auto_tag_terms = flatten_auto_tags(texture_map)
    family_tree = (texture_map or {}).get("pattern_family_tree") or {}
    family_tree_terms: list[str] = []
    if isinstance(family_tree, dict):
        for key in ("root", "branch", "color_branch", "scale_branch", "path", "label"):
            if family_tree.get(key):
                family_tree_terms.append(str(family_tree[key]))
        family_tree_terms.extend(str(x) for x in (family_tree.get("branches") or []) if x)
    if not color_terms and (texture_map or {}).get("color_family"):
        color_terms = [str(texture_map.get("color_family"))]
    if (texture_map or {}).get("dominant_palette"):
        color_terms = list(
            dict.fromkeys(
                color_terms + [str(x) for x in texture_map.get("dominant_palette") or []]
            )
        )
    parts = [
        filename,
        Path(path).parent.name if path else "",
        Path(path).stem if path else "",
        customer,
        source_name,
        ocr_enriched or ocr_text,
        pf,
        pt,
        sub,
        tf,
        prof.animal_print_type,
        prof.pattern_family,
        prof.texture_family,
        prof.color_family,
        group_label,
        category_path,
        " ".join(category_aliases or []),
        " ".join(feedback_labels or []),
        " ".join(semantic_terms),
        " ".join(dna_terms),
        " ".join(color_terms),
        " ".join(auto_tag_terms),
        " ".join(family_tree_terms),
    ]
    return normalize_turkish(" ".join(p for p in parts if p))


def extract_index_fields(texture_map: dict[str, Any] | None) -> tuple[str, str, str]:
    prof = TextureProfile.from_dict(texture_map or {})
    pf = prof.pattern_family or "unknown"
    pt = prof.animal_print_type or prof.texture_family or ""
    tf = prof.texture_family or ""
    return pf, pt, tf


def text_search_score(
    query: str,
    rec: dict[str, Any],
    *,
    semantic_enabled: bool = False,
    terms: list[str] | None = None,
    hints: dict[str, str] | None = None,
    parsed: Any | None = None,
    cat_match: Any | None = None,
    q_norm: str | None = None,
    terms_norm: list[str] | None = None,
    brand_needles: list[str] | None = None,
    clip_scores: dict[str, float] | None = None,
) -> tuple[float, dict[str, float], list[str]]:
    """
    Metin arama skoru ve kırılım.
    Klasör adı tek başına family değiştirmez — sadece düşük boost.
    Arama döngüsünde terms/hints/parsed bir kez üretilip geçirilmeli.
    """
    q = (query or "").strip()
    if not q:
        return 0.0, {}, []

    if terms is None:
        terms = expand_query_terms(q, include_semantic=semantic_enabled)
    if cat_match is None:
        cat_match = resolve_category_query(q)
        if cat_match.category_path:
            terms = list(dict.fromkeys(list(terms) + list(cat_match.aliases)))
    if hints is None:
        hints = query_family_hints(q, include_semantic=semantic_enabled)
    if q_norm is None:
        q_norm = normalize_turkish(q)
    if terms_norm is None:
        terms_norm = [normalize_turkish(t) for t in terms if str(t).strip()]
    # Ham sorgu + kısa terimler önce (uzun eşanlamlılar skorlamayı bozmasın)
    ordered_terms: list[str] = []
    for t in [q] + sorted(
        (str(x).strip() for x in terms if str(x).strip()),
        key=lambda s: (len(s.split()), len(s)),
    ):
        if t and t not in ordered_terms:
            ordered_terms.append(t)
    score_terms = ordered_terms[:12]
    if terms_norm is not None:
        full_map = {
            str(t).strip(): n
            for t, n in zip(terms, terms_norm)
            if str(t).strip()
        }
        score_norms = [full_map.get(t) or normalize_turkish(t) for t in score_terms]
    else:
        score_norms = [normalize_turkish(t) for t in score_terms]

    fname = rec.get("filename") or ""
    path = rec.get("path") or ""
    ocr = rec.get("ocr_text") or ""
    # OCR işi bitmeden (processed=0 ve metin boş/pending) "OCR eşleşmesi" yazma
    if int(rec.get("ocr_processed") or 0) == 0 and not str(ocr).strip():
        ocr = ""
    customer = rec.get("customer") or ""
    source = rec.get("source_name") or ""
    blob = rec.get("text_search_blob") or ""
    group_label = rec.get("group_label") or rec.get("pattern_group_label") or ""
    feedback_raw = rec.get("feedback_labels") or ""
    if isinstance(feedback_raw, list):
        feedback_text = " ".join(feedback_raw)
    else:
        feedback_text = str(feedback_raw)

    pf = rec.get("pattern_family") or ""
    pt = rec.get("pattern_type") or rec.get("pattern_subtype") or ""
    tf = rec.get("texture_family") or ""
    rec_category = str(
        rec.get("manual_category_path") or rec.get("category_path") or ""
    ).strip()
    texture_map = rec.get("texture_map") or {}
    if isinstance(texture_map, str):
        try:
            texture_map = json.loads(texture_map)
        except json.JSONDecodeError:
            texture_map = {}
    if not pf and texture_map:
        pf, pt, tf = extract_index_fields(texture_map)

    from core.semantic_tags import flatten_semantic_tags
    semantic_tags = texture_map.get("semantic_tags") or {} if semantic_enabled else {}
    semantic_terms = flatten_semantic_tags(semantic_tags) if semantic_enabled else []
    semantic_blob = normalize_turkish(" ".join(semantic_terms)) if semantic_terms else ""
    semantic_confidence = float(
        semantic_tags.get("confidence", 0) if isinstance(semantic_tags, dict) else 0
    )

    pattern_confidence = float(
        rec.get("pattern_confidence")
        or texture_map.get("classification_confidence", 0)
        or 0
    )
    visual_animal = float(texture_map.get("animal_score", 0) or 0)
    floral_score_value = float(texture_map.get("floral_score", 0) or 0)
    motif_coverage = float(texture_map.get("background_foreground_ratio", 1.0) or 0)
    scale_score = float(texture_map.get("scale_pattern_score", 0) or 0)
    open_floral_motif = (
        floral_score_value >= 0.35 and motif_coverage < 0.55 and scale_score < 0.82
    )
    animal_type = str(texture_map.get("animal_print_type") or "").strip()
    animal_family_reliable = pf != "animal_print" or (
        (
            pattern_confidence >= 0.70
            and visual_animal >= 0.70
            and not open_floral_motif
        )
        or bool(animal_type)
        or bool(texture_map.get("user_labeled"))
    )
    family_reliable = animal_family_reliable and (
        pf in ("", "unknown") or pattern_confidence >= 0.50
    )

    fname_l = fname.lower()
    stem = Path(fname).stem.lower()
    folder = Path(path).parent.name.lower() if path else ""
    stem_n = normalize_turkish(stem)
    fname_n = normalize_turkish(fname_l)
    # Blob yoksa ağır yeniden kurma — hafif alanlar yeterli
    if blob:
        blob_n = normalize_turkish(blob)
    else:
        blob_n = normalize_turkish(
            " ".join(
                p
                for p in (fname, folder, customer, source, ocr, pf, pt, tf, group_label)
                if p
            )
        )

    from core.brand_aliases import brand_match_score, query_brand_needles
    from core.natural_language_query import natural_language_score, parse_natural_query

    if parsed is None:
        parsed = parse_natural_query(q)
    if brand_needles is None:
        brand_needles = query_brand_needles(q)

    breakdown: dict[str, float] = {
        "filename_score": 0.0,
        "folder_score": 0.0,
        "family_score": 0.0,
        "texture_score": 0.0,
        "ocr_score": 0.0,
        "brand_alias_score": 0.0,
        "feedback_score": 0.0,
        "group_score": 0.0,
        "source_score": 0.0,
        "text_score": 0.0,
        "semantic_score": 0.0,
        "nl_score": 0.0,
        "auto_tag_score": 0.0,
        "family_reliable": 1.0 if family_reliable else 0.0,
    }
    reasons: list[str] = []

    # Dosya adı — yüksek (OCR yoksa birincil metin sinyali)
    for term, tnorm in zip(score_terms, score_norms):
        if q_norm == stem_n or tnorm == stem_n:
            breakdown["filename_score"] = max(breakdown["filename_score"], 0.95)
            reasons.append(f"Dosya adı: {stem}")
            break
        if tnorm in fname_n or term.lower() in fname_l:
            breakdown["filename_score"] = max(breakdown["filename_score"], 0.88)
            reasons.append(f"Dosya adı: {term}")
        q_tokens = tokenize(term)
        stem_tokens = tokenize(stem)
        if q_tokens and q_tokens & stem_tokens:
            overlap = len(q_tokens & stem_tokens) / max(len(q_tokens), 1)
            breakdown["filename_score"] = max(
                breakdown["filename_score"],
                0.82 + 0.10 * overlap,
            )
            if f"Dosya adı: {term}" not in reasons:
                reasons.append(f"Dosya adı: {term}")

    # Öğretilmiş / otomatik kategori yolu — yüksek öncelik
    if cat_match.category_path and rec_category:
        if (
            rec_category == cat_match.category_path
            or is_category_descendant(cat_match.category_path, rec_category)
            or is_category_descendant(rec_category, cat_match.category_path)
        ):
            breakdown["family_score"] = max(breakdown["family_score"], 0.92)
            reasons.append(f"Kategori: {rec_category}")
        elif normalize_turkish(cat_match.category_path) in normalize_turkish(
            rec_category
        ):
            breakdown["family_score"] = max(breakdown["family_score"], 0.88)
            reasons.append(f"Kategori eşleşmesi: {rec_category}")

    # Pattern / texture family — yüksek (yalnızca index alanları; klasör adı dahil değil)
    family_blob = normalize_turkish(f"{pf} {pt} {tf}") if family_reliable else ""
    pt_n = normalize_turkish(pt) if pt else ""
    for term, tnorm in zip(score_terms, score_norms):
        if tnorm and tnorm in family_blob:
            subtype_match = bool(pt_n and tnorm in pt_n)
            breakdown["family_score"] = max(
                breakdown["family_score"],
                0.86 if subtype_match else 0.78,
            )
            if pf and pf != "unknown":
                reasons.append(f"Doku ailesi: {pf}")
            elif pt:
                reasons.append(f"Desen tipi: {pt}")
            break
    if hints:
        hint_pf = hints.get("pattern_family", "")
        hint_pt = hints.get("pattern_type", "")
        if hint_pf and pf == hint_pf and family_reliable:
            skip_parent = (
                hint_pf == "geometric"
                and hint_pt
                and hint_pt not in ("", "geometric")
            )
            if not skip_parent:
                breakdown["family_score"] = max(breakdown["family_score"], 0.78)
                reasons.append(f"Doku ailesi: {hint_pf}")
        if (
            hint_pt
            and family_reliable
            and (pt == hint_pt or texture_map.get("animal_print_type") == hint_pt)
        ):
            breakdown["family_score"] = max(breakdown["family_score"], 0.86)
            reasons.append(f"Desen tipi: {hint_pt}")
        try:
            from core.geometric_concepts import dna_concept_alignment

            align, label = dna_concept_alignment(q, rec, texture_map)
            if align >= 0.86:
                breakdown["family_score"] = max(breakdown["family_score"], align)
                if label:
                    reasons.append(f"Geometrik kavram: {label}")
            elif align > 0:
                breakdown["family_score"] = max(breakdown["family_score"], min(align, 0.78))
        except Exception:
            pass

    # Texture map etiketleri
    if texture_map and family_reliable and breakdown["family_score"] < 0.85:
        prof = TextureProfile.from_dict(texture_map)
        apt_n = normalize_turkish(prof.animal_print_type)
        pf_n = normalize_turkish(prof.pattern_family)
        for term, tnorm in zip(score_terms, score_norms):
            if tnorm and tnorm in apt_n:
                breakdown["texture_score"] = max(breakdown["texture_score"], 0.86)
                reasons.append(f"Görsel etiket: {prof.animal_print_type}")
            if tnorm and tnorm in pf_n:
                generic_family_score = 0.68 if hints.get("pattern_type") else 0.84
                breakdown["texture_score"] = max(
                    breakdown["texture_score"],
                    generic_family_score,
                )
                reasons.append(f"Görsel etiket: {prof.pattern_family}")

    # OCR — ham metin yeterli (enrich_ocr aramada O(n) patlıyordu)
    ocr_n = normalize_turkish(ocr) if ocr else ""
    import re as _re
    for term, tnorm in zip(score_terms, score_norms):
        if not tnorm or not ocr_n:
            continue
        ocr_hit = tnorm in ocr_n
        if ocr_hit and len(tnorm) <= 4:
            ocr_hit = bool(
                _re.search(rf"(?<![a-z0-9]){_re.escape(tnorm)}(?![a-z0-9])", ocr_n)
            )
        if ocr_hit:
            breakdown["ocr_score"] = max(breakdown["ocr_score"], 0.98)
            reasons.append("OCR eşleşmesi")
            break
        # Marka iğnesi OCR'da
        if brand_needles and ocr_n:
            for kn in brand_needles[:8]:
                if kn and kn in ocr_n:
                    breakdown["ocr_score"] = max(breakdown["ocr_score"], 0.95)
                    reasons.append("OCR marka eşleşmesi")
                    break
            if breakdown["ocr_score"] >= 0.95:
                break

    # Genel `marka` sorgusu, arama motorundaki ortak marka kanıtıyla
    # aynı kümeden beslenir. Böylece belirli marka sorgusunda görünen bir
    # kayıt genel marka sorgusunda kaybolmaz.
    if cat_match.primary_family == normalize_turkish("Marka"):
        from core.brand_evidence import extract_brand_evidence
        from core.brand_aliases import resolve_brand_alias, normalize_brand_key

        ev = extract_brand_evidence(rec)
        if not cat_match.primary_subtype:
            if ev:
                breakdown["brand_alias_score"] = max(breakdown["brand_alias_score"], 0.90)
                breakdown["family_score"] = max(breakdown["family_score"], 0.90)
                reasons.append("Marka kanıtı")
        else:
            want = normalize_brand_key(
                resolve_brand_alias(cat_match.primary_subtype) or cat_match.primary_subtype
            )
            if want and want in ev:
                breakdown["brand_alias_score"] = max(breakdown["brand_alias_score"], 0.90)
                reasons.append("Marka kanıtı")

    # Marka alias (lv↔louis vuitton) — OCR / dosya adı / semantic
    brand_score = brand_match_score(
        q,
        [ocr, fname, " ".join(semantic_terms), blob],
        needles=brand_needles,
    )
    if brand_score > 0:
        breakdown["brand_alias_score"] = max(breakdown["brand_alias_score"], brand_score)
        reasons.append("Marka alias eşleşmesi")

    # Semantic tags are generated once while indexing and searched from FTS metadata.
    if semantic_blob and (semantic_confidence >= 0.45 or texture_map.get("user_labeled")):
        semantic_tokens = {normalize_turkish(value) for value in semantic_terms}
        brand_refs = {
            normalize_turkish(value)
            for value in (semantic_tags.get("brand_references", []) or [])
        } if isinstance(semantic_tags, dict) else set()
        for term, tnorm in zip(score_terms, score_norms):
            if not tnorm:
                continue
            if tnorm in brand_refs:
                breakdown["semantic_score"] = max(breakdown["semantic_score"], 0.80)
                reasons.append(f"Marka referansı: {term}")
                break
            if tnorm in semantic_tokens:
                breakdown["semantic_score"] = max(breakdown["semantic_score"], 0.90)
                reasons.append(f"AI etiketi: {term}")
            elif tnorm in semantic_blob:
                breakdown["semantic_score"] = max(breakdown["semantic_score"], 0.82)
                reasons.append(f"AI etiketi: {term}")

    # Feedback / pattern group — yüksek
    fb_n = normalize_turkish(feedback_text) if feedback_text else ""
    gl_n = normalize_turkish(group_label) if group_label else ""
    for term, tnorm in zip(score_terms, score_norms):
        if tnorm and tnorm in fb_n:
            breakdown["feedback_score"] = max(breakdown["feedback_score"], 0.88)
            reasons.append(f"Etiket: {term}")
        if tnorm and tnorm in gl_n:
            breakdown["group_score"] = max(breakdown["group_score"], 0.87)
            reasons.append(f"Pattern group: {group_label}")

    # Klasör — orta (family değiştirmez)
    folder_n = normalize_turkish(folder) if folder else ""
    for tnorm in score_norms:
        if tnorm and tnorm in folder_n:
            breakdown["folder_score"] = max(breakdown["folder_score"], 0.42)
            reasons.append(f"Klasör eşleşmesi: {folder}")
            break

    # Kaynak / müşteri — düşük/orta
    cust_n = normalize_turkish(customer) if customer else ""
    src_n = normalize_turkish(source) if source else ""
    for tnorm in score_norms:
        if tnorm and tnorm in cust_n:
            breakdown["source_score"] = max(breakdown["source_score"], 0.50)
        if tnorm and tnorm in src_n:
            breakdown["source_score"] = max(breakdown["source_score"], 0.48)

    # Blob genel eşleşme
    query_pf = hints.get("pattern_family", "")
    if query_pf and pf == query_pf and not family_reliable:
        blob_n = normalize_turkish(" ".join((fname, ocr, feedback_text, group_label)))
    for tnorm in score_norms:
        if tnorm and tnorm in blob_n:
            breakdown["text_score"] = max(breakdown["text_score"], 0.72)

    # Doğal dil bileşenleri (marka + renk + motif + repeat + stil + doku)
    if parsed.has_components:
        nl_final, nl_parts, nl_reasons = natural_language_score(
            parsed,
            rec,
            texture_map=texture_map,
            blob_n=blob_n,
            ocr=ocr,
            fname=fname,
        )
        if nl_final > 0:
            breakdown["nl_score"] = max(breakdown.get("nl_score", 0.0), nl_final)
            breakdown.update({k: v for k, v in nl_parts.items() if k.startswith("nl_")})
            reasons.extend(nl_reasons[:3])

    # AI otomatik etiketler (Floral, Luxury, Animal, …)
    from core.auto_tags import auto_tag_match_score, expand_auto_tag_query, flatten_auto_tags

    tag_score = auto_tag_match_score(q, texture_map)
    if tag_score > 0:
        breakdown["auto_tag_score"] = max(breakdown.get("auto_tag_score", 0.0), tag_score)
        tags = expand_auto_tag_query(q)
        if tags:
            reasons.append(f"Etiket: {tags[0]}")
    else:
        q_tags = expand_auto_tag_query(q)
        rec_tags = {str(t).lower() for t in flatten_auto_tags(texture_map)}
        for tag in q_tags:
            if tag.lower() in rec_tags:
                breakdown["auto_tag_score"] = max(breakdown.get("auto_tag_score", 0.0), 0.88)
                reasons.append(f"Etiket: {tag}")
                break

    final = max(
        breakdown["filename_score"],
        breakdown["family_score"] * 0.98,
        breakdown["texture_score"],
        breakdown["ocr_score"],
        breakdown["brand_alias_score"],
        breakdown["feedback_score"],
        breakdown["group_score"],
        breakdown["text_score"],
        breakdown["semantic_score"],
        breakdown.get("nl_score", 0.0),
        breakdown.get("auto_tag_score", 0.0),
        breakdown["folder_score"] * 0.85,
        breakdown["source_score"] * 0.80,
        0.0,
    )

    # Çelişen aile cezası
    if query_pf and pf and families_conflict(query_pf, pf):
        if breakdown["family_score"] < 0.5 and breakdown["filename_score"] < 0.7:
            final *= 0.35

    # Küçük çiçek boost
    if any(t in q_norm for t in ("kucuk cicek", "mini", "small flower", "minik")):
        if "small" in blob_n or "mini" in blob_n or "kucuk" in blob_n:
            final = min(1.0, final + 0.06)

    # Klasör tek başına yüksek skor vermesin
    if (
        breakdown["folder_score"] > 0
        and breakdown["family_score"] < 0.4
        and breakdown["filename_score"] < 0.4
    ):
        final = min(final, 0.45 + breakdown["folder_score"] * 0.55)

    from core.query_evidence import (
        apply_query_evidence,
        build_search_plan,
        collect_candidate_evidence,
    )
    from core.pattern_intelligence_v2 import parse_pattern_query_v2

    v2q = None
    try:
        v2q = parse_pattern_query_v2(q)
    except Exception:
        v2q = None
    plan = build_search_plan(q, parsed=parsed, v2q=v2q, hints=hints)
    ev_report = collect_candidate_evidence(
        plan,
        rec,
        texture_map=texture_map,
        breakdown=breakdown,
        clip_scores=clip_scores,
    )
    # `marka` bir nesne sorgusu değil, üst kategori komutudur. Güçlü marka
    # kanıtı bulunan kaydı görsel-nesne kanıtı yok diye aşağı çekme; aksi halde
    # `dior`/`amiri` sonuçları `marka` listesinden kaybolur.
    is_general_brand = (
        cat_match.primary_family == normalize_turkish("Marka")
        and not cat_match.primary_subtype
        and breakdown.get("brand_alias_score", 0.0) >= 0.90
    )
    if is_general_brand:
        final = max(float(final), 0.90)
        ev_report.delta = 0.0
        ev_report.contradiction_penalty = 0.0
        ev_report.confidence = "HIGH"
    else:
        final, ev_report = apply_query_evidence(final, plan, ev_report)
    breakdown["query_evidence"] = ev_report.query_evidence
    breakdown["composite_evidence"] = ev_report.composite_evidence
    breakdown["contradiction_penalty"] = ev_report.contradiction_penalty
    breakdown["query_evidence_delta"] = ev_report.delta
    breakdown["query_evidence_report"] = ev_report.to_dict()
    breakdown["search_evidence_plan"] = plan.to_dict()

    breakdown["final_score"] = round(final, 4)
    unique_reasons = list(dict.fromkeys(reasons))[:4]
    return final, breakdown, unique_reasons
