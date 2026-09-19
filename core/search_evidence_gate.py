"""Text-search evidence gate — similarity is not presence.

Read-only ranking overlay. Does not write index/FAISS/DINO/OpenCLIP.
Lane A (pattern DNA) is not filtered here.
"""
from __future__ import annotations

import re
from typing import Any

from core.textile_terms import normalize_turkish

OBJECT_TOP_N = 50
EMPTY_BRAND = "Marka kanıtı bulunamadı"

# Lane A — DESEN/KAVRAM (do not gate CLIP; Pattern DNA owns ranking).
_PATTERN_LANE = frozenset(
    {
        "cicek", "floral", "flower", "gul", "rose",
        "ekose", "kareli", "tartan", "plaid", "check",
        "cizgi", "cizgili", "stripe", "striped",
        "yaprak", "leaf", "leaves", "foliage",
        "geometrik", "geometric",
        "yilan", "snake", "serpent",
        "leopar", "leopard", "leo",
        "zebra",
        "kamuflaj", "camouflage", "camo",
        "paisley",
        "barok", "baroque", "rokoko", "rococo",
        "suluboya", "watercolor",
        "etnik", "ethnic_print",
        "puantiye", "damask",
        "firca", "brushstroke", "sicrama", "paint_splatter", "splatter",
        "vintage", "retro",
    }
)

# Lane B — GERÇEK NESNE. CLIP “looks like” is not presence.
_PERSON_LANE = {
    "kadin": ("woman", "Kadın"),
    "kadın": ("woman", "Kadın"),
    "woman": ("woman", "Kadın"),
    "women": ("woman", "Kadın"),
    "female": ("woman", "Kadın"),
    "bayan": ("woman", "Kadın"),
    "erkek": ("man", "Erkek"),
    "man": ("man", "Erkek"),
    "men": ("man", "Erkek"),
    "male": ("man", "Erkek"),
    "insan": ("person", "İnsan"),
    "kisi": ("person", "İnsan"),
    "kişi": ("person", "İnsan"),
    "person": ("person", "İnsan"),
    "yuz": ("face", "Yüz"),
    "yüz": ("face", "Yüz"),
    "face": ("face", "Yüz"),
    "cocuk": ("child", "Çocuk"),
    "çocuk": ("child", "Çocuk"),
    "child": ("child", "Çocuk"),
}

_OBJECT_LANE = {
    "kopek": ("dog", "Köpek"),
    "köpek": ("dog", "Köpek"),
    "dog": ("dog", "Köpek"),
    "kedi": ("cat", "Kedi"),
    "cat": ("cat", "Kedi"),
    "kus": ("bird", "Kuş"),
    "kuş": ("bird", "Kuş"),
    "bird": ("bird", "Kuş"),
    "tavsan": ("rabbit", "Tavşan"),
    "tavşan": ("rabbit", "Tavşan"),
    "rabbit": ("rabbit", "Tavşan"),
    "kaplan": ("tiger", "Kaplan"),
    "tiger": ("tiger", "Kaplan"),
    "bisiklet": ("bicycle", "Bisiklet"),
    "bicycle": ("bicycle", "Bisiklet"),
    "bike": ("bicycle", "Bisiklet"),
    "araba": ("car", "Araba"),
    "otomobil": ("car", "Araba"),
    "car": ("car", "Araba"),
    "canta": ("handbag", "Çanta"),
    "çanta": ("handbag", "Çanta"),
    "handbag": ("handbag", "Çanta"),
    "bag": ("handbag", "Çanta"),
    "ayakkabi": ("shoe", "Ayakkabı"),
    "ayakkabı": ("shoe", "Ayakkabı"),
    "shoe": ("shoe", "Ayakkabı"),
    "kiraz": ("cherry", "Kiraz"),
    "cherry": ("cherry", "Kiraz"),
    "cilek": ("strawberry", "Çilek"),
    "çilek": ("strawberry", "Çilek"),
    "strawberry": ("strawberry", "Çilek"),
    "elma": ("apple", "Elma"),
    "apple": ("apple", "Elma"),
    "taki": ("jewelry", "Takı"),
    "takı": ("jewelry", "Takı"),
    "jewelry": ("jewelry", "Takı"),
    "mucevher": ("jewelry", "Takı"),
    "mücevher": ("jewelry", "Takı"),
    "kolye": ("jewelry", "Takı"),
    "necklace": ("jewelry", "Takı"),
    "kupe": ("jewelry", "Takı"),
    "küpe": ("jewelry", "Takı"),
    "earring": ("jewelry", "Takı"),
    "earrings": ("jewelry", "Takı"),
    "yuzuk": ("jewelry", "Takı"),
    "yüzük": ("jewelry", "Takı"),
    "ring": ("jewelry", "Takı"),
    "bileklik": ("jewelry", "Takı"),
    "bracelet": ("jewelry", "Takı"),
    "bros": ("jewelry", "Takı"),
    "broş": ("jewelry", "Takı"),
    "brooch": ("jewelry", "Takı"),
    "madalyon": ("jewelry", "Takı"),
    "pendant": ("jewelry", "Takı"),
    "karga": ("crow", "Karga"),
    "crow": ("crow", "Karga"),
    "kartal": ("eagle", "Kartal"),
    "eagle": ("eagle", "Kartal"),
    "kelebek": ("butterfly", "Kelebek"),
    "butterfly": ("butterfly", "Kelebek"),
    "dudak": ("lips", "Dudak"),
    "lips": ("lips", "Dudak"),
    "fare": ("mouse", "Fare"),
    "mouse": ("mouse", "Fare"),
    "timsah": ("crocodile", "Timsah"),
    "crocodile": ("crocodile", "Timsah"),
}

_GENERIC_BRAND = frozenset({"marka", "brand", "brands", "logo"})

_CLIP_KINDS = frozenset({"openclip", "clip", "open"})
_HARD_ENTITY_KINDS = frozenset({"object_detector", "fusion", "visual_concept"})

ZERO_SHOT_TYPE = "ZERO_SHOT_VISUAL"
ZERO_SHOT_SOURCE = "openclip"
OPEN_VOCAB_TYPE = "OPEN_VOCAB_OBJECT"
# Display/gate floor. CLIP retrieve still uses CLIP_FLOOR (0.20).
# Live FAISS on this archive compresses all prompts into ~0.27–0.31.
# 0.28 keeps the top tail and drops the 80-wide fabric haze.
ZERO_SHOT_FLOOR = 0.28
ZERO_SHOT_TOP_N = 12
# CLIP target must beat rivals by this margin (GDino confidence is not used).
ZERO_SHOT_MARGIN = 0.015
ZERO_SHOT_FRUIT_TEXTILE_GAP = 0.05

_ZS_PROMPT = {
    "lips": "a close-up photograph of human lips and mouth, not a textile print",
    "mouth": "a close-up photograph of human lips and mouth, not a textile print",
    "cherry": "a photograph of a cherry fruit, not cherry blossom and not a floral textile",
    "strawberry": "a photograph of a strawberry fruit, not a flower and not a floral textile",
    "flower": "a photograph of a flower bloom with petals, not a strawberry or cherry fruit",
    "jewelry": "fashion jewelry product photo, necklace bracelet earrings ring, not a baroque fabric ornament",
    "crow": "a photograph of a crow bird, corvus",
    "crocodile": "a photograph of a crocodile",
    "butterfly": "a photograph of a butterfly",
    "eagle": "a photograph of an eagle bird",
    "sparrow": "a photograph of a sparrow bird",
    "lion": "a photograph of a lion",
    "elephant": "a photograph of an elephant",
}
_ZS_RIVALS = {
    "crow": ("butterfly", "cherry"),
    "butterfly": ("crow", "cherry"),
    "jewelry": ("cherry", "crow", "lips"),
    "lips": ("butterfly", "crow"),
    "cherry": ("butterfly", "jewelry", "flower"),
    "strawberry": ("butterfly", "jewelry", "flower"),
}
_ZS_TEXTILE_PROMPT = {
    "floral": "a repeating floral textile print, flowers on fabric, not a real fruit",
    "animal_print": "an animal print textile, leopard or zebra fabric pattern, not a real animal",
    "textile": "a repeating decorative fabric pattern, not a photograph of a real object",
}
_ZS_MARGIN_CONCEPTS = frozenset({"crow", "butterfly", "jewelry"})
_ZS_FRUIT = frozenset({"cherry", "strawberry"})
_ZS_PRINT_FAMS = frozenset({"floral", "animal_print", "paisley", "baroque"})
_TIGER_TYPES = frozenset({"tiger", "tiger_stripe", "kaplan"})
_LEOPARD_TYPES = frozenset({"leopard", "leopar", "cheetah", "jaguar"})

_DETECTOR_SOURCES = frozenset(
    {
        "object_detector", "object detector", "detector", "fusion",
        "rtdetr", "rt-detr", "uvi", "face", "face_index",
    }
)


def _tokens(text: str) -> list[str]:
    try:
        from core.visual_concept import rewrite_visual_phrases

        blob = rewrite_visual_phrases(text)
    except Exception:
        blob = normalize_turkish(text or "")
    return [
        x for x in re.findall(r"[a-z0-9_ğüşöçıİĞÜŞÖÇ]+", blob)
        if len(x) >= 2
    ]


def concept_label_tr(lemma: str) -> str:
    key = normalize_turkish(lemma or "")
    if key in _PERSON_LANE:
        return _PERSON_LANE[key][1]
    if key in _OBJECT_LANE:
        return _OBJECT_LANE[key][1]
    return str(lemma or "Kavram").strip().capitalize() or "Kavram"


def kanit_yok_label(lemma: str) -> str:
    return f"{concept_label_tr(lemma)} — Kanıt yok"


def classify_accuracy_lane(text: str) -> str:
    """pattern | object | brand | hybrid | none"""
    try:
        from core.visual_concept_dna import parse_search_bags

        bags = parse_search_bags(text)
        if bags.pattern_weighted:
            return "pattern"
    except Exception:
        bags = None
    toks = _tokens(text)
    if not toks:
        return "none"
    folded = [normalize_turkish(t) for t in toks]
    if folded == ["marka"] or (len(folded) == 1 and folded[0] in _GENERIC_BRAND):
        return "brand"
    has_pat = any(t in _PATTERN_LANE for t in folded)
    has_obj = any(t in _PERSON_LANE or t in _OBJECT_LANE for t in folded)
    has_brand = False
    try:
        from core.brand_aliases import resolve_brand_alias

        raw = str(text or "").strip()
        if normalize_turkish(raw) not in _GENERIC_BRAND:
            has_brand = bool(resolve_brand_alias(raw) or any(resolve_brand_alias(t) for t in toks))
    except Exception:
        has_brand = False
    bits = int(has_pat) + int(has_obj) + int(has_brand)
    if bits >= 2:
        return "hybrid"
    if has_brand:
        return "brand"
    if has_obj:
        return "object"
    if has_pat:
        return "pattern"
    return "none"


def object_query_lemmas(text: str) -> list[tuple[str, str]]:
    """(canonical_en, label_tr) for lane-B tokens in the query."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tok in _tokens(text):
        key = normalize_turkish(tok)
        pair = _PERSON_LANE.get(key) or _OBJECT_LANE.get(key)
        if not pair or pair[0] in seen:
            continue
        seen.add(pair[0])
        out.append(pair)
    return out


def is_generic_brand_query(text: str) -> bool:
    toks = _tokens(text)
    return bool(toks) and all(normalize_turkish(t) in _GENERIC_BRAND for t in toks)


def is_specific_brand_query(text: str) -> bool:
    if is_generic_brand_query(text):
        return False
    try:
        from core.brand_aliases import resolve_brand_alias

        raw = str(text or "").strip()
        if resolve_brand_alias(raw):
            return True
        return any(bool(resolve_brand_alias(t)) for t in _tokens(text))
    except Exception:
        return False


def _dna(result: Any) -> dict[str, Any]:
    dbg = getattr(result, "debug", {}) or {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    return dna if isinstance(dna, dict) else {}


def _animal_type(result: Any) -> str:
    dna = _dna(result)
    dbg = getattr(result, "debug", {}) or {}
    return str(
        dna.get("animal_print_type")
        or dbg.get("animal_print_type")
        or dna.get("motif")
        or dna.get("type")
        or ""
    ).strip().lower()


def clip_only_similarity(result: Any) -> bool:
    dbg = getattr(result, "debug", {}) or {}
    kind = str(dbg.get("entity_evidence_kind") or "").lower()
    if kind in _HARD_ENTITY_KINDS:
        return False
    if dbg.get("clip_as_detector") or kind in _CLIP_KINDS:
        return True
    clip = max(
        float(dbg.get("clip_score") or 0.0),
        float((getattr(result, "breakdown", {}) or {}).get("clip") or 0.0),
        float(dbg.get("entity_semantic_score") or 0.0),
    )
    hard = has_hard_person_evidence(result) or has_hard_object_evidence(result, ())
    return clip >= 0.20 and not hard


def is_gender_object_query(text: str) -> bool:
    cans = {c for c, _ in object_query_lemmas(text)}
    return bool(cans & {"woman", "man", "face", "child"})


def zero_shot_spec(text: str) -> dict[str, str] | None:
    """OpenCLIP concept spec. None for pattern/gender/detector-backed classes."""
    if is_gender_object_query(text):
        return None
    lane = classify_accuracy_lane(text)
    if lane in {"pattern", "brand"}:
        return None
    lane_b = [c for c, _ in object_query_lemmas(text)]
    if lane_b and all(c not in _ZS_PROMPT for c in lane_b):
        return None
    try:
        from core.visual_concept import (
            compile_visual_query,
            detector_labels_for_lemma,
            english_lemma,
            prompt_for_lemma,
        )
        from core.visual_concept_dna import parse_search_bags

        bags = parse_search_bags(text)
        if bags.pattern_weighted or bags.query_sense == "pattern":
            return None
        lemma = ""
        try:
            from core.universal_visual_intel import parse_universal_query

            nid = str(getattr(parse_universal_query(text), "node_id", "") or "")
            if nid.startswith("open:"):
                nid = nid.split(":", 1)[-1].replace("_", " ")
            if nid in {"person", "female_person", "male_person", "child", "girl", "face"}:
                nid = ""
            if nid and not detector_labels_for_lemma(nid):
                lemma = nid
        except Exception:
            lemma = ""
        compiled = compile_visual_query(text)
        if not lemma:
            for c in compiled.concepts:
                raw = str(c.concept_id.split(":", 1)[-1] or "").replace("_", " ").strip()
                if not raw or raw in {"person", "woman", "man", "face", "child"}:
                    continue
                if detector_labels_for_lemma(raw):
                    continue
                lemma = raw
                break
        if not lemma:
            for obj in list(bags.objects or []):
                raw = str(obj or "")
                if raw.startswith("open:"):
                    raw = raw.split(":", 1)[-1]
                raw = raw.replace("_", " ").strip()
                if not raw or raw in {"person", "woman", "man", "face"}:
                    continue
                if detector_labels_for_lemma(raw):
                    continue
                lemma = english_lemma(raw) or raw
                break
        if not lemma:
            return None
        try:
            from core.visual_concept import concept_type_for_lemma

            if concept_type_for_lemma(lemma) in {
                "PATTERN", "STYLE", "VISUAL_EFFECT", "TEXTURE", "COMPOSITION",
            }:
                return None
        except Exception:
            pass
        return {
            "concept": lemma,
            "prompt": str(_ZS_PROMPT.get(lemma) or prompt_for_lemma(lemma)),
            "type": ZERO_SHOT_TYPE,
            "source": ZERO_SHOT_SOURCE,
        }
    except Exception:
        return None


def zero_shot_verify_prompts(concept: str) -> dict[str, str]:
    """Extra CLIP texts for target-vs-rival-vs-textile. No Grounding DINO."""
    key = str(concept or "").strip().lower()
    out: dict[str, str] = dict(_ZS_TEXTILE_PROMPT)
    for rival in _ZS_RIVALS.get(key, ()):
        prompt = _ZS_PROMPT.get(rival)
        if prompt:
            out[rival] = prompt
    return out


def _pattern_family_of(result: Any) -> str:
    fam = str(getattr(result, "pattern_family", "") or "").lower()
    if fam:
        return fam
    dbg = getattr(result, "debug", {}) or {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    return str(tm.get("pattern_family") or dbg.get("pattern_family") or "").lower()


def print_blocks_zero_shot(result: Any, concept: str) -> bool:
    """Pattern DNA / family print is not a real object photo."""
    key = str(concept or "").strip().lower()
    fam = _pattern_family_of(result)
    if fam in _ZS_PRINT_FAMS and key in {
        "crow", "butterfly", "cherry", "strawberry", "jewelry", "lips",
        "eagle", "sparrow", "crocodile", "lion", "elephant",
    }:
        return True
    animal = _animal_type(result)
    if key in {"crow", "butterfly", "eagle", "sparrow"} and animal in _LEOPARD_TYPES | {
        "zebra", "snake", "tiger", "tiger_stripe",
    }:
        return True
    return False


def score_zero_shot_gate(
    concept: str,
    target: float,
    rivals: dict[str, float] | None,
) -> tuple[bool, dict[str, Any]]:
    """CLIP-only accept/reject. Missing rivals → floor-only (unit tests / no FAISS)."""
    key = str(concept or "").strip().lower()
    tgt = float(target or 0.0)
    meta: dict[str, Any] = {
        "clip_target": round(tgt, 4),
        "clip_margin": None,
        "clip_textile": None,
        "clip_best_rival": "",
    }
    if tgt < float(ZERO_SHOT_FLOOR):
        return False, meta
    blob = {str(k): float(v or 0.0) for k, v in (rivals or {}).items() if k and k != "_similarity"}
    if not blob:
        return True, meta
    textile = max(
        float(blob.get("floral") or 0.0),
        float(blob.get("animal_print") or 0.0),
        float(blob.get("textile") or 0.0),
    )
    meta["clip_textile"] = round(textile, 4)
    skip = set(_ZS_TEXTILE_PROMPT)
    rival_items = [(k, v) for k, v in blob.items() if k not in skip]
    best_name = ""
    best_rival = 0.0
    if rival_items:
        best_name, best_rival = max(rival_items, key=lambda kv: kv[1])
    margin = tgt - best_rival
    meta["clip_margin"] = round(margin, 4)
    meta["clip_best_rival"] = best_name
    if key in _ZS_FRUIT and (tgt - textile) < float(ZERO_SHOT_FRUIT_TEXTILE_GAP):
        return False, meta
    if key in _ZS_FRUIT:
        flower = float(blob.get("flower") or 0.0)
        if (tgt - flower) < float(ZERO_SHOT_MARGIN):
            return False, meta
    if textile >= tgt and key in _ZS_MARGIN_CONCEPTS | _ZS_FRUIT | {"lips"}:
        return False, meta
    if key in _ZS_MARGIN_CONCEPTS and rival_items and margin < float(ZERO_SHOT_MARGIN):
        return False, meta
    return True, meta


def zero_shot_verify_ok(result: Any, concept: str) -> bool:
    if print_blocks_zero_shot(result, concept):
        return False
    dbg = getattr(result, "debug", {}) or {}
    if dbg.get("zs_verified") is False:
        return False
    rivals = dbg.get("zs_clip_rivals")
    if not isinstance(rivals, dict):
        rivals = None
    ok, _ = score_zero_shot_gate(concept, zero_shot_score(result), rivals)
    return ok


def zero_shot_score(result: Any) -> float:
    dbg = getattr(result, "debug", {}) or {}
    bd = getattr(result, "breakdown", {}) or {}
    return max(
        float(dbg.get("zero_shot_score") or 0.0),
        float(dbg.get("clip_score") or 0.0),
        float(bd.get("clip") or 0.0),
        float(bd.get("_similarity") or 0.0),
    )


def has_zero_shot_visual(result: Any, text: str = "") -> bool:
    if text and is_gender_object_query(text):
        return False
    dbg = getattr(result, "debug", {}) or {}
    if dbg.get("object_index_hit") or dbg.get("face_gender_match"):
        return False
    if dbg.get("clip_as_detector"):
        return False
    spec = zero_shot_spec(text) if text else None
    if text and spec is None:
        return False
    concept = str(
        (spec or {}).get("concept")
        or dbg.get("zero_shot_concept")
        or ""
    )
    if zero_shot_score(result) < ZERO_SHOT_FLOOR:
        return False
    if concept:
        return zero_shot_verify_ok(result, concept)
    return str(dbg.get("evidence_type") or "") == ZERO_SHOT_TYPE


def stamp_zero_shot(result: Any, spec: dict[str, str], score: float) -> None:
    dbg = dict(getattr(result, "debug", {}) or {})
    if dbg.get("object_index_hit") or dbg.get("face_gender_match"):
        return
    if str(dbg.get("evidence_type") or "") == OPEN_VOCAB_TYPE or dbg.get("open_vocab_object"):
        sc = max(0.0, float(score or 0.0))
        dbg["zero_shot_visual"] = True
        dbg["zero_shot_concept"] = str(spec.get("concept") or "")
        dbg["zero_shot_score"] = round(sc, 4)
        dbg["evidence_type"] = OPEN_VOCAB_TYPE
        result.debug = dbg
        return
    sc = max(0.0, float(score or 0.0))
    concept = str(spec.get("concept") or "")
    rivals = dbg.get("zs_clip_rivals") if isinstance(dbg.get("zs_clip_rivals"), dict) else None
    ok, meta = score_zero_shot_gate(concept, sc, rivals)
    if print_blocks_zero_shot(result, concept):
        ok = False
    dbg["zs_verified"] = bool(ok)
    dbg["zs_clip_margin"] = meta.get("clip_margin")
    dbg["zs_clip_textile"] = meta.get("clip_textile")
    dbg["zs_clip_best_rival"] = meta.get("clip_best_rival") or ""
    if not ok:
        result.debug = dbg
        return
    dbg["evidence_type"] = ZERO_SHOT_TYPE
    dbg["evidence_source"] = ZERO_SHOT_SOURCE
    dbg["zero_shot_visual"] = True
    dbg["zero_shot_concept"] = str(spec.get("concept") or "")
    dbg["zero_shot_score"] = round(sc, 4)
    dbg["clip_as_detector"] = False
    dbg["similarity_is_not_truth"] = True
    result.debug = dbg


def has_hard_person_evidence(result: Any) -> bool:
    dbg = getattr(result, "debug", {}) or {}
    if dbg.get("face_gender_match") or dbg.get("face_match") or dbg.get("face_index_hit"):
        return True
    if float(dbg.get("gender_visual_score") or 0.0) >= 0.18:
        return True
    if dbg.get("uvi_person") or dbg.get("uvi_person_hit"):
        return True
    kind = str(dbg.get("entity_evidence_kind") or "")
    if kind in _HARD_ENTITY_KINDS and float(dbg.get("entity_evidence_score") or 0.0) >= 0.45:
        qents = {str(x).lower() for x in (dbg.get("entity_query") or [])}
        personish = qents & {
            "person", "woman", "man", "female", "male", "face", "child",
            "kadin", "kadın", "erkek",
        }
        if personish:
            return True
        src = str(dbg.get("entity_evidence_source") or "").lower()
        if src in _DETECTOR_SOURCES:
            return True
    goi = _dna(result)
    # UVI / RT-DETR person boxes live on texture_map.global_object_intelligence
    wrap = (getattr(result, "debug", {}) or {}).get("texture_map") or {}
    goi_obj = wrap.get("global_object_intelligence") if isinstance(wrap, dict) else None
    if isinstance(goi_obj, dict):
        for obj in goi_obj.get("objects") or ():
            if not isinstance(obj, dict):
                continue
            lab = str(obj.get("label") or obj.get("label_tr") or "").lower()
            src = str(obj.get("source") or obj.get("backend") or "object_detector").lower()
            if src in _CLIP_KINDS:
                continue
            if any(x in lab for x in ("person", "woman", "man", "face", "human", "kadin", "erkek")):
                try:
                    if float(obj.get("confidence") or 0.0) >= 0.25:
                        return True
                except (TypeError, ValueError):
                    return True
    _ = goi
    return False


def has_hard_object_evidence(result: Any, lemmas: tuple[str, ...]) -> bool:
    dbg = getattr(result, "debug", {}) or {}
    kind = str(dbg.get("entity_evidence_kind") or "")
    if kind in _HARD_ENTITY_KINDS and float(dbg.get("entity_evidence_score") or 0.0) >= 0.45:
        qents = {str(x).lower() for x in (dbg.get("entity_query") or [])}
        wanted = {str(x).lower() for x in lemmas}
        if not wanted or (qents & wanted) or not qents:
            return True
    if dbg.get("object_index_hit") or dbg.get("visual_concept_detector"):
        return True
    try:
        from core.visual_concept_dna import dna_hits_for_lemmas, read_visual_concept_dna

        hits = dna_hits_for_lemmas(read_visual_concept_dna(result), lemmas)
        if any(float(h.get("confidence") or 0.0) >= 0.35 for h in hits):
            return True
    except Exception:
        pass
    wrap = (getattr(result, "debug", {}) or {}).get("texture_map") or {}
    goi_obj = wrap.get("global_object_intelligence") if isinstance(wrap, dict) else None
    wanted = {str(x).lower() for x in lemmas}
    if isinstance(goi_obj, dict):
        for obj in goi_obj.get("objects") or ():
            if not isinstance(obj, dict):
                continue
            lab = str(obj.get("label") or obj.get("label_tr") or "").lower()
            src = str(obj.get("source") or obj.get("backend") or "object_detector").lower()
            if src in _CLIP_KINDS or src in {"grounding_dino", "open_vocab_object"}:
                continue
            if wanted and not any(w in lab or lab in w for w in wanted):
                continue
            try:
                if float(obj.get("confidence") or 0.0) >= 0.25:
                    return True
            except (TypeError, ValueError):
                return True
    return False


def has_tiger_evidence(result: Any) -> bool:
    at = _animal_type(result)
    if at in _TIGER_TYPES:
        return True
    dbg = getattr(result, "debug", {}) or {}
    if str(dbg.get("entity_evidence_kind") or "") in _HARD_ENTITY_KINDS:
        qents = {str(x).lower() for x in (dbg.get("entity_query") or [])}
        lines = " ".join(str(x) for x in (dbg.get("entity_debug_lines") or [])).lower()
        if qents & {"tiger", "kaplan"} or "tiger" in lines or "kaplan" in lines:
            if float(dbg.get("entity_evidence_score") or 0.0) >= 0.45:
                return True
    blob = f"{at} {getattr(result, 'pattern_family', '')} {_animal_type(result)}".lower()
    if "tiger" in blob or "kaplan" in blob:
        if at in _LEOPARD_TYPES:
            return False
        wrap = (getattr(result, "debug", {}) or {}).get("texture_map") or {}
        goi_obj = wrap.get("global_object_intelligence") if isinstance(wrap, dict) else None
        if isinstance(goi_obj, dict):
            for obj in goi_obj.get("objects") or ():
                if not isinstance(obj, dict):
                    continue
                lab = str(obj.get("label") or "").lower()
                if "tiger" in lab or "kaplan" in lab:
                    return True
    return False


def leopard_dna_not_tiger(result: Any) -> bool:
    at = _animal_type(result)
    fam = str(getattr(result, "pattern_family", "") or _dna(result).get("family") or "").lower()
    if at in _LEOPARD_TYPES:
        return True
    if fam == "animal_print" and at not in _TIGER_TYPES:
        return True
    return False


def brand_evidence(result: Any) -> bool:
    dbg = getattr(result, "debug", {}) or {}
    bd = getattr(result, "breakdown", {}) or {}
    fq = dbg.get("fusion_query_v2") if isinstance(dbg.get("fusion_query_v2"), dict) else {}
    if dbg.get("brand_evidence") or dbg.get("brand_match") or dbg.get("protected_exact"):
        return True
    if float(bd.get("brand_alias_score") or 0.0) >= 0.5:
        return True
    if float(bd.get("brand_evidence_hit") or 0.0) >= 0.5:
        return True
    if float(fq.get("brand") or 0.0) >= 0.5:
        return True
    return False


def has_hard_evidence_for_query(result: Any, text: str) -> bool:
    lemmas = object_query_lemmas(text)
    cans = tuple(c for c, _ in lemmas)
    person_q = any(c in {"woman", "man", "person", "face", "child"} for c in cans)
    tiger_q = "tiger" in cans
    if tiger_q:
        return has_tiger_evidence(result) and not (
            leopard_dna_not_tiger(result) and not has_tiger_evidence(result)
        )
    if person_q:
        return has_hard_person_evidence(result)
    obj_cans = tuple(
        c for c in cans if c not in {"woman", "man", "person", "face", "child"}
    )
    if obj_cans:
        return has_hard_object_evidence(result, obj_cans)
    return False


def _stamp(row: Any, extra: dict[str, Any]) -> None:
    dbg = dict(getattr(row, "debug", {}) or {})
    dbg.update(extra)
    row.debug = dbg


def _keep_unevidenced(text: str) -> bool:
    """Keep CLIP-only rows only for compound queries (kadın çanta, kırmızı çanta)."""
    toks = [normalize_turkish(t) for t in _tokens(text)]
    if len(toks) < 2:
        return False
    n_person = sum(1 for t in toks if t in _PERSON_LANE)
    n_obj = sum(1 for t in toks if t in _OBJECT_LANE)
    n_pat = sum(1 for t in toks if t in _PATTERN_LANE)
    extra = [
        t for t in toks
        if t not in _PERSON_LANE and t not in _OBJECT_LANE and t not in _PATTERN_LANE
        and t not in _GENERIC_BRAND
    ]
    return (n_person and n_obj) or (n_obj + n_person >= 1 and (extra or n_pat))


def apply_search_evidence_gate(
    results: list[Any],
    text: str,
    *,
    multi_channel: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Filter/rerank by hard evidence. Pattern lane is a no-op."""
    lane = classify_accuracy_lane(text)
    lemmas = object_query_lemmas(text)
    label = lemmas[0][1] if lemmas else concept_label_tr(text)
    meta: dict[str, Any] = {
        "accuracy_lane": lane,
        "clip_as_detector": False,
        "similarity_is_not_truth": True,
        "empty_state": "",
    }
    if not results:
        if lane == "brand":
            meta["empty_state"] = EMPTY_BRAND
        elif lane == "object":
            meta["empty_state"] = kanit_yok_label(label)
        return results, meta

    if lane in {"pattern", "none"}:
        return results, meta

    if lane == "hybrid":
        return results, meta

    if lane == "brand" or is_specific_brand_query(text) or is_generic_brand_query(text):
        evidenced = [r for r in results if brand_evidence(r)]
        weak = [r for r in results if not brand_evidence(r)]
        if not evidenced:
            for row in weak:
                _stamp(row, {
                    "kanit_yok": True,
                    "kanit_yok_label": EMPTY_BRAND,
                    "evidence_ok": False,
                    "top50_ok": False,
                    "similarity_is_not_truth": True,
                })
            meta["empty_state"] = EMPTY_BRAND
            # Do not fill with random prints / CLIP lookalikes.
            return [], meta
        for row in evidenced:
            _stamp(row, {"evidence_ok": True, "top50_ok": True, "kanit_yok": False})
        for row in weak:
            _stamp(row, {
                "evidence_ok": False,
                "top50_ok": False,
                "kanit_yok": True,
                "kanit_yok_label": EMPTY_BRAND,
            })
        return evidenced + weak, meta

    if lane != "object" and not lemmas:
        return results, meta

    primary = lemmas[0][0] if lemmas else ""
    eligible: list[Any] = []
    blocked: list[Any] = []
    for row in results:
        ok = has_hard_evidence_for_query(row, text)
        if primary == "tiger":
            ok = has_tiger_evidence(row)
            if leopard_dna_not_tiger(row) and not has_tiger_evidence(row):
                ok = False
        clip_fill = clip_only_similarity(row) and not ok
        ovd = (not ok) and bool(
            (getattr(row, "debug", {}) or {}).get("open_vocab_object")
        ) and not (getattr(row, "debug", {}) or {}).get("ovd_rejected")
        zs = (not ok) and has_zero_shot_visual(row, text)
        missing = kanit_yok_label(label)
        found = ""
        if ok:
            try:
                from core.visual_concept_dna import (
                    detected_label,
                    dna_hits_for_lemmas,
                    read_visual_concept_dna,
                )

                hits = dna_hits_for_lemmas(
                    read_visual_concept_dna(row), tuple(c for c, _ in lemmas) or (primary,),
                )
                if hits:
                    best = max(hits, key=lambda h: float(h.get("confidence") or 0.0))
                    found = detected_label(
                        str(best.get("lemma") or primary or label),
                        float(best.get("confidence") or 0.0),
                    )
                    base = max(0.0, min(1.0, float(getattr(row, "score", 0.0) or 0.0)))
                    row.score = max(base, 0.72 + 0.26 * float(best.get("confidence") or 0.0))
                    row.score_percent = round(float(row.score) * 100.0, 1)
            except Exception:
                found = f"{label} tespit edildi"
        elif ovd:
            child = str((getattr(row, "debug", {}) or {}).get("ovd_matched_child") or label)
            found = f"{label} — {child}"
            if zs:
                spec = zero_shot_spec(text) or {}
                stamp_zero_shot(row, spec, zero_shot_score(row))
        elif zs:
            spec = zero_shot_spec(text) or {}
            stamp_zero_shot(row, spec, zero_shot_score(row))
            found = f"{label} — görsel benzerlik"
            sc = zero_shot_score(row)
            if sc > float(getattr(row, "score", 0.0) or 0.0):
                row.score = sc
                row.score_percent = round(sc * 100.0, 1)
        _stamp(row, {
            "accuracy_lane": "object",
            "evidence_ok": bool(ok),
            "top50_ok": bool(ok or ovd or zs),
            "kanit_yok": not ok and not ovd and not zs,
            "kanit_yok_label": "" if (ok or ovd or zs) else missing,
            "concept_found_label": found,
            "clip_as_detector": False,
            "similarity_is_not_truth": True,
            "clip_only_blocked": bool(clip_fill and not zs and not ovd),
        })
        if ok or ovd or zs:
            eligible.append(row)
        else:
            blocked.append(row)

    exact_e = [
        r for r in eligible
        if (getattr(r, "debug", {}) or {}).get("evidence_ok")
        or (getattr(r, "debug", {}) or {}).get("object_index_hit")
    ]
    ovd_e = [
        r for r in eligible
        if r not in exact_e
        and (getattr(r, "debug", {}) or {}).get("open_vocab_object")
        and not (getattr(r, "debug", {}) or {}).get("ovd_rejected")
    ]
    zs_e = [r for r in eligible if r not in exact_e and r not in ovd_e]
    if zs_e:
        zs_e.sort(key=zero_shot_score, reverse=True)
        zs_e = zs_e[: int(ZERO_SHOT_TOP_N)]
    eligible = exact_e + ovd_e + zs_e

    keep_weak = _keep_unevidenced(text)
    meta["evidence_count"] = len(eligible)
    if not eligible:
        meta["empty_state"] = kanit_yok_label(label)
        if keep_weak:
            return blocked, meta
        return [], meta

    page = eligible[:OBJECT_TOP_N]
    rest_eligible = eligible[OBJECT_TOP_N:]
    if keep_weak:
        return page + rest_eligible + blocked, meta
    return page + rest_eligible, meta
