"""Open visual concept compiler — not a closed 7-class catalog.

Unknown Turkish/English tokens still become CLIP prompts and ranking channels.
Detector labels are optional evidence, never the only vocabulary.
Does not write patterns.db / FAISS / index_v3 jobs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.textile_terms import normalize_turkish

_DATA = Path(__file__).resolve().parents[1] / "data" / "visual_open_concepts.json"

_STOP = frozenset(
    {
        "ve", "ile", "bir", "bu", "su", "o", "da", "de", "ki", "mi", "mu",
        "and", "or", "the", "a", "an", "of", "in", "on", "with",
        "desen", "deseni", "pattern", "motif", "print", "photo", "fotograf",
        "görsel", "gorsel", "resim", "image", "query", "ara",
        "olan", "olani", "olanı", "sahip", "icerir", "içerir", "iceren", "içeren",
        "etkisi", "darbesi", "efekti", "effect",
    }
)

# Longest-first; folded keys. Keeps "fırça etkisi" as one concept.
_PHRASE_LEMMAS: tuple[tuple[str, str], ...] = (
    ("boya sicramasi", "paint_splatter"),
    ("firca darbesi", "brushstroke"),
    ("firca etkisi", "brushstroke"),
    ("etnik desen", "ethnic_print"),
    ("barok desen", "baroque"),
)
# Longest-first Turkish inflections: çantalı, arabası, çiçekli, kadınlar…
_TR_SUFFIXES = (
    "larindan", "lerinden", "larinin", "lerinin",
    "sindaki", "sindeki",
    "lilar", "liler",
    "lik", "luk", "li", "lu",
    "lar", "ler",
    "nin", "nun",
    "dan", "den", "tan", "ten",
    "si", "su",
    "yi", "yu",
    "ya", "ye",
)
_COLORS = {
    "kirmizi": "red",
    "kırmızı": "red",
    "red": "red",
    "mavi": "blue",
    "blue": "blue",
    "yesil": "green",
    "yeşil": "green",
    "green": "green",
    "sari": "yellow",
    "sarı": "yellow",
    "yellow": "yellow",
    "siyah": "black",
    "black": "black",
    "beyaz": "white",
    "white": "white",
    "pembe": "pink",
    "pink": "pink",
    "mor": "purple",
    "purple": "purple",
    "turuncu": "orange",
    "orange": "orange",
    "kahverengi": "brown",
    "brown": "brown",
    "gri": "gray",
    "gray": "gray",
    "altin": "gold",
    "altın": "gold",
    "gold": "gold",
}


@dataclass
class VisualConcept:
    concept_id: str
    prompt: str
    source: str  # ontology | lexicon | open
    category: str = ""
    hypernyms: tuple[str, ...] = ()
    detector_labels: tuple[str, ...] = ()


@dataclass
class VisualConceptQuery:
    raw: str
    concepts: list[VisualConcept] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    conjunction: str = "and"
    spatial_text: str = ""
    clip_prompts: list[str] = field(default_factory=list)

    def open_ids(self) -> list[str]:
        return [c.concept_id for c in self.concepts if c.concept_id.startswith("open:")]


@dataclass
class VisualChannelScores:
    """Future fused ranking — CLIP / DINO / object boxes / color / DNA / OCR / face."""

    clip: float = 0.0
    dino: float = 0.0
    object_index: float = 0.0
    color: float = 0.0
    dna: float = 0.0
    ocr: float = 0.0
    face: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    object_confidence: float = 0.0
    relation: str = ""


DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "clip": 0.34,
    "dino": 0.22,
    "object_index": 0.16,
    "color": 0.08,
    "dna": 0.08,
    "ocr": 0.06,
    "face": 0.06,
}


def fuse_visual_channels(
    scores: VisualChannelScores,
    weights: dict[str, float] | None = None,
) -> float:
    w = dict(DEFAULT_FUSION_WEIGHTS)
    w.update(weights or {})
    total = 0.0
    acc = 0.0
    for key, wt in w.items():
        if wt <= 0:
            continue
        acc += float(wt) * float(getattr(scores, key, 0.0) or 0.0)
        total += float(wt)
    return acc / total if total else 0.0


@lru_cache(maxsize=1)
def load_open_concepts() -> dict[str, Any]:
    if not _DATA.is_file():
        return {"synonyms": {}, "hypernyms": {}, "detector_labels": {}, "categories": {}}
    try:
        data = json.loads(_DATA.read_text(encoding="utf-8"))
    except Exception:
        return {"synonyms": {}, "hypernyms": {}, "detector_labels": {}, "categories": {}}
    syn = {
        _fold_key(k): str(v).strip().lower()
        for k, v in dict(data.get("synonyms") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    hyp = {
        str(k).strip().lower(): [str(x) for x in (v or [])]
        for k, v in dict(data.get("hypernyms") or {}).items()
    }
    det = {
        str(k).strip().lower(): [str(x) for x in (v or []) if str(x)]
        for k, v in dict(data.get("detector_labels") or {}).items()
    }
    cats = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in dict(data.get("categories") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    ctypes = {
        str(k).strip().lower(): str(v).strip().upper()
        for k, v in dict(data.get("concept_types") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    prompts = {
        str(k).strip().lower(): str(v).strip()
        for k, v in dict(data.get("clip_prompts") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    return {
        "synonyms": syn,
        "hypernyms": hyp,
        "detector_labels": det,
        "categories": cats,
        "concept_types": ctypes,
        "clip_prompts": prompts,
    }


def _fold_key(text: str) -> str:
    return normalize_turkish(str(text or "")).strip()


def rewrite_visual_phrases(text: str) -> str:
    """Collapse multi-word style/effect phrases before tokenization."""
    norm = normalize_turkish(text or "")
    for phrase, lemma in _PHRASE_LEMMAS:
        if phrase in norm:
            norm = norm.replace(phrase, f" {lemma} ")
    return " ".join(norm.split())


def _tokens(text: str) -> list[str]:
    norm = rewrite_visual_phrases(text or "")
    norm = norm.replace("&", " ").replace(",", " ").replace("+", " ")
    parts = [p for p in re.split(r"[^\wçğıöşü]+", norm, flags=re.IGNORECASE) if p]
    return [p for p in parts if p and p not in _STOP and len(p) > 1]


def stem_tr_token(token: str) -> list[str]:
    """Yield folded token then suffix-stripped stems (lexicon lookup, not a closed list)."""
    t = _fold_key(token)
    out: list[str] = []
    if t:
        out.append(t)
    cur = t
    for _ in range(2):
        hit = False
        for suf in _TR_SUFFIXES:
            fs = _fold_key(suf)
            if fs and cur.endswith(fs) and len(cur) - len(fs) >= 3:
                cur = cur[: -len(fs)]
                if cur and cur not in out:
                    out.append(cur)
                hit = True
                break
        if not hit:
            break
    return out


def english_lemma(token: str) -> str:
    data = load_open_concepts()
    syn = data["synonyms"]
    for cand in stem_tr_token(token):
        if cand in syn:
            return str(syn[cand])
    cats = data.get("categories") or {}
    for cand in stem_tr_token(token):
        if cand in cats:
            return cand
    return str(syn.get(_fold_key(token)) or _fold_key(token))


def expand_clip_prompt(raw: str) -> str:
    """Short class names (bag, car, eye) are valid CLIP text; do not reject by length."""
    p = str(raw or "").strip()
    if not p:
        return "a photograph of an object"
    if " " not in p:
        return f"a photograph of a {p}, real object, not a textile print"
    return p


def concept_type_for_lemma(lemma: str) -> str:
    data = load_open_concepts()
    return str((data.get("concept_types") or {}).get(str(lemma).strip().lower()) or "")


def prompt_for_lemma(lemma: str) -> str:
    lemma = str(lemma or "").strip().lower() or "object"
    data = load_open_concepts()
    custom = str((data.get("clip_prompts") or {}).get(lemma) or "").strip()
    if custom:
        return custom
    ctype = str((data.get("concept_types") or {}).get(lemma) or "")
    if ctype in {"PATTERN", "STYLE", "VISUAL_EFFECT", "TEXTURE", "COMPOSITION"}:
        return (
            f"a textile {ctype.lower().replace('_', ' ')} of {lemma}, "
            "fabric pattern, not a photograph of a real isolated object"
        )
    return expand_clip_prompt(lemma)


def category_for_lemma(lemma: str) -> str:
    data = load_open_concepts()
    return str(data.get("categories", {}).get(str(lemma).strip().lower()) or "")


def catalog_stats() -> dict[str, int]:
    data = load_open_concepts()
    cats = set((data.get("categories") or {}).values())
    return {
        "synonyms": len(data.get("synonyms") or {}),
        "lemmas": len(set((data.get("synonyms") or {}).values())),
        "categories": len(cats),
    }


def prompt_for_open_id(concept_id: str) -> str:
    raw = str(concept_id or "")
    if raw.startswith("open:"):
        lemma = raw.split(":", 1)[-1].replace("_", " ")
        return prompt_for_lemma(lemma)
    return prompt_for_lemma(raw)


def detector_labels_for_lemma(lemma: str) -> tuple[str, ...]:
    data = load_open_concepts()
    labels = data["detector_labels"].get(str(lemma).lower()) or []
    return tuple(labels)


def hypernyms_for_lemma(lemma: str) -> tuple[str, ...]:
    data = load_open_concepts()
    return tuple(data["hypernyms"].get(str(lemma).lower()) or ())


def compile_visual_query(text: str) -> VisualConceptQuery:
    raw = str(text or "").strip()
    q = VisualConceptQuery(raw=raw)
    if not raw:
        return q
    folded = normalize_turkish(raw)
    if " ve " in f" {folded} " or "+" in raw or "&" in raw:
        q.conjunction = "and"
    tokens = _tokens(raw)
    seen: set[str] = set()
    for tok in tokens:
        fk = _fold_key(tok)
        if fk in _COLORS:
            col = _COLORS[fk]
            if col not in q.colors:
                q.colors.append(col)
            continue
        lemma = english_lemma(tok)
        try:
            from core.brand_aliases import resolve_brand_alias

            if resolve_brand_alias(tok) or resolve_brand_alias(lemma):
                continue
        except Exception:
            pass
        syn_hit = any(
            _fold_key(x) in load_open_concepts()["synonyms"] for x in stem_tr_token(tok)
        )
        cid = f"open:{lemma.replace(' ', '_')}"
        if cid in seen:
            continue
        seen.add(cid)
        q.concepts.append(
            VisualConcept(
                concept_id=cid,
                prompt=prompt_for_lemma(lemma),
                source="lexicon" if syn_hit else "open",
                category=category_for_lemma(lemma),
                hypernyms=hypernyms_for_lemma(lemma),
                detector_labels=detector_labels_for_lemma(lemma),
            )
        )
    q.clip_prompts = [c.prompt for c in q.concepts]
    if q.colors and q.clip_prompts:
        tint = q.colors[0]
        q.clip_prompts = [f"{tint} {p}" for p in q.clip_prompts]
    return q


def resolve_detector_labels(token: str) -> set[str] | None:
    """Optional COCO labels for object_index. None = not a detector-backed class."""
    lemma = english_lemma(token)
    labels = detector_labels_for_lemma(lemma)
    if labels:
        return set(labels)
    return None
