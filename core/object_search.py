"""Text → object-index queries. Detector class labels only — not CLIP similarity."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from core.object_index import ObjectIndexStore

# COCO class + Turkish aliases. CLIP/zero-shot is not used here.
_CLASS_ALIASES: dict[str, str] = {
    "kedi": "cat",
    "cat": "cat",
    "kopek": "dog",
    "köpek": "dog",
    "dog": "dog",
    "kus": "bird",
    "kuş": "bird",
    "bird": "bird",
    "insan": "person",
    "kisi": "person",
    "kişi": "person",
    "person": "person",
    "people": "person",
    "canta": "handbag",
    "çanta": "handbag",
    "bag": "handbag",
    "handbag": "handbag",
    "araba": "car",
    "car": "car",
    "otomobil": "car",
}

# çanta covers common bag COCO classes (still detector labels, not CLIP).
_EXPAND: dict[str, set[str]] = {
    "handbag": {"handbag", "backpack"},
}

_TR_NUM = {
    "bir": 1, "iki": 2, "uc": 3, "üç": 3, "dort": 4, "dört": 4,
    "bes": 5, "beş": 5,
}


def _fold(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).lower().replace("ı", "i")
    t = "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", t).strip()


def resolve_object_class(token: str) -> dict[str, Any] | None:
    """Map a query token to detector class labels. Not CLIP."""
    folded = _fold(token)
    label = _CLASS_ALIASES.get(folded)
    if not label:
        try:
            from core.visual_concept import resolve_detector_labels

            extra = resolve_detector_labels(token)
        except Exception:
            extra = None
        if extra:
            return {"label": next(iter(extra)), "labels": set(extra)}
        return None
    return {"label": label, "labels": set(_EXPAND.get(label, {label}))}


def parse_object_query(text: str) -> dict[str, Any] | None:
    q = _fold(text)
    if not q:
        return None
    q = q.replace("&", " + ").replace(",", " + ")
    q = re.sub(r"\bve\b", "+", q)
    q = re.sub(r"\band\b", "+", q)
    parts = [p.strip() for p in q.split("+") if p.strip()]
    if not parts:
        return None

    groups: list[dict[str, Any]] = []
    for part in parts:
        m = re.fullmatch(r"(\d+|bir|iki|uc|üç|dort|dört|bes|beş)\s+([a-zçğıöşü]+)", part)
        min_count = 1
        token = part
        if m:
            raw_n, token = m.group(1), m.group(2)
            min_count = int(raw_n) if raw_n.isdigit() else int(_TR_NUM.get(raw_n, 1))
        label = _CLASS_ALIASES.get(token)
        if not label:
            return None
        groups.append({"label": label, "labels": _EXPAND.get(label, {label}), "min_count": min_count})

    if len(groups) == 1:
        g = groups[0]
        kind = "count" if g["min_count"] > 1 else "class"
        return {"kind": kind, "groups": groups, "source": "detector_class"}
    return {"kind": "and", "groups": groups, "source": "detector_class"}


class ObjectSearch:
    def __init__(self, db_path: str, *, readonly: bool = True):
        self.store = ObjectIndexStore(db_path, readonly=readonly)

    def file_ids(self, text: str, limit: int = 400) -> list[int]:
        spec = parse_object_query(text)
        if not spec:
            return []
        groups = spec["groups"]
        if spec["kind"] == "and":
            return self.store.search_labels_and(groups, limit=limit)
        g = groups[0]
        return self.store.search_label(g["labels"], min_count=int(g["min_count"]), limit=limit)
