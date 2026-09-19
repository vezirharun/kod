"""Query-time OPEN_VOCAB_OBJECT layer. Does not rebuild indexes or Pattern DNA."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from core.search_evidence_gate import classify_accuracy_lane
from core.textile_terms import normalize_turkish

OPEN_VOCAB_TYPE = "OPEN_VOCAB_OBJECT"
OPEN_VOCAB_SOURCE = "grounding_dino"
OVD_CACHE_VERSION = 1
OVD_MIN_CONFIDENCE = 0.35
OVD_MAX_BOX_AREA = 0.85
OVD_NMS_IOU = 0.50
OVD_CANDIDATE_LIMIT = 12

_TEXTILE_FAMILIES = frozenset({
    "floral", "animal_print", "paisley", "baroque", "ethnic",
    "watercolor", "textile_repeat", "reptile_skin",
})
_TEXTILE_CONCEPTS = frozenset({
    "strawberry", "cherry", "apple", "crocodile", "butterfly",
    "crow", "eagle", "bird", "mouse",
})

# query token (folded) -> parent english concept
_QUERY_CONCEPT = {
    "taki": "jewelry", "takı": "jewelry", "jewelry": "jewelry",
    "mucevher": "jewelry", "mücevher": "jewelry",
    "kolye": "necklace", "necklace": "necklace",
    "kupe": "earring", "küpe": "earring", "earring": "earring", "earrings": "earring",
    "yuzuk": "ring", "yüzük": "ring", "ring": "ring",
    "bileklik": "bracelet", "bracelet": "bracelet",
    "bros": "brooch", "broş": "brooch", "brooch": "brooch",
    "madalyon": "pendant", "pendant": "pendant",
    "karga": "crow", "crow": "crow",
    "kartal": "eagle", "eagle": "eagle",
    "kus": "bird", "kuş": "bird", "bird": "bird",
    "kelebek": "butterfly", "butterfly": "butterfly",
    "dudak": "lips", "lips": "lips",
    "fare": "mouse", "mouse": "mouse",
    "cilek": "strawberry", "çilek": "strawberry", "strawberry": "strawberry",
    "kiraz": "cherry", "cherry": "cherry",
    "timsah": "crocodile", "crocodile": "crocodile",
    "kedi": "cat", "cat": "cat",
    "kopek": "dog", "köpek": "dog", "dog": "dog",
    "araba": "car", "car": "car",
    "canta": "handbag", "çanta": "handbag", "handbag": "handbag",
}

_CHILD_OF = {
    "necklace": "jewelry", "earring": "jewelry", "ring": "jewelry",
    "bracelet": "jewelry", "brooch": "jewelry", "pendant": "jewelry",
    "crow": "bird", "eagle": "bird",
}

_EXPAND = {
    "jewelry": ("jewelry", "necklace", "earring", "ring", "bracelet", "brooch", "pendant"),
    "necklace": ("necklace",),
    "earring": ("earring",),
    "ring": ("ring",),
    "bracelet": ("bracelet",),
    "brooch": ("brooch",),
    "pendant": ("pendant",),
    "bird": ("bird", "crow", "eagle"),
    "crow": ("crow",),
    "eagle": ("eagle",),
    "butterfly": ("butterfly",),
    "lips": ("lips", "mouth"),
    "mouse": ("mouse",),
    "strawberry": ("strawberry",),
    "cherry": ("cherry",),
    "crocodile": ("crocodile",),
    "cat": ("cat",),
    "dog": ("dog",),
    "car": ("car",),
    "handbag": ("handbag", "bag"),
}


def ovd_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "open_vocab_object_enabled", False))


def ovd_query_concept(text: str) -> str:
    from core.search_evidence_gate import _tokens

    for tok in _tokens(text):
        key = normalize_turkish(tok)
        if key in _QUERY_CONCEPT:
            return _QUERY_CONCEPT[key]
    return ""


def ovd_prompts(text: str) -> tuple[str, tuple[str, ...]]:
    concept = ovd_query_concept(text)
    kids = _EXPAND.get(concept, (concept,) if concept else ())
    return concept, tuple(k for k in kids if k)


def should_run_ovd(text: str, enabled: bool = True) -> bool:
    if not enabled:
        return False
    if classify_accuracy_lane(text) != "object":
        return False
    concept, prompts = ovd_prompts(text)
    return bool(concept and prompts)


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes: list[dict[str, Any]], iou_thr: float = OVD_NMS_IOU) -> list[dict[str, Any]]:
    ordered = sorted(boxes, key=lambda b: float(b.get("confidence") or 0.0), reverse=True)
    keep: list[dict[str, Any]] = []
    for box in ordered:
        xy = box.get("xyxy") or []
        if len(xy) != 4:
            continue
        if any(_iou(xy, k.get("xyxy") or [0, 0, 0, 0]) >= float(iou_thr) for k in keep):
            continue
        keep.append(box)
    return keep


def filter_box_area(
    boxes: list[dict[str, Any]],
    image_area: float,
    max_ratio: float = OVD_MAX_BOX_AREA,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    area = max(1.0, float(image_area or 1.0))
    for box in boxes:
        xy = box.get("xyxy") or []
        if len(xy) != 4:
            continue
        bw = max(0.0, float(xy[2]) - float(xy[0]))
        bh = max(0.0, float(xy[3]) - float(xy[1]))
        ratio = (bw * bh) / area
        if ratio >= float(max_ratio):
            continue
        item = dict(box)
        item["bbox_area_ratio"] = round(ratio, 4)
        out.append(item)
    return out


def _dna_families(result: Any) -> set[str]:
    fam = {str(getattr(result, "pattern_family", "") or "").lower().strip()}
    dbg = getattr(result, "debug", {}) or {}
    tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    for key in ("family", "motif", "type"):
        fam.add(str(dna.get(key) or "").lower().strip())
    fam.add(str(dbg.get("animal_print_type") or "").lower().strip())
    return {x for x in fam if x}


def textile_blocks_ovd(result: Any, concept: str) -> bool:
    families = _dna_families(result)
    if not (families & _TEXTILE_FAMILIES):
        return False
    parent = _CHILD_OF.get(concept, concept)
    return parent in _TEXTILE_CONCEPTS or concept in _TEXTILE_CONCEPTS


def has_open_vocab_object(result: Any) -> bool:
    dbg = getattr(result, "debug", {}) or {}
    if dbg.get("ovd_rejected"):
        return False
    if not dbg.get("open_vocab_object"):
        return False
    return float(dbg.get("ovd_confidence") or 0.0) >= float(OVD_MIN_CONFIDENCE)


def _cache_path(settings: Any) -> Path:
    root = Path(str(getattr(settings, "cache_dir", "") or "data"))
    return root / "ovd_cache_v1.json"


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def cache_key(
    *,
    image_id: str,
    concept: str,
    model: str,
    version: str,
    threshold: float,
) -> str:
    raw = "|".join([
        f"v{OVD_CACHE_VERSION}",
        str(image_id),
        str(concept),
        str(model),
        str(version),
        f"{float(threshold):.3f}",
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _image_id(result: Any) -> str:
    path = str(getattr(result, "path", "") or "")
    fid = getattr(result, "file_id", "")
    mtime = getattr(result, "mtime", 0)
    size = getattr(result, "file_size", 0)
    return f"{fid}:{mtime}:{size}:{path}"


def _parent_concept(child: str, query_concept: str) -> str:
    return query_concept or _CHILD_OF.get(child, child)


def stamp_open_vocab(
    result: Any,
    *,
    query_concept: str,
    boxes: list[dict[str, Any]],
    rejected: str = "",
) -> None:
    dbg = dict(getattr(result, "debug", {}) or {})
    exact = bool(dbg.get("object_index_hit")) and not bool(dbg.get("object_index_parent_hit"))
    best = boxes[0] if boxes else {}
    child = str(best.get("label") or query_concept)
    dbg["open_vocab_object"] = bool(boxes) and not rejected
    dbg["ovd_rejected"] = bool(rejected)
    dbg["ovd_reject_reason"] = rejected
    dbg["ovd_source"] = OPEN_VOCAB_SOURCE
    dbg["ovd_query_concept"] = query_concept
    dbg["ovd_matched_child"] = child
    dbg["ovd_boxes"] = boxes
    dbg["ovd_confidence"] = round(float(best.get("confidence") or 0.0), 4)
    if rejected:
        if str(dbg.get("evidence_type") or "") == OPEN_VOCAB_TYPE:
            dbg.pop("evidence_type", None)
        result.debug = dbg
        return
    if exact:
        dbg["ovd_supporting"] = True
        et = str(dbg.get("evidence_type") or "")
        if et in {"", OPEN_VOCAB_TYPE, "ZERO_SHOT_VISUAL"}:
            dbg["evidence_type"] = "EXACT_OBJECT"
        dbg["ovd_parent_concept"] = _parent_concept(child, query_concept)
        result.debug = dbg
        return
    if str(dbg.get("evidence_type") or "") == "ZERO_SHOT_VISUAL":
        dbg["zero_shot_visual"] = True
    dbg["evidence_type"] = OPEN_VOCAB_TYPE
    dbg["evidence_source"] = OPEN_VOCAB_SOURCE
    dbg["ovd_parent_concept"] = _parent_concept(child, query_concept)
    result.debug = dbg


_LAST_OVD_META: dict[str, Any] = {}


def last_ovd_meta() -> dict[str, Any]:
    return dict(_LAST_OVD_META)


def apply_open_vocab_layer(
    results: list[Any],
    text: str,
    *,
    settings: Any = None,
    backend: Any = None,
    candidate_limit: int | None = None,
) -> list[Any]:
    enabled = ovd_enabled(settings) if settings is not None else backend is not None
    if backend is None and not enabled:
        return results
    if not should_run_ovd(text, enabled=True):
        return results

    concept, prompts = ovd_prompts(text)
    if not concept:
        return results
    limit = int(
        candidate_limit
        if candidate_limit is not None
        else getattr(settings, "ovd_candidate_limit", OVD_CANDIDATE_LIMIT)
        if settings is not None
        else OVD_CANDIDATE_LIMIT
    )
    limit = max(8, min(20, limit))
    thresh = float(
        getattr(settings, "ovd_min_confidence", OVD_MIN_CONFIDENCE)
        if settings is not None
        else OVD_MIN_CONFIDENCE
    )
    max_area = float(
        getattr(settings, "ovd_max_box_area", OVD_MAX_BOX_AREA)
        if settings is not None
        else OVD_MAX_BOX_AREA
    )
    if backend is None:
        from core.ovd_backend import get_grounding_dino_backend

        backend = get_grounding_dino_backend()

    cache: dict[str, Any] = {}
    cache_file: Path | None = None
    if settings is not None:
        cache_file = _cache_path(settings)
        cache = _load_cache(cache_file)

    t0 = time.perf_counter()
    n_run = 0
    n_boxes = 0
    n_acc = 0
    per_ms: list[float] = []
    model = str(getattr(backend, "name", OPEN_VOCAB_SOURCE))
    version = str(getattr(backend, "version", "1"))

    for row in results[:limit]:
        path = str(getattr(row, "path", "") or "")
        key = cache_key(
            image_id=_image_id(row),
            concept=concept,
            model=model,
            version=version,
            threshold=thresh,
        )
        packed = cache.get(key)
        if packed is None:
            if not path:
                continue
            try:
                from PIL import Image

                im = Image.open(path).convert("RGB")
            except Exception:
                continue
            t1 = time.perf_counter()
            try:
                raw = backend.detect(im, list(prompts), thresh) or []
            except Exception:
                per_ms.append((time.perf_counter() - t1) * 1000.0)
                n_run += 1
                packed = {"boxes": []}
                cache[key] = packed
            else:
                per_ms.append((time.perf_counter() - t1) * 1000.0)
                n_run += 1
                w, h = im.size
                area = float(w * h)
                kept = filter_box_area(raw, area, max_area)
                kept = [
                    b for b in kept
                    if float(b.get("confidence") or 0.0) >= thresh
                ]
                kept = nms_boxes(kept)
                for b in kept:
                    b["source"] = OPEN_VOCAB_SOURCE
                    b["query_concept"] = concept
                    lab = str(b.get("label") or "")
                    b["matched_child_concept"] = lab
                    b["parent_concept"] = _parent_concept(lab, concept)
                packed = {"boxes": kept}
                cache[key] = packed
        else:
            n_run += 1
        boxes = list(packed.get("boxes") or [])
        n_boxes += len(boxes)
        reason = ""
        if textile_blocks_ovd(row, concept):
            reason = "textile_pattern_family"
            boxes = []
        if boxes:
            n_acc += 1
        stamp_open_vocab(row, query_concept=concept, boxes=boxes, rejected=reason)

    if cache_file is not None:
        try:
            _save_cache(cache_file, cache)
        except Exception:
            pass

    global _LAST_OVD_META
    _LAST_OVD_META = {
        "ovd_total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "ovd_per_image_ms": round(sum(per_ms) / len(per_ms), 1) if per_ms else 0.0,
        "ovd_candidates": n_run,
        "ovd_boxes": n_boxes,
        "ovd_accepted": n_acc,
    }
    return results
