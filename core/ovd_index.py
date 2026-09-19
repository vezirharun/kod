"""Index-time OPEN_VOCAB_OBJECT queue. Never runs Grounding DINO during search."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from core.index_freeze import (
    in_index_write_session,
    in_search_session,
    process_search_active,
)
from core.open_vocab_object import (
    OVD_MAX_BOX_AREA,
    OVD_MIN_CONFIDENCE,
    _CHILD_OF,
    filter_box_area,
    nms_boxes,
    ovd_prompts,
)
from core.object_index import ObjectIndexStore
from core.owlv2_index_probe import OPEN_CONCEPTS, judge_box

logger = logging.getLogger(__name__)

VOCAB_VERSION = "owlv2-ovd-v1"
OVD_DETECTOR = "owlv2"
OVD_MODEL_VERSION = "base-patch16"
FLORAL_PENALTY = 0.0  # benchmark later; must not drop floral candidates

# Index-time OWL vocab: EXACT COCO / jewelry / lips are not OWL work.
INDEX_VOCAB: tuple[str, ...] = ("crow", "butterfly", "cherry", "eagle")
_TEXTILE_SKIP_OWL = frozenset({
    "floral", "animal_print", "paisley", "baroque", "ethnic",
    "watercolor", "textile_repeat", "reptile_skin",
})

_TR = {
    "bird": "Kuş", "crow": "Karga", "eagle": "Kartal", "butterfly": "Kelebek",
    "cat": "Kedi", "dog": "Köpek", "mouse": "Fare", "car": "Araba",
    "apple": "Elma", "cherry": "Kiraz", "crocodile": "Timsah", "lips": "Dudak",
    "jewelry": "Takı", "necklace": "Kolye", "earring": "Küpe", "ring": "Yüzük",
    "bracelet": "Bileklik", "brooch": "Broş", "pendant": "Madalyon",
}


def ovd_work_allowed() -> bool:
    """B: OVD index work pauses only during search write-protection.

    INDEX_FROZEN (A) alone must not disable OVD forever. Artifact writes still
    go through allow_index_writes / guard_index_write as usual.
    """
    if in_search_session() or process_search_active():
        return False
    return True


def query_ovd_labels(text: str) -> list[str]:
    concept, kids = ovd_prompts(text)
    if not concept:
        return []
    out = [concept, *kids]
    parent = _CHILD_OF.get(concept)
    if parent:
        out.append(parent)
    seen: list[str] = []
    for x in out:
        x = str(x).strip().lower()
        if x and x not in seen:
            seen.append(x)
    return seen


def candidate_priority(rec: dict[str, Any], *, has_rtdetr: bool = False) -> float:
    """Soft rank only. Floral/animal_print never eliminate. Floral penalty is 0."""
    score = 1.0
    if has_rtdetr:
        score += 0.05
    fam = str(rec.get("pattern_family") or "").lower()
    if fam in {"floral", "animal_print", "baroque", "paisley"}:
        score -= FLORAL_PENALTY
    return score


def owlv2_progress(store: ObjectIndexStore, *, total: int) -> dict[str, Any]:
    return store.owlv2_scan_progress(total=int(total))


def _stamp(
    store: ObjectIndexStore,
    file_id: int,
    image_path: str,
    mtime: float,
    size: int,
    model: str,
    threshold: float,
    *,
    status: str,
    error: str = "",
) -> None:
    store.upsert_open_vocab_objects(
        int(file_id), str(image_path), [],
        mtime=mtime, file_size=size, vocab_version=VOCAB_VERSION,
        model=model, model_version=OVD_MODEL_VERSION, threshold=float(threshold),
        scan_status=status, scan_error=error, last_path=image_path,
    )


def owl_preview_scan_key(preview_path: str, *, preview_version: str = "") -> dict[str, Any]:
    """Scan identity = Preview fingerprint, not original raster/EPS bytes."""
    st = os.stat(preview_path)
    mtime = float(st.st_mtime)
    size = int(st.st_size)
    vocab = f"{VOCAB_VERSION}|pv:{size}:{int(mtime)}:{preview_version or ''}"
    return {"mtime": mtime, "file_size": size, "vocab_version": vocab}


def owl_source_kind(image_path: str) -> str:
    p = str(image_path or "").replace("\\", "/").lower()
    if "feature_previews" in p or p.endswith(".webp"):
        return "feature_previews"
    return "original"


# COCO-like exact labels that do not fill INDEX_VOCAB (crow/eagle/butterfly/cherry).
_EXACT_NO_OPEN_VOCAB = frozenset({
    "car", "cat", "dog", "apple", "person", "bicycle", "motorcycle", "bus",
    "truck", "boat", "chair", "bottle", "cup", "laptop", "phone", "book",
    "clock", "vase", "scissors", "toothbrush", "keyboard", "mouse",
    "remote", "tv", "couch", "bed", "toilet", "sink", "oven", "toaster",
    "microwave", "refrigerator", "tie", "umbrella", "handbag", "suitcase",
    "frisbee", "skis", "snowboard", "kite", "surfboard", "tennis racket",
    "sports ball", "baseball bat", "baseball glove", "skateboard",
    "wine glass", "fork", "knife", "spoon", "bowl", "banana", "orange",
    "broccoli", "carrot", "pizza", "hot dog", "donut", "cake", "sandwich",
    "potted plant", "dining table", "parking meter", "fire hydrant",
    "stop sign", "bench", "traffic light", "airplane", "train",
    "horse", "sheep", "cow", "elephant", "bear", "giraffe", "zebra",
})
_OPEN_VOCAB_HINTS = (
    "crow", "karga", "raven", "eagle", "kartal", "butterfly", "kelebek",
    "cherry", "kiraz", "bird", "kuş", "kus",
)


# Claim tail_seq for pending OWLv2 jobs (smaller = earlier). 0 = unranked legacy.
OWL_RANK_OPEN_GAP = 1
OWL_RANK_OPEN_HINT = 2
OWL_RANK_UNCERTAIN = 3
OWL_RANK_WEAK = 4


def _textile_families(pattern_family: str, texture_map: Any = None) -> list[str]:
    tm = texture_map if isinstance(texture_map, dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    sem = tm.get("semantic_tags") if isinstance(tm.get("semantic_tags"), dict) else {}
    out: list[str] = []
    for raw in (
        pattern_family,
        tm.get("pattern_family"),
        dna.get("family"),
        dna.get("main_family"),
        sem.get("family") if isinstance(sem, dict) else "",
    ):
        s = str(raw or "").strip().lower()
        if s and s not in out:
            out.append(s)
    return out


def skip_owl_file(pattern_family: str, texture_map: Any = None) -> str:
    for fam in _textile_families(pattern_family, texture_map):
        if fam in _TEXTILE_SKIP_OWL:
            return "textile_pattern_skip_owl"
    tm = texture_map if isinstance(texture_map, dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    animal = str(dna.get("animal_print_type") or tm.get("animal_print_type") or "").lower()
    if animal in {"leopard", "leopar", "snake", "snakeskin", "zebra"}:
        return "pattern_dna_skip_owl"
    return ""


def _blob_has_open_vocab_hint(blob: str, rtdetr_labels: list[str]) -> bool:
    text = str(blob or "").lower()
    labs = {str(x).strip().lower() for x in (rtdetr_labels or [])}
    if labs & {"bird", "crow", "eagle", "butterfly", "cherry"}:
        return True
    return any(h in text for h in _OPEN_VOCAB_HINTS)


def owl_enrichment_decision(
    *,
    pattern_family: str = "",
    texture_map: Any = None,
    rtdetr_labels: list[str] | None = None,
    filename: str = "",
    path: str = "",
) -> dict[str, str]:
    """SKIP or RUN. Does not convert EXACT/PATTERN/CLIP into OPEN_VOCAB_OBJECT.

    Open-vocab hints win over textile SKIP (floral + butterfly stays RUN).
    weak_evidence is never converted into SKIP here.
    """
    tm = texture_map if isinstance(texture_map, dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    sem = tm.get("semantic_tags") if isinstance(tm.get("semantic_tags"), dict) else {}
    blob = " ".join(
        [
            str(filename or ""),
            str(path or ""),
            str(pattern_family or ""),
            str(dna.get("motif") or ""),
            str(dna.get("motif_family") or ""),
            str(sem.get("motifs") or ""),
        ]
    )
    labs = [str(x).strip().lower() for x in (rtdetr_labels or []) if str(x).strip()]
    if _blob_has_open_vocab_hint(blob, labs):
        return {"decision": "RUN", "reason": "open_vocab_gap"}
    reason = skip_owl_file(pattern_family, texture_map)
    if reason:
        return {"decision": "SKIP", "reason": reason}
    if labs and set(labs) <= _EXACT_NO_OPEN_VOCAB:
        return {"decision": "SKIP", "reason": "exact_object_sufficient"}
    return {"decision": "RUN", "reason": "weak_evidence"}


def owl_queue_rank(gate: dict[str, str], rtdetr_labels: list[str] | None = None) -> int:
    """1=open_gap labels, 2=open hint, 3=uncertain objects, 4=weak. 0=do not enqueue."""
    if str(gate.get("decision") or "") != "RUN":
        return 0
    labs = [str(x).strip().lower() for x in (rtdetr_labels or []) if str(x).strip()]
    reason = str(gate.get("reason") or "")
    if reason == "open_vocab_gap":
        if set(labs) & {"bird", "crow", "eagle", "butterfly", "cherry"}:
            return OWL_RANK_OPEN_GAP
        return OWL_RANK_OPEN_HINT
    if labs:
        return OWL_RANK_UNCERTAIN
    return OWL_RANK_WEAK


def owl_progress_tone(
    *,
    completed: int,
    total: int,
    priority_pending: int,
    weak_pending: int,
    unranked_pending: int = 0,
) -> str:
    """classic | deferred | health. No UI copy; color only."""
    if int(total or 0) > 0 and int(completed or 0) >= int(total):
        return "classic"
    if (
        int(priority_pending or 0) <= 0
        and int(weak_pending or 0) > 0
        and int(unranked_pending or 0) <= 0
    ):
        return "deferred"
    return "health"


def apply_owl_queue_policy(
    jobs: list[Any],
    *,
    db: Any,
    job_store: Any = None,
    settings: Any = None,
) -> list[Any]:
    """Drop safe OWL SKIPs from enqueue; set tail_seq rank on remaining OWL jobs.

    Does not create a second job per file. weak_evidence stays RUN at rank 4.
    """
    from dataclasses import replace

    from core.index_freeze import allow_index_writes
    from core.index_v3.artifact_state import object_index_path_for_patterns_db
    from core.index_v3.types import Artifact
    from core.manual_label_guard import parse_texture_map
    from core.object_index import ObjectIndexStore

    if not jobs:
        return jobs
    owl_jobs = [j for j in jobs if getattr(getattr(j, "artifact", None), "value", "") == "owlv2"]
    if not owl_jobs:
        return jobs
    obj_path = object_index_path_for_patterns_db(db, settings)
    obj = ObjectIndexStore(obj_path) if obj_path else None
    keep: list[Any] = []
    seen_owl: set[int] = set()
    for job in jobs:
        if getattr(getattr(job, "artifact", None), "value", "") != "owlv2":
            keep.append(job)
            continue
        fid = int(job.file_id)
        if fid in seen_owl:
            continue
        seen_owl.add(fid)
        row = db.get_file_by_id(fid) or {}
        feat = db.get_features(fid) or {}
        tm = parse_texture_map(feat.get("texture_map"))
        family = str(row.get("pattern_family") or "")
        if not family and isinstance(tm, dict):
            family = str((tm.get("pattern_dna") or {}).get("motif_family") or "")
        labs = obj.exact_object_labels(fid) if obj is not None else []
        gate = owl_enrichment_decision(
            pattern_family=family,
            texture_map=tm,
            rtdetr_labels=labs,
            filename=str(row.get("filename") or ""),
            path=str(row.get("path") or job.path or ""),
        )
        rank = owl_queue_rank(gate, labs)
        if rank <= 0:
            st = ""
            if job_store is not None:
                try:
                    st = str(job_store.job_state(fid, Artifact.OWLV2) or "")
                except Exception:
                    st = ""
            if st != "claimed" and obj is not None:
                preview = str(row.get("feature_preview_path") or "")
                try:
                    with allow_index_writes():
                        _stamp_owl_skip(obj, fid, preview, str(gate.get("reason") or "skip"))
                except Exception:
                    keep.append(replace(job, owl_priority=OWL_RANK_WEAK) if hasattr(job, "owl_priority") else job)
                    continue
                if job_store is not None and st in {"pending", "failed", "retry", ""}:
                    try:
                        if st == "pending":
                            job_store.complete(fid, Artifact.OWLV2)
                    except Exception:
                        pass
            continue
        keep.append(replace(job, owl_priority=int(rank)))
    return keep


def _stamp_owl_skip(store: Any, file_id: int, image_path: str, reason: str) -> None:
    model = f"{OVD_DETECTOR}/{OVD_MODEL_VERSION}"
    mtime = 0.0
    size = 0
    vocab = VOCAB_VERSION
    path = str(image_path or "")
    if path and os.path.isfile(path):
        key = owl_preview_scan_key(path)
        mtime = float(key["mtime"])
        size = int(key["file_size"])
        vocab = str(key["vocab_version"])
    store.upsert_open_vocab_objects(
        int(file_id),
        path or "",
        [],
        mtime=mtime,
        file_size=size,
        vocab_version=vocab,
        model=model,
        model_version=OVD_MODEL_VERSION,
        threshold=0.10,
        scan_status="skipped",
        scan_error=str(reason or "")[:300],
        last_path=path,
    )


def dry_run_queue(records: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    """Rank candidates. Does not load OWL or write DBs."""
    ranked = sorted(
        records,
        key=lambda r: (-candidate_priority(r), int(r.get("id") or r.get("file_id") or 0)),
    )
    out = []
    for rec in ranked[: max(0, int(limit))]:
        fid = int(rec.get("id") or rec.get("file_id") or 0)
        out.append({
            "file_id": fid,
            "path": str(rec.get("path") or ""),
            "pattern_family": str(rec.get("pattern_family") or ""),
            "priority": candidate_priority(rec),
            "eliminated": False,
        })
    return out


def _norm_bbox(xyxy: list[float], width: int, height: int) -> list[float]:
    w = max(1.0, float(width or 1))
    h = max(1.0, float(height or 1))
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    return [
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
        max(0.0, min(1.0, x2 / w)),
        max(0.0, min(1.0, y2 / h)),
    ]


def index_open_vocab_image(
    *,
    file_id: int,
    image_path: str,
    store: ObjectIndexStore,
    backend: Any = None,
    use_gpu: bool = False,
    threshold: float = 0.10,
    force: bool = False,
    pattern_family: str = "",
    texture_map: Any = None,
    clip_fn: Any = None,
    preview_version: str = "",
    archive_path: str = "",
) -> dict[str, Any]:
    """One OWLv2 forward on an existing Preview. Index session only."""
    allowed = ovd_work_allowed()
    source = owl_source_kind(image_path)
    file_path = str(archive_path or image_path)
    logger.info(
        "OWL_WORK_ALLOWED=%s file_id=%s write_session=%s search=%s source=%s",
        allowed,
        int(file_id),
        in_index_write_session(),
        bool(in_search_session() or process_search_active()),
        source,
    )
    if not allowed:
        return {"ok": False, "skipped": "frozen_or_search", "boxes": 0, "source": source}
    t_lookup = time.perf_counter()
    if not image_path or not os.path.isfile(image_path):
        return {"ok": False, "skipped": "preview_required", "boxes": 0, "source": source}
    key = owl_preview_scan_key(image_path, preview_version=str(preview_version or ""))
    mtime = float(key["mtime"])
    size = int(key["file_size"])
    vocab = str(key["vocab_version"])
    lookup_ms = (time.perf_counter() - t_lookup) * 1000.0
    model = f"{OVD_DETECTOR}/{OVD_MODEL_VERSION}"
    if not force and store.ovd_has_done_scan(int(file_id)):
        logger.info("OWL_DECISION=SKIP reason=resume file_id=%s", int(file_id))
        return {"ok": True, "skipped": "resume", "boxes": 0, "source": source, "owl_decision": "SKIP"}

    t_dna = time.perf_counter()
    rtdetr_labels = store.exact_object_labels(int(file_id))
    gate = owl_enrichment_decision(
        pattern_family=pattern_family,
        texture_map=texture_map,
        rtdetr_labels=rtdetr_labels,
        filename=os.path.basename(file_path),
        path=file_path,
    )
    dna_ms = (time.perf_counter() - t_dna) * 1000.0
    logger.info(
        "OWL_DECISION=%s reason=%s file_id=%s",
        gate["decision"],
        gate["reason"],
        int(file_id),
    )
    if gate["decision"] == "SKIP":
        store.upsert_open_vocab_objects(
            int(file_id), file_path, [],
            mtime=mtime, file_size=size, vocab_version=vocab,
            model=model, model_version=OVD_MODEL_VERSION, threshold=float(threshold),
            scan_status="skipped", scan_error=str(gate["reason"])[:300], last_path=image_path,
        )
        return {
            "ok": True,
            "skipped": gate["reason"],
            "boxes": 0,
            "source": source,
            "dna_ms": dna_ms,
            "owl_decision": "SKIP",
        }

    # Preview is already raster (webp/jpg/png). Do not open original EPS/PDF/AI.
    ext = os.path.splitext(image_path)[1].lower()
    if source != "feature_previews" and ext in {".pdf", ".eps", ".ai", ".svg", ".cdr"}:
        store.upsert_open_vocab_objects(
            int(file_id), file_path, [],
            mtime=mtime, file_size=size, vocab_version=vocab,
            model=model, model_version=OVD_MODEL_VERSION, threshold=float(threshold),
            scan_status="unsupported", scan_error="non_raster", last_path=image_path,
        )
        return {"ok": True, "skipped": "non_raster", "boxes": 0, "source": source}

    if backend is None:
        from core.ovd_backend import get_owlv2_backend

        backend = get_owlv2_backend(use_gpu=bool(use_gpu))
    from PIL import Image

    try:
        with Image.open(image_path) as raw_im:
            im = raw_im.convert("RGB")
    except Exception as exc:
        store.upsert_open_vocab_objects(
            int(file_id), file_path, [],
            mtime=mtime, file_size=size, vocab_version=vocab,
            model=model, model_version=OVD_MODEL_VERSION, threshold=float(threshold),
            scan_status="failed", scan_error=str(exc)[:300], last_path=image_path,
        )
        return {"ok": False, "skipped": "unreadable", "boxes": 0, "source": source, "error": str(exc)[:300]}
    w, h = im.size
    if w * h > 12_000_000:
        store.upsert_open_vocab_objects(
            int(file_id), file_path, [],
            mtime=mtime, file_size=size, vocab_version=vocab,
            model=model, model_version=OVD_MODEL_VERSION, threshold=float(threshold),
            scan_status="unsupported", scan_error="image_too_large", last_path=image_path,
        )
        return {"ok": True, "skipped": "image_too_large", "boxes": 0, "source": source}
    logger.info("OWL_SOURCE=%s file_id=%s path=%s", source, int(file_id), image_path)
    logger.info("INFERENCE_START file_id=%s path=%s", int(file_id), image_path)
    t_inf = time.perf_counter()
    raw = backend.detect(im, list(INDEX_VOCAB), float(threshold)) or []
    inf_ms = (time.perf_counter() - t_inf) * 1000.0
    logger.info(
        "INFERENCE_END file_id=%s ms=%.1f raw_boxes=%s",
        int(file_id),
        inf_ms,
        len(raw),
    )
    kept = filter_box_area(raw, float(w * h), OVD_MAX_BOX_AREA)
    kept = [b for b in kept if float(b.get("confidence") or 0) >= float(threshold)]
    kept = nms_boxes(kept)
    rec = {"role": "", "pattern_family": pattern_family or "", "texture_map": texture_map or {}}
    rows = []
    t_clip = time.perf_counter()
    for b in kept:
        xy = b.get("xyxy") or [0, 0, 0, 0]
        lab = str(b.get("label") or "").strip().lower().rstrip(".")
        if lab not in INDEX_VOCAB:
            for v in INDEX_VOCAB:
                if v in lab or lab in v:
                    lab = v
                    break
        if lab not in OPEN_CONCEPTS:
            continue
        clip_meta = clip_fn(im, [float(x) for x in xy], lab) if clip_fn is not None else None
        judged = judge_box(rec, lab, {
            "confidence": float(b.get("confidence") or 0),
            "xyxy_norm": _norm_bbox([float(x) for x in xy], w, h),
        }, clip_meta)
        if judged["evidence_type"] != "OPEN_VOCAB_OBJECT":
            continue
        rows.append({
            "label": lab,
            "label_tr": _TR.get(lab, lab),
            "confidence": float(b.get("confidence") or 0),
            "bbox": _norm_bbox([float(x) for x in xy], w, h),
            "area_ratio": float(b.get("bbox_area_ratio") or 0),
        })
    clip_ms = (time.perf_counter() - t_clip) * 1000.0
    logger.info("DB_WRITE file_id=%s boxes=%s", int(file_id), len(rows))
    t_db = time.perf_counter()
    n = store.upsert_open_vocab_objects(
        int(file_id),
        file_path,
        rows,
        mtime=mtime,
        file_size=size,
        vocab_version=vocab,
        model=model,
        model_version=OVD_MODEL_VERSION,
        threshold=float(threshold),
        scan_status="done",
        last_path=image_path,
    )
    db_ms = (time.perf_counter() - t_db) * 1000.0
    logger.info(
        "COMMIT file_id=%s box_count=%s db_ms=%.1f inf_ms=%.1f source=%s",
        int(file_id),
        n,
        db_ms,
        inf_ms,
        source,
    )
    return {
        "ok": True,
        "skipped": "",
        "boxes": n,
        "labels": [r["label"] for r in rows],
        "source": source,
        "preview_lookup_ms": round(lookup_ms, 2),
        "inference_ms": round(inf_ms, 1),
        "clip_ms": round(clip_ms, 1),
        "dna_ms": round(dna_ms, 2),
        "db_ms": round(db_ms, 2),
        "owl_decision": "RUN",
    }
