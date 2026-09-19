"""Backfill Görsel Nesne/Kavram DNA from existing Feature Preview only.

Does not re-run HASH/DINO/CLIP, does not read NAS originals, does not touch FAISS.
INDEX_FROZEN / search_session skips writes.
"""
from __future__ import annotations

import os
from typing import Any

from core.db import Database
from core.index_freeze import in_search_session, process_search_active
from core.index_object_evidence import attach_preview_object_evidence
from core.object_index import ObjectIndexStore
from core.settings import AppSettings


def list_object_dna_gaps(db: Database, *, limit: int = 20) -> list[dict[str, Any]]:
    """Files with a local feature preview and CLIP blob, missing object scan."""
    sql = """
        SELECT f.id AS file_id, f.feature_preview_path AS preview_path, f.filename
        FROM files f
        JOIN features fe ON fe.file_id = f.id
        WHERE f.status NOT IN ('missing','excluded_internal')
          AND COALESCE(f.feature_preview_path,'') != ''
          AND fe.clip_embedding IS NOT NULL
          AND length(fe.clip_embedding) > 0
        ORDER BY f.id
        LIMIT ?
    """
    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, (max(1, int(limit) * 8),)).fetchall()]
    return rows


def persist_goi_texture_map(db: Database, file_id: int, goi: dict[str, Any]) -> None:
    feat = db.get_features(int(file_id)) or {}
    tm = dict(feat.get("texture_map") or {})
    tm["global_object_intelligence"] = goi
    if goi.get("visual_concept_dna"):
        tm["visual_concept_dna"] = goi["visual_concept_dna"]
    db.upsert_features(int(file_id), {"texture_map": tm})


def _rows_for_file_ids(db: Database, file_ids: list[int]) -> list[dict[str, Any]]:
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    sql = f"""
        SELECT f.id AS file_id, f.feature_preview_path AS preview_path, f.filename
        FROM files f
        WHERE f.id IN ({q})
        ORDER BY f.id
    """
    with db.connect() as conn:
        return [dict(r) for r in conn.execute(sql, ids).fetchall()]


def backfill_visual_concept_dna(
    settings: AppSettings,
    *,
    limit: int = 20,
    detector=None,
    file_ids: list[int] | None = None,
    force: bool = False,
    require_rtdetr: bool = False,
) -> dict[str, Any]:
    """Process at most ``limit`` files. Never starts a full archive rebuild."""
    report: dict[str, Any] = {
        "preview_found": 0,
        "preview_missing": 0,
        "detector_ran": 0,
        "rtdetr_ran": 0,
        "objects_detected": 0,
        "visual_concept_dna_written": 0,
        "object_index_files": 0,
        "skipped": 0,
        "errors": 0,
        "error_samples": [],
        "file_ids": [],
        "backend": "",
        "stopped_search": False,
        "limit": int(limit),
        "rtdetr_load_failed": False,
    }
    if process_search_active() or in_search_session():
        report["stopped_search"] = True
        return report

    if require_rtdetr:
        from core.object_intelligence import DETECTOR_BACKEND, _load_rtdetr, _ultra_err

        if not _load_rtdetr():
            report["errors"] = 1
            report["rtdetr_load_failed"] = True
            report["backend"] = "unavailable"
            report["error_samples"].append(
                _ultra_err or "RT-DETR-L could not load; archive backfill not started"
            )
            return report
        report["backend"] = DETECTOR_BACKEND

    db = Database(settings.db_path)
    obj_path = str(getattr(settings, "object_db_path", "") or "")
    store = ObjectIndexStore(obj_path, readonly=False)
    if file_ids is not None:
        candidates = _rows_for_file_ids(db, list(file_ids))
        if int(limit) > 0:
            candidates = candidates[: int(limit)]
    else:
        candidates = list_object_dna_gaps(db, limit=limit)
    done = 0
    for row in candidates:
        if done >= int(limit):
            break
        if process_search_active() or in_search_session():
            report["stopped_search"] = True
            break
        fid = int(row.get("file_id") or 0)
        preview = str(row.get("preview_path") or "").strip()
        if not force and store.has_scan(fid):
            report["skipped"] += 1
            continue
        if not preview or not os.path.isfile(preview):
            report["preview_missing"] += 1
            continue
        report["preview_found"] += 1
        try:
            goi = attach_preview_object_evidence(
                file_id=fid,
                preview_path=preview,
                object_db_path=obj_path,
                min_confidence=float(
                    getattr(settings, "global_object_detection_min_confidence", 0.45)
                    or 0.45
                ),
                detector=detector,
                force=force,
                require_rtdetr=require_rtdetr,
            )
            if not goi:
                report["skipped"] += 1
                continue
            persist_goi_texture_map(db, fid, goi)
            report["visual_concept_dna_written"] += 1
            report["objects_detected"] += int(goi.get("object_count") or 0)
            if goi.get("backend") and goi.get("backend") != "unavailable":
                report["detector_ran"] += 1
                if "rtdetr" in str(goi.get("backend") or "").lower():
                    report["rtdetr_ran"] += 1
            report["backend"] = str(goi.get("backend") or report["backend"])
            report["file_ids"].append(fid)
            done += 1
        except Exception as exc:
            report["errors"] += 1
            if len(report["error_samples"]) < 8:
                report["error_samples"].append(f"{fid}: {type(exc).__name__}: {exc}")
    try:
        report["object_index_files"] = int(
            __import__("sqlite3")
            .connect(obj_path)
            .execute("SELECT COUNT(*) FROM object_files")
            .fetchone()[0]
        )
    except Exception:
        report["object_index_files"] = 0
    return report
