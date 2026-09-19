"""Index doğrulama ve onarım — eksik parçaları listele/tamamla."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.db import Database
from core.logger import setup_logger

logger = setup_logger(__name__)

CHECKS = (
    "thumbnail",
    "preview",
    "pattern_dna",
    "ocr",
    "embedding",
    "semantic",
    "duplicate_info",
)


def _has_dna(tm: dict) -> bool:
    dna = tm.get("pattern_dna")
    return isinstance(dna, dict) and bool(dna) and dna != {}


def _has_semantic(tm: dict) -> bool:
    sem = tm.get("semantic_tags")
    return isinstance(sem, dict) and bool(sem)


def verify_index(settings, *, sample_limit: int = 5000) -> dict[str, Any]:
    """Eksik bileşenleri say — dosya silinmez."""
    db = Database(settings.db_path)
    dash = db.count_index_pipeline_dashboard()
    issues: dict[str, int] = {k: 0 for k in CHECKS}
    samples: dict[str, list[dict[str, Any]]] = {k: [] for k in CHECKS}

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.filename, f.path, f.thumbnail_path, f.feature_preview_path,
                   f.ocr_text, fe.dino_embedding, fe.clip_embedding, fe.texture_map
            FROM files f
            LEFT JOIN features fe ON fe.file_id=f.id
            WHERE f.status='indexed'
            LIMIT ?
            """,
            (int(sample_limit),),
        ).fetchall()

    import json

    for row in rows:
        rec = dict(row)
        fid = int(rec["id"])
        tm_raw = rec.get("texture_map")
        try:
            tm = json.loads(tm_raw) if isinstance(tm_raw, str) else (tm_raw or {})
        except json.JSONDecodeError:
            tm = {}

        def _sample(key: str) -> None:
            if len(samples[key]) < 5:
                samples[key].append({"id": fid, "filename": rec.get("filename", "")})

        thumb = str(rec.get("thumbnail_path") or "")
        if not thumb or not os.path.isfile(thumb):
            issues["thumbnail"] += 1
            _sample("thumbnail")

        preview = str(rec.get("feature_preview_path") or "")
        if not preview or not os.path.isfile(preview):
            issues["preview"] += 1
            _sample("preview")

        if not _has_dna(tm if isinstance(tm, dict) else {}):
            issues["pattern_dna"] += 1
            _sample("pattern_dna")

        if settings.ocr_enabled and not str(rec.get("ocr_text") or "").strip():
            issues["ocr"] += 1
            _sample("ocr")

        has_emb = bool(rec.get("dino_embedding")) or bool(rec.get("clip_embedding"))
        if settings.ai_embedding_enabled and not has_emb:
            issues["embedding"] += 1
            _sample("embedding")

        if not _has_semantic(tm if isinstance(tm, dict) else {}):
            issues["semantic"] += 1
            _sample("semantic")

        if not (tm if isinstance(tm, dict) else {}).get("duplicate_info"):
            issues["duplicate_info"] += 1
            _sample("duplicate_info")

    return {
        "ok": sum(issues.values()) == 0,
        "sampled": len(rows),
        "dashboard": dash,
        "issues": issues,
        "samples": samples,
        "checks": list(CHECKS),
    }


def repair_index(
    settings,
    report: dict[str, Any] | None = None,
    *,
    only: list[str] | None = None,
) -> dict[str, Any]:
    """Yalnızca eksik parçaları kuyruğa al / bakım çalıştır."""
    from core.index_maintenance import (
        backfill_missing_semantic_tags,
        purge_broken_index_records,
        queue_missing_ai_embeddings,
        requeue_missing_thumbnails,
    )

    rep = report or verify_index(settings)
    issues = rep.get("issues") or {}
    targets = only or [k for k, v in issues.items() if int(v or 0) > 0]
    results: dict[str, Any] = {"repaired": {}, "skipped": []}

    if "thumbnail" in targets or "preview" in targets:
        results["repaired"]["thumbnails"] = requeue_missing_thumbnails(settings)
    if "embedding" in targets:
        results["repaired"]["embeddings"] = queue_missing_ai_embeddings(settings)
    if "semantic" in targets or "pattern_dna" in targets:
        results["repaired"]["semantic_dna"] = backfill_missing_semantic_tags(
            settings, limit=500
        )
    if "ocr" in targets:
        db = Database(settings.db_path)
        results["repaired"]["ocr"] = db.queue_files_missing_ocr(limit=500)
    if "duplicate_info" in targets:
        results["repaired"]["duplicate_info"] = {
            "note": "heavy index pass ile duplicate_info doldurulur",
            "queued_full": Database(settings.db_path).count_pending_full(),
        }

    results["repaired"]["purge_broken"] = purge_broken_index_records(settings)
    logger.info("Index repair: %s", results)
    return results
