"""Index bakım — bozuk kayıt temizliği ve kuyruk yönetimi."""

from __future__ import annotations

from typing import Any

import json

from core.db import Database
from core.index_stages import FAILED, FULL_DONE, LIGHT_DONE, PENDING_FULL, PENDING_LIGHT
from core.settings import AppSettings


def purge_broken_index_records(settings: AppSettings) -> dict[str, int]:
    db = Database(settings.db_path)
    return db.purge_broken_index_records()


def requeue_missing_thumbnails(settings: AppSettings) -> dict[str, int]:
    db = Database(settings.db_path)
    return db.requeue_files_missing_thumbnails()


def queue_missing_ai_embeddings(settings: AppSettings) -> dict[str, int]:
    db = Database(settings.db_path)
    if not settings.ai_embedding_enabled:
        return {"queued": 0, "reason": "ai_disabled"}
    return db.queue_files_missing_ai_embeddings()


def count_stage_summary(settings: AppSettings) -> dict[str, int]:
    db = Database(settings.db_path)
    return db.count_by_index_stage()


def backfill_missing_semantic_tags(
    settings: AppSettings,
    *,
    limit: int = 0,
    progress_callback=None,
) -> dict[str, int | str]:
    """Populate missing semantic metadata without opening source images or loading AI."""
    if not getattr(settings, "semantic_text_search_enabled", False):
        return {"processed": 0, "remaining": 0, "reason": "semantic_disabled"}

    from core.semantic_tags import build_semantic_tags
    from core.classification_pipeline import (
        apply_identity_to_texture_map,
        resolve_pattern_identity,
    )
    from core.pattern_dna import apply_dna_to_texture_map, build_pattern_dna
    from core.category_predictions import category_predictions_from_metadata
    from core.text_index import build_text_search_blob, extract_index_fields

    db = Database(settings.db_path)
    processed = 0
    batch_size = 500
    while True:
        remaining_limit = max(0, int(limit) - processed) if limit else batch_size
        take = min(batch_size, remaining_limit) if limit else batch_size
        if take <= 0:
            break
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.filename, f.path, f.customer, f.ocr_text,
                       f.source_id, f.category_path, f.manual_category_path,
                       fe.texture_map
                FROM files f
                JOIN features fe ON fe.file_id=f.id
                WHERE f.status='indexed'
                  AND fe.texture_map IS NOT NULL
                  AND fe.texture_map!=''
                  AND (
                    fe.texture_map NOT LIKE '%\"semantic_tags\"%'
                    OR fe.texture_map NOT LIKE '%\"pattern_dna\"%'
                    OR fe.texture_map NOT LIKE '%\"repeat_type\"%'
                  )
                ORDER BY f.id
                LIMIT ?
                """,
                (take,),
            ).fetchall()
        if not rows:
            break
        for row in rows:
            rec = dict(row)
            try:
                tm = json.loads(rec.get("texture_map") or "{}")
            except json.JSONDecodeError:
                continue
            category = str(rec.get("manual_category_path") or "")
            if not category and str(tm.get("category_source") or "") in (
                "manual_user", "gold_dataset"
            ):
                category = str(rec.get("category_path") or "")
            tm["semantic_tags"] = build_semantic_tags(
                tm,
                category_path=category,
                ocr_text=rec.get("ocr_text", "") or "",
                filename=rec.get("filename", ""),
            )
            feedback_labels = db.get_feedback_labels_for_file(int(rec["id"]))
            identity = resolve_pattern_identity(
                texture_map=tm,
                filename=rec.get("filename", ""),
                category_path=category,
                manual_category_path=str(rec.get("manual_category_path") or ""),
                feedback_labels=feedback_labels,
            )
            tm = apply_identity_to_texture_map(tm, identity)
            dna = build_pattern_dna(
                tm,
                semantic_tags=tm.get("semantic_tags"),
                category_path=category,
                source=identity.source,
            )
            tm = apply_dna_to_texture_map(tm, dna)
            tm["ai_category_predictions"] = category_predictions_from_metadata(
                str(tm.get("pattern_family") or ""),
                str(tm.get("animal_print_type") or tm.get("pattern_subtype") or ""),
                tm,
            )
            db.upsert_texture_map(int(rec["id"]), tm)
            source = db.get_source(int(rec.get("source_id") or 0)) or {}
            group = db.get_pattern_group_for_file(int(rec["id"])) or {}
            feedback = [
                *feedback_labels,
                *db.get_feedback_custom_tags_for_file(int(rec["id"])),
            ]
            pf, pt, tf = extract_index_fields(tm)
            blob = build_text_search_blob(
                filename=rec.get("filename", ""),
                path=rec.get("path", ""),
                customer=rec.get("customer", ""),
                source_name=source.get("name", ""),
                ocr_text=rec.get("ocr_text", "") or "",
                texture_map=tm,
                feedback_labels=feedback,
                group_label=group.get("label", ""),
                pattern_family=pf,
                pattern_type=pt,
                texture_family=tf,
                pattern_subtype=str(tm.get("pattern_subtype") or pt),
                category_path=category,
                category_aliases=list(tm.get("category_aliases") or []),
                semantic_enabled=True,
            )
            db.update_file_text_index(
                int(rec["id"]),
                pattern_family=pf,
                pattern_type=pt,
                texture_family=tf,
                pattern_subtype=str(tm.get("pattern_subtype") or pt),
                pattern_confidence=float(tm.get("classification_confidence", 0) or 0),
                text_search_blob=blob,
            )
            processed += 1
            if progress_callback and processed % 100 == 0:
                progress_callback({"processed": processed})
        if len(rows) < take or (limit and processed >= limit):
            break

    with db.connect() as conn:
        remaining = int(conn.execute(
            """
            SELECT COUNT(*) AS c FROM files f
            JOIN features fe ON fe.file_id=f.id
            WHERE f.status='indexed'
              AND fe.texture_map IS NOT NULL AND fe.texture_map!=''
              AND (
                fe.texture_map NOT LIKE '%\"semantic_tags\"%'
                OR fe.texture_map NOT LIKE '%\"pattern_dna\"%'
                OR fe.texture_map NOT LIKE '%\"repeat_type\"%'
              )
            """
        ).fetchone()["c"])
    return {"processed": processed, "remaining": remaining, "reason": "ok"}
