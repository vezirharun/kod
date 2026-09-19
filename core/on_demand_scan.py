"""Arama öncesi hızlı index — sorgu dosyası ve yakın bekleyen dosyalar."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.logger import setup_logger
from core.settings import AppSettings
from core.utils import normalize_path

logger = setup_logger(__name__)


def _folder_prefix(path: str) -> str:
    norm = normalize_path(path)
    parent = os.path.dirname(norm)
    return parent.casefold().rstrip("\\/") + "\\"


def should_prompt_query_folder_index(_query_path: str = "") -> bool:
    """Sorgu görseli hedef koleksiyon değildir; klasör index diyaloğu açılmaz."""
    return False


def query_needs_index(record: dict[str, Any] | None, settings: AppSettings) -> bool:
    if not record:
        return True
    if record.get("status") != "indexed":
        return True
    if settings.preflight_reindex_query_if_incomplete:
        from core.indexer import Indexer

        return Indexer(settings).needs_search_features(record)
    return False


def collect_preflight_paths(
    settings: AppSettings,
    image_path: str,
    *,
    batch_size: int | None = None,
    query_only: bool = False,
) -> list[str]:
    from core.db import Database

    batch_size = batch_size or settings.preflight_batch_size
    norm = normalize_path(image_path)
    if not norm:
        return []

    db = Database(settings.db_path)
    rec = db.get_file_by_path(norm)
    if not rec:
        return []
    if query_only:
        return [norm] if query_needs_index(rec, settings) else []

    paths: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        key = normalize_path(path).casefold()
        if key and key not in seen and os.path.isfile(path):
            seen.add(key)
            paths.append(path)

    add(norm)

    record = db.get_file_by_path(norm)
    source_id = int(record["source_id"]) if record and record.get("source_id") else None

    if settings.preflight_folder_siblings:
        prefix = _folder_prefix(norm)
        rows = db.get_processing_files(source_id=source_id, limit=batch_size * 3)
        for row in rows:
            if len(paths) >= batch_size + 1:
                break
            row_path = row.get("path") or ""
            if row_path.casefold().startswith(prefix):
                add(row_path)
    else:
        rows = db.get_processing_files(source_id=source_id, limit=batch_size)
        for row in rows:
            if len(paths) >= batch_size + 1:
                break
            add(row.get("path") or "")

    return paths[: batch_size + 1]


def run_search_preflight(
    settings: AppSettings,
    image_path: str,
    progress_callback=None,
    *,
    query_only: bool | None = None,
) -> dict[str, Any]:
    """Görsel aramadan önce sorgu + yakın bekleyen dosyaları tam özellikle indexle."""
    from core.index_freeze import search_write_protection_active

    # B only: INDEX_FROZEN (A) alone must not skip on-demand index forever.
    if search_write_protection_active():
        return {"skipped": True, "reason": "search_session"}
    if not getattr(settings, "auto_preflight_index_on_search", True):
        return {"skipped": True, "reason": "disabled"}

    norm = normalize_path(image_path)
    if not norm or not os.path.isfile(norm):
        return {"skipped": True, "reason": "missing_file"}

    from core.db import Database
    from core.indexer import Indexer

    db = Database(settings.db_path)
    indexer = Indexer(settings)
    record = db.get_file_by_path(norm)
    if not record:
        return {"skipped": True, "reason": "external_query"}
    if query_only is None:
        query_only = bool(getattr(settings, "preflight_query_only_on_search", True))
    paths = collect_preflight_paths(settings, norm, query_only=query_only)

    if not paths:
        return {
            "skipped": True,
            "reason": "query_ready",
            "query_indexed": bool(record and record.get("status") == "indexed"),
            "paths_considered": 0,
        }

    if record and not query_needs_index(record, settings) and len(paths) <= 1:
        return {
            "skipped": True,
            "reason": "query_ready",
            "query_indexed": True,
            "paths_considered": 1,
        }

    stats = {
        "paths_considered": len(paths),
        "processed": 0,
        "errors": 0,
        "query_indexed": False,
    }

    def emit(**extra):
        if progress_callback:
            progress_callback({"stage": "preflight_index", **stats, **extra})

    emit(current_file=Path(norm).name)
    prev_fast = settings.fast_hash_only
    try:
        settings.fast_hash_only = False
        indexer._ensure_extractor(
            use_ai=settings.index_uses_ai(),
            fast_hash_only=False,
        )
        batch_stats = indexer.index_paths(paths, deep=False)
        stats["processed"] = int(batch_stats.get("processed", 0))
        stats["errors"] = int(batch_stats.get("errors", 0))
    finally:
        settings.fast_hash_only = prev_fast
        indexer._ensure_extractor(
            use_ai=settings.index_uses_ai(),
            fast_hash_only=prev_fast,
        )

    after = db.get_file_by_path(norm)
    stats["query_indexed"] = bool(
        after
        and after.get("status") == "indexed"
        and not indexer.needs_search_features(after)
    )
    emit(done=True)
    logger.info(
        "Preflight index: %s dosya, %s işlendi, sorgu hazır=%s",
        len(paths),
        stats["processed"],
        stats["query_indexed"],
    )
    return stats
