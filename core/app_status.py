"""Hafif durum sorgusu — index sırasında tam Indexer oluşturmadan."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from core.db import Database
from core.index_pipeline_status import build_pipeline_status
from core.logger import setup_logger
from core.pipeline_diagnostics import (
    build_embedding_status,
    build_full_done_status,
    build_ocr_status,
    format_pipeline_dashboard_log,
)
from core.settings import AppSettings

logger = setup_logger(__name__)

_status_ram: dict[str, Any] = {}
_status_ram_ts: float = 0.0
_status_ram_db_path = ""
_status_ram_scope = ""
_status_ram_lock = threading.Lock()
_STATUS_RAM_TTL_SEC = 5.0
_pipeline_db_refresh_lock = threading.Lock()
_pipeline_db_refresh_ts = 0.0
_PIPELINE_DB_REFRESH_SEC = 30.0
_physical_reconcile_ts = 0.0
_PHYSICAL_RECONCILE_SEC = 30.0
_physical_reconcile_lock = threading.Lock()

_BOUNDED_PIPELINE_KEYS = (
    "preview_ready",
    "verified_preview_ready",
    "embedding_ready",
    "db_dino_embeddings",
    "db_clip_embeddings",
    "pattern_dna_count",
    "semantic_tag_count",
    "ocr_done",
    "texture_done",
    "patch_embedding_ready",
    "ai_final_ready",
    "preview_db_path_ready",
    "physical_preview_missing",
    "physical_preview_unverified",
)


def _classify_remaining_reason(row: dict[str, Any], executable_file_ids: set[int]) -> str:
    from core.quarantine import (
        DECODE_ERROR,
        FILE_CORRUPT,
        FORMAT_UNSUPPORTED,
        NETWORK_TIMEOUT,
        PATH_INVALID,
        PERMISSION_DENIED,
        TRUNCATED,
    )

    fid = int(row.get("id") or 0)
    if str(row.get("light_status") or "") == "processing":
        return "processing"
    if fid in executable_file_ids:
        return "queued"
    reason = str(row.get("repair_failure_reason") or row.get("quarantine_reason") or "").strip()
    if not reason:
        err = str(row.get("error_msg") or "").lower()
        if "permission" in err or "access is denied" in err:
            reason = PERMISSION_DENIED
        elif "timeout" in err:
            reason = NETWORK_TIMEOUT
        elif "cannot identify" in err or "decode" in err:
            reason = DECODE_ERROR
        elif "truncated" in err:
            reason = TRUNCATED
        elif "corrupt" in err:
            reason = FILE_CORRUPT
        elif "unsupported" in err:
            reason = FORMAT_UNSUPPORTED
        elif "invalid" in err:
            reason = PATH_INVALID
    if int(row.get("repair_retryable", 1) or 0) and reason:
        return "retryable"
    if reason == FORMAT_UNSUPPORTED:
        return "unsupported"
    if reason in (DECODE_ERROR, TRUNCATED, FILE_CORRUPT):
        return "decode_error"
    if reason == PERMISSION_DENIED:
        return "permission_error"
    if reason in (NETWORK_TIMEOUT, "source_unavailable", "io_timeout", "nas_timeout"):
        return "retryable"
    if reason == PATH_INVALID:
        return "invalid_file"
    # Status refresh NAS/disk taramaz; exists UI/worker'ı kilitlemesin.
    return "blocked"


def _remaining_reason_summary(
    settings: AppSettings,
    db: Database,
    source_ids: list[int],
) -> dict[str, Any]:
    scope_sql = ""
    params: list[Any] = []
    if source_ids:
        ph = ",".join("?" * len(source_ids))
        scope_sql = f" AND source_id IN ({ph})"
        params.extend(int(x) for x in source_ids)

    jobs_path = Path(settings.db_path).with_name(
        Path(settings.db_path).stem + ".v3jobs.db"
    )
    executable_file_ids: set[int] = set()
    if jobs_path.exists():
        with sqlite3.connect(jobs_path) as conn:
            if source_ids:
                jph = ",".join("?" * len(source_ids))
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT file_id
                    FROM index_v3_jobs
                    WHERE state='pending' AND source_id IN ({jph})
                    """,
                    [int(x) for x in source_ids],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT DISTINCT file_id FROM index_v3_jobs WHERE state='pending'"
                ).fetchall()
        executable_file_ids = {int(r[0]) for r in rows}

    with db.connect() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                f"""
                SELECT id, path, status, light_status, heavy_status,
                       quarantine_reason, error_msg, repair_failure_reason,
                       repair_last_error, repair_retryable, updated_at
                FROM files
                WHERE status NOT IN ('excluded_internal','missing')
                  AND light_status!='done'
                  {scope_sql}
                ORDER BY id
                """,
                params,
            ).fetchall()
        ]

    order = (
        "queued",
        "processing",
        "retryable",
        "decode_error",
        "unsupported",
        "permission_error",
        "invalid_file",
        "missing",
        "blocked",
    )
    labels = {
        "queued": "kuyruğa hazır",
        "processing": "işleniyor",
        "retryable": "retryable",
        "decode_error": "bozuk/decode",
        "unsupported": "desteklenmeyen",
        "permission_error": "izin hatası",
        "invalid_file": "geçersiz yol/dosya",
        "missing": "eksik",
        "blocked": "bloklu",
    }
    counts = {key: 0 for key in order}
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in order}
    for row in rows:
        reason = _classify_remaining_reason(row, executable_file_ids)
        counts.setdefault(reason, 0)
        counts[reason] += 1
        bucket = samples.setdefault(reason, [])
        if len(bucket) < 5:
            bucket.append(
                {
                    "id": int(row.get("id") or 0),
                    "path": str(row.get("path") or ""),
                    "status": str(row.get("status") or ""),
                    "light_status": str(row.get("light_status") or ""),
                    "reason": reason,
                    "error": str(
                        row.get("repair_last_error")
                        or row.get("error_msg")
                        or row.get("repair_failure_reason")
                        or row.get("quarantine_reason")
                        or ""
                    )[:200],
                    "updated_at": str(row.get("updated_at") or ""),
                }
            )
    ordered_counts = {key: int(counts.get(key, 0) or 0) for key in order if counts.get(key, 0)}
    summary = " · ".join(
        f"{ordered_counts[key]:,} {labels.get(key, key)}" for key in ordered_counts
    )
    return {
        "remaining_light_total": len(rows),
        "remaining_light_executable_files": len(executable_file_ids),
        "remaining_reason_counts": ordered_counts,
        "remaining_reason_samples": {k: v for k, v in samples.items() if v},
        "remaining_reason_summary": summary,
    }


def _progress_source_ids(settings: AppSettings) -> list[int] | None:
    """Deprecated path — resolve_index_scope kullan; boş seçim = tüm arşiv id listesi."""
    from core.db import Database
    from core.index_v3.scope import resolve_index_scope

    selected = [int(x) for x in (settings.selected_source_ids or []) if int(x) > 0]
    db = Database(settings.db_path, read_only=True)
    return list(resolve_index_scope(db, selected).source_ids)


def _status_scope_key(settings: AppSettings) -> str:
    ids = _progress_source_ids(settings)
    return ",".join(str(i) for i in sorted(ids)) if ids else "*"


def get_cached_status(*, db_path: str | None = None) -> dict[str, Any]:
    """RAM'deki son status — SQLite yok."""
    with _status_ram_lock:
        if (
            db_path is not None
            and _status_ram_db_path
            and _status_ram_db_path != str(db_path)
        ):
            return {}
        return dict(_status_ram) if _status_ram else {}


def put_cached_status(
    status: dict[str, Any],
    *,
    db_path: str | None = None,
    scope: str | None = None,
) -> None:
    global _status_ram, _status_ram_ts, _status_ram_db_path, _status_ram_scope
    with _status_ram_lock:
        _status_ram = dict(status)
        _status_ram_ts = time.time()
        _status_ram_db_path = str(db_path) if db_path is not None else ""
        if scope is not None:
            _status_ram_scope = scope


def invalidate_status_cache() -> None:
    """Kaynak seçimi değişince eski payda cache'ini düşür."""
    global _status_ram, _status_ram_ts, _status_ram_scope
    with _status_ram_lock:
        _status_ram = {}
        _status_ram_ts = 0.0
        _status_ram_scope = ""


def _refresh_exact_pipeline_counts(
    settings: AppSettings, base: dict[str, Any], total: int
) -> dict[str, Any]:
    """Event değil, benzersiz files/features satırlarından kesin sayaçlar."""
    db = Database(settings.db_path)
    scope_ids = _progress_source_ids(settings)
    dashboard = db.count_index_pipeline_dashboard(scope_ids)
    embeddings = db.count_embeddings()
    physical = db.count_physical_readiness()
    pipeline = build_pipeline_status(dashboard, archive_total=total)
    refreshed = {
        **base,
        **dashboard,
        **pipeline,
        "db_dino_embeddings": int(
            pipeline.get("db_dino_embeddings", embeddings.get("dino", 0)) or 0
        ),
        "db_clip_embeddings": int(
            pipeline.get("db_clip_embeddings", embeddings.get("clip", 0)) or 0
        ),
        "patch_embedding_ready": int(
            pipeline.get("patch_embedding_ready", 0) or 0
        ),
        "ai_final_ready": int(pipeline.get("ai_final_ready", 0) or 0),
        "physical_ready": int(physical.get("physical_ready", 0) or 0),
        "physical_missing": int(physical.get("physical_missing", 0) or 0),
        "physical_unverified": int(physical.get("physical_unverified", 0) or 0),
        "physical_sync_total": int(physical.get("total", 0) or 0),
        "physical_last_verified_at": str(
            physical.get("last_verified_at", "") or ""
        ),
        "pipeline_counts_exact": True,
        "status_scope_source_ids": list(scope_ids or []),
    }
    for key in _BOUNDED_PIPELINE_KEYS:
        value = max(0, int(refreshed.get(key, 0) or 0))
        if total >= 0 and value > total:
            logger.error(
                "pipeline_counter_out_of_bounds key=%s value=%s total=%s; clamped",
                key,
                value,
                total,
            )
            value = total
        refreshed[key] = value
    return refreshed


def get_lightweight_status(settings: AppSettings, *, force_db: bool = False) -> dict:
    """UI yenileme — önce RAM cache; SQLite yalnızca TTL dolunca."""
    global _status_ram, _status_ram_ts, _pipeline_db_refresh_ts
    scope = _status_scope_key(settings)
    with _status_ram_lock:
        age = time.time() - _status_ram_ts if _status_ram else 9999.0
        same_db = not _status_ram_db_path or _status_ram_db_path == str(
            settings.db_path
        )
        same_scope = _status_ram_scope == scope
        if (
            not force_db
            and same_db
            and same_scope
            and _status_ram
            and age < _STATUS_RAM_TTL_SEC
        ):
            cached = dict(_status_ram)
            cached["from_ram_cache"] = True
            return cached

    status = _fetch_lightweight_status_from_db(settings)
    put_cached_status(status, db_path=settings.db_path, scope=scope)
    _pipeline_db_refresh_ts = time.monotonic()
    status["from_ram_cache"] = False
    return status


def refresh_lane_counts(settings: AppSettings) -> dict:
    """Index sırasında hızlı SSOT — V3 artifact READY (event counter yok).

    Bilerek reconcile YOK: index yazarken her saniye disk+UPDATE kilidi
    worker'ı düşürüyordu (UI: Durdu).
    """
    from core.index_v3.scope import resolve_index_scope
    from core.index_v3.ui_bridge import v3_status_dict
    from core.index_v3.queues import JobStore
    from core.index_v3.types import QueueKind

    global _pipeline_db_refresh_ts
    scope_key = _status_scope_key(settings)
    selected = [int(x) for x in (settings.selected_source_ids or []) if int(x) > 0]
    db = Database(settings.db_path)
    resolved = resolve_index_scope(db, selected)
    jobs_path = Path(settings.db_path).with_name(
        Path(settings.db_path).stem + ".v3jobs.db"
    )
    job_store = JobStore(jobs_path)
    pending_jobs = job_store.count_pending(source_ids=resolved.source_ids)
    pending_light_jobs = sum(
        job_store.count_pending(queue, source_ids=resolved.source_ids)
        for queue in (QueueKind.LIGHT, QueueKind.PREVIEW)
    )
    pending_heavy_jobs = sum(
        job_store.count_pending(queue, source_ids=resolved.source_ids)
        for queue in (QueueKind.HEAVY, QueueKind.REPAIR)
    )
    # UI İşleniyor: claimed state (fresh şartı yok — uzun PATCH heartbeat gecikmesinde 0 olmasın)
    claimed_light = job_store.count_claimed(
        source_ids=resolved.source_ids,
        queues=(QueueKind.LIGHT, QueueKind.PREVIEW),
    )
    claimed_heavy = job_store.count_claimed(
        source_ids=resolved.source_ids,
        queues=(QueueKind.HEAVY, QueueKind.REPAIR),
    )
    claimed_jobs = claimed_light + claimed_heavy
    claimed_fresh_jobs = job_store.count_claimed_fresh(source_ids=resolved.source_ids)
    jobs_done_light_1m = job_store.count_completed_recent(
        60.0,
        source_ids=resolved.source_ids,
        queues=(QueueKind.LIGHT, QueueKind.PREVIEW),
    )
    jobs_done_heavy_1m = job_store.count_completed_recent(
        60.0,
        source_ids=resolved.source_ids,
        queues=(QueueKind.HEAVY, QueueKind.REPAIR),
    )
    retry_jobs = job_store.count_pending_retry(source_ids=resolved.source_ids)
    failed_permanent_jobs = job_store.count_failed_permanent(
        source_ids=resolved.source_ids
    )
    failed_permanent_files = job_store.count_failed_permanent_files(
        source_ids=resolved.source_ids
    )
    merged = v3_status_dict(
        db,
        resolved.source_ids,
        pending_jobs=pending_jobs,
        processing=claimed_jobs,
        claimed=claimed_jobs,
        light_processing=claimed_light,
        heavy_processing=claimed_heavy,
        failed_permanent_files=failed_permanent_files,
        scope=resolved,
    )
    merged.update(
        {
            "claimed_jobs": claimed_jobs,
            "claimed_fresh_jobs": claimed_fresh_jobs,
            "claimed_light_jobs": claimed_light,
            "claimed_heavy_jobs": claimed_heavy,
            "retry_jobs": retry_jobs,
            "failed_permanent_jobs": failed_permanent_jobs,
            "failed_permanent_files": failed_permanent_files,
            "pending_light_jobs": pending_light_jobs,
            "pending_heavy_jobs": pending_heavy_jobs,
            "pending_heavy_files": job_store.count_pending_files(
                queues=(QueueKind.HEAVY, QueueKind.REPAIR),
                source_ids=resolved.source_ids,
            ),
            "jobs_completed_light_1m": jobs_done_light_1m,
            "jobs_completed_heavy_1m": jobs_done_heavy_1m,
            "light_job_speed_pm": float(jobs_done_light_1m),
            "heavy_job_speed_pm": float(jobs_done_heavy_1m),
        }
    )
    _pipeline_db_refresh_ts = time.monotonic()
    put_cached_status(merged, db_path=settings.db_path, scope=scope_key)
    return merged


def _fetch_lightweight_status_from_db(settings: AppSettings) -> dict:
    """UI yenileme — V3 artifact SSOT + kaynak listesi."""
    from core.index_v3.physical_reconcile import reconcile_stale_physical_flags
    from core.index_v3.scope import orphan_source_report, resolve_index_scope
    from core.index_v3.ui_bridge import v3_status_dict
    from core.index_v3.queues import JobStore
    from core.index_v3.types import QueueKind
    from core.quarantine import backfill_failed_quarantine_reasons, failed_reason_report

    db = Database(settings.db_path)
    selected = [int(x) for x in (settings.selected_source_ids or []) if int(x) > 0]
    resolved = resolve_index_scope(db, selected)
    # Idle UI yenilemede throttle — index hot-path'te değil
    global _physical_reconcile_ts
    now = time.monotonic()
    with _physical_reconcile_lock:
        due = (now - _physical_reconcile_ts) >= _PHYSICAL_RECONCILE_SEC
        if due:
            _physical_reconcile_ts = now
    if due:
        try:
            reconcile_stale_physical_flags(db, resolved.source_ids)
        except Exception:
            pass
    try:
        backfill_failed_quarantine_reasons(db, limit=5000)
    except Exception:
        pass
    counts = db.count_by_status()
    quarantine_reasons = db.count_quarantine_by_reason()
    jobs_path = Path(settings.db_path).with_name(
        Path(settings.db_path).stem + ".v3jobs.db"
    )
    job_store = JobStore(jobs_path)
    pending_jobs = job_store.count_pending(source_ids=resolved.source_ids)
    pending_light_jobs = sum(
        job_store.count_pending(queue, source_ids=resolved.source_ids)
        for queue in (QueueKind.LIGHT, QueueKind.PREVIEW)
    )
    pending_heavy_jobs = sum(
        job_store.count_pending(queue, source_ids=resolved.source_ids)
        for queue in (QueueKind.HEAVY, QueueKind.REPAIR)
    )
    claimed_light = job_store.count_claimed(
        source_ids=resolved.source_ids,
        queues=(QueueKind.LIGHT, QueueKind.PREVIEW),
    )
    claimed_heavy = job_store.count_claimed(
        source_ids=resolved.source_ids,
        queues=(QueueKind.HEAVY, QueueKind.REPAIR),
    )
    claimed_jobs = claimed_light + claimed_heavy
    claimed_fresh_jobs = job_store.count_claimed_fresh(source_ids=resolved.source_ids)
    jobs_done_light_1m = job_store.count_completed_recent(
        60.0,
        source_ids=resolved.source_ids,
        queues=(QueueKind.LIGHT, QueueKind.PREVIEW),
    )
    jobs_done_heavy_1m = job_store.count_completed_recent(
        60.0,
        source_ids=resolved.source_ids,
        queues=(QueueKind.HEAVY, QueueKind.REPAIR),
    )
    retry_jobs = job_store.count_pending_retry(source_ids=resolved.source_ids)
    failed_permanent_jobs = job_store.count_failed_permanent(
        source_ids=resolved.source_ids
    )
    failed_permanent_files = job_store.count_failed_permanent_files(
        source_ids=resolved.source_ids
    )
    v3 = v3_status_dict(
        db,
        resolved.source_ids,
        pending_jobs=pending_jobs,
        processing=claimed_jobs,
        claimed=claimed_jobs,
        light_processing=claimed_light,
        heavy_processing=claimed_heavy,
        failed_permanent_files=failed_permanent_files,
        scope=resolved,
    )
    v3.update(
        {
            "claimed_jobs": claimed_jobs,
            "claimed_fresh_jobs": claimed_fresh_jobs,
            "claimed_light_jobs": claimed_light,
            "claimed_heavy_jobs": claimed_heavy,
            "retry_jobs": retry_jobs,
            "failed_permanent_jobs": failed_permanent_jobs,
            "failed_permanent_files": failed_permanent_files,
            "pending_light_jobs": pending_light_jobs,
            "pending_heavy_jobs": pending_heavy_jobs,
            "pending_heavy_files": job_store.count_pending_files(
                queues=(QueueKind.HEAVY, QueueKind.REPAIR),
                source_ids=resolved.source_ids,
            ),
            "jobs_completed_light_1m": jobs_done_light_1m,
            "jobs_completed_heavy_1m": jobs_done_heavy_1m,
            "light_job_speed_pm": float(jobs_done_light_1m),
            "heavy_job_speed_pm": float(jobs_done_heavy_1m),
        }
    )
    total = int(v3.get("total") or 0)

    sources = []
    try:
        from core.sources import SourceManager

        sources = SourceManager(settings, run_maintenance=False).list_sources(
            active_only=True
        )
    except Exception:
        sources = []

    failed_report = {}
    try:
        failed_report = failed_reason_report(db) or {}
    except Exception:
        pass

    orphans = []
    try:
        orphans = orphan_source_report(db)
    except Exception:
        orphans = []

    status = {
        **v3,
        "indexed": int(counts.get("indexed", 0) or 0),
        "processing": int(v3.get("processing") or 0),
        "failed": int(counts.get("failed", 0) or 0),
        "pending": int(counts.get("pending", 0) or 0),
        "quarantine_by_reason": quarantine_reasons,
        "failed_reason_report": failed_report,
        "source_count": int(v3.get("source_count") or len(sources)),
        "sources": sources,
        "status_scope_source_ids": list(resolved.source_ids),
        "orphan_sources_report": orphans,
        "ocr_done": int(v3.get("ocr_done") or 0),
        "embedding_ready": int(v3.get("db_dino_embeddings") or 0),
        "lanes_only": False,
        "from_ram_cache": False,
        "engine": "index_v3",
        "archive_total": total,
    }
    try:
        status.update(_remaining_reason_summary(settings, db, resolved.source_ids))
    except Exception:
        pass
    for key in _BOUNDED_PIPELINE_KEYS:
        if key in status:
            value = max(0, int(status.get(key, 0) or 0))
            if total >= 0 and value > total:
                value = total
            status[key] = value
    return status
