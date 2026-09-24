"""Index Engine V3 — discovery (scan → insert / missing) without legacy Indexer."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from core.index_ssot import archive_content_changed
from core.index_v3.artifact_state import assess_file, assess_rows, plan_missing
from core.index_v3.physical_reconcile import reconcile_stale_physical_flags
from core.index_v3.planner import plan_jobs_for_file, reconcile_stale_pendings
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, Mode, QueueKind
from core.settings import SUPPORTED_EXTENSIONS
from core.utils import fs_access_path, normalize_path, normalize_source_root


def _root_candidates(root: str) -> list[str]:
    raw = str(root or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for candidate in (
        raw,
        raw.replace("/", "\\"),
        normalize_source_root(raw),
    ):
        text = str(candidate or "").strip()
        if text and text not in out:
            out.append(text)
        try:
            access = fs_access_path(text)
        except Exception:
            access = ""
        if access and access not in out:
            out.append(access)
    return out


def source_root_is_dir(root: str) -> bool:
    """UNC/NAS dahil mevcut scanner erişim yolunu kullan (Path.is_dir yetmez)."""
    for candidate in _root_candidates(root):
        try:
            if os.path.isdir(candidate):
                return True
        except OSError:
            continue
        try:
            if Path(candidate).is_dir():
                return True
        except OSError:
            continue
    return False


def _walk_root(root: str) -> str:
    for candidate in _root_candidates(root):
        try:
            if os.path.isdir(candidate):
                return candidate
        except OSError:
            continue
    return str(root or "").strip()


@dataclass
class DiscoveryStats:
    scanned: int = 0
    inserted: int = 0
    changed: int = 0
    unchanged: int = 0
    missing_marked: int = 0
    purged_missing: int = 0
    jobs_enqueued: int = 0
    errors: list[str] = field(default_factory=list)


def should_walk_on_index_start(
    *,
    explicit_source_id: int = 0,
    existing_file_count: int = 0,
    root_path: str = "",
    post_ga: bool = False,
) -> bool:
    """Kaynak Ekle / seçili tara: disk walk zorunlu. POST_GA walk yok."""
    if post_ga:
        return False
    if not str(root_path or "").strip():
        return False
    if int(explicit_source_id or 0) > 0:
        return True
    return int(existing_file_count or 0) == 0


def _iter_files(root: str) -> Iterable[tuple[str, int, float]]:
    if not source_root_is_dir(root):
        return
    root = _walk_root(root)
    skip_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "$RECYCLE.BIN",
        "System Volume Information",
        ".v3_cache",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            # macOS/metadata sidecar files (AppleDouble) are not real images.
            # They must never enter the index queue.
            if name.startswith("._") or name in (".DS_Store", "Thumbs.db"):
                continue
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            full = normalize_path(str(Path(dirpath) / name))
            try:
                st = os.stat(full)
                yield full, int(st.st_size), float(st.st_mtime)
            except OSError:
                continue


def count_indexable_files(
    root_path: str,
    *,
    on_progress: Callable[[int], None] | None = None,
    every: int = 25,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Yalnız os.walk sayımı — DB/kuyruk/indexer yok."""
    n = 0
    last_emit = 0
    step = max(1, int(every or 25))
    for _path, _size, _mtime in _iter_files(root_path) or []:
        if should_stop is not None and should_stop():
            break
        n += 1
        if on_progress is not None and (n == 1 or (n - last_emit) >= step):
            last_emit = n
            on_progress(n)
    if on_progress is not None:
        on_progress(n)
    return n


def discover_source(
    db: Any,
    store: JobStore,
    *,
    source_id: int,
    root_path: str,
    mode: Mode,
    mark_missing: bool = True,
    ocr_enabled: bool = False,
    patch_enabled: bool = True,
    settings: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    stream_chunk: int = 25,
) -> DiscoveryStats:
    """Streaming discovery: bul → DB → kuyruk; tam walk bitmeden sayaç artar."""
    stats = DiscoveryStats()
    known: list[str] = []
    if not source_root_is_dir(root_path):
        stats.errors.append("source_unavailable")
        return stats

    repair = mode == Mode.REPAIR
    reconcile_stale_physical_flags(
        db,
        [int(source_id)],
        cache_dir=str(getattr(settings, "cache_dir", "") or "") or None,
    )

    existing_rows = db.get_source_discovery_rows(int(source_id))
    existing = {normalize_path(str(r["path"])): r for r in existing_rows}
    chunk_n = max(1, min(int(stream_chunk or 25), 50))

    def _emit() -> None:
        if progress_callback is None:
            return
        queued = 0
        indexed = 0
        try:
            queued = int(store.count_pending(source_ids=[int(source_id)]) or 0)
        except Exception:
            queued = 0
        try:
            with db.connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                      SUM(CASE WHEN light_status='done' OR status='indexed'
                               THEN 1 ELSE 0 END) AS idx
                    FROM files
                    WHERE source_id=?
                      AND status NOT IN ('missing','excluded_internal')
                    """,
                    (int(source_id),),
                ).fetchone()
                indexed = int((row["idx"] if row else 0) or 0)
        except Exception:
            indexed = max(0, int(stats.inserted) - queued)
        try:
            progress_callback(
                {
                    "source_id": int(source_id),
                    "found": int(stats.scanned),
                    "inserted": int(stats.inserted),
                    "jobs_enqueued": int(stats.jobs_enqueued),
                    "queued": queued,
                    "indexed": indexed,
                }
            )
        except Exception:
            pass

    def _touch_source_count() -> None:
        try:
            with db.connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM files WHERE source_id=? AND status!='missing'",
                    (int(source_id),),
                ).fetchone()
                count = int(row["c"] or 0) if row else 0
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    UPDATE sources SET file_count=?, cache_status='scanning',
                    last_scan_at=?, updated_at=? WHERE id=?
                    """,
                    (count, now, now, int(source_id)),
                )
                conn.commit()
        except Exception:
            pass

    def _flush(batch_items: list[tuple[str, int, float]]) -> None:
        if not batch_items:
            return
        new_records: list[dict[str, Any]] = []
        changed: list[tuple[dict[str, Any], str, int, float]] = []
        for path, size, mtime in batch_items:
            row = existing.get(path)
            if row is None:
                new_records.append(
                    {
                        "path": path,
                        "filename": Path(path).name,
                        "source_id": int(source_id),
                        "file_size": size,
                        "mtime": mtime,
                    }
                )
            elif archive_content_changed(row, size, mtime):
                changed.append((row, path, size, mtime))
            else:
                stats.unchanged += 1
        inserted_ids = db.bulk_insert_source_files(new_records)
        stats.inserted += len(inserted_ids)
        for path, fid in inserted_ids.items():
            existing[path] = {"id": fid, "path": path, "file_size": 0, "mtime": 0}
        plan_ids = [int(fid) for fid in inserted_ids.values()]
        for row, path, size, mtime in changed:
            fid = int(row["id"])
            db.mark_file_pipeline_reset_on_change(fid, content_changed=True)
            db.upsert_file(
                {
                    "path": path,
                    "filename": Path(path).name,
                    "source_id": int(source_id),
                    "file_size": size,
                    "mtime": mtime,
                    "status": "pending",
                }
            )
            stats.changed += 1
            plan_ids.append(fid)
        if plan_ids:
            rows = db.get_artifact_rows_for_ids(plan_ids)
            chunk_jobs: list = []
            size_by_id: dict[int, int] = {}
            for file_row, feat_row in rows:
                report = assess_rows(file_row, feat_row, require_disk=False)
                report.source_id = int(source_id)
                report.path = str(file_row.get("path") or "")
                fid = int(file_row.get("id") or 0)
                if fid:
                    size_by_id[fid] = int(file_row.get("file_size") or 0)
                chunk_jobs.extend(
                    plan_jobs_for_file(
                        report,
                        mode,
                        repair=repair,
                        ocr_enabled=bool(ocr_enabled),
                        patch_enabled=bool(patch_enabled),
                    )
                )
            from dataclasses import replace

            from core.ovd_index import apply_owl_queue_policy

            chunk_jobs = [
                replace(j, file_size=int(size_by_id.get(int(j.file_id), 0) or 0))
                for j in chunk_jobs
            ]
            chunk_jobs = apply_owl_queue_policy(
                chunk_jobs, db=db, job_store=store, settings=settings
            )
            stats.jobs_enqueued += int(
                store.enqueue(
                    chunk_jobs,
                    reopen_permanent=repair,
                    reopen_done=repair,
                    settings=settings,
                )
                or 0
            )
        _touch_source_count()
        _emit()

    batch: list[tuple[str, int, float]] = []
    flushed_once = False
    for path, size, mtime in _iter_files(root_path) or []:
        stats.scanned += 1
        path = normalize_path(path)
        known.append(path)
        batch.append((path, int(size), float(mtime)))
        limit = 1 if not flushed_once else chunk_n
        if len(batch) >= limit:
            _flush(batch)
            batch = []
            flushed_once = True
    if batch:
        _flush(batch)

    if mark_missing:
        try:
            stats.missing_marked = int(
                db.mark_missing_files_for_source(int(source_id), known) or 0
            )
            if stats.missing_marked:
                with db.connect() as conn:
                    mids = [
                        int(r[0])
                        for r in conn.execute(
                            """
                            SELECT id FROM files
                            WHERE source_id=? AND status='missing'
                            """,
                            (int(source_id),),
                        ).fetchall()
                    ]
                if mids:
                    store.cancel_jobs_for_file_ids(mids)
                    try:
                        from core.index_v3.faiss_sync import exclude_file_ids
                        exclude_file_ids(settings, mids)
                    except Exception as exc:
                        stats.errors.append(
                            f"faiss_exclude:{type(exc).__name__}:{exc}"
                        )
                    # Missing kaydı kalır; satır silinmez (lifecycle / SSOT).
                    # Explicit purge yalnız BackgroundIndexScan(purge_missing=True).
        except Exception as exc:
            stats.errors.append(f"missing_reconcile:{type(exc).__name__}:{exc}")

    # Keep source counters aligned with discovered live files.
    try:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM files WHERE source_id=? AND status!='missing'",
                (int(source_id),),
            ).fetchone()
            count = int(row["c"] or 0) if row else 0
            conn.execute(
                "UPDATE sources SET file_count=?, last_scan_at=?, updated_at=? WHERE id=?",
                (
                    count,
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    int(source_id),
                ),
            )
            conn.commit()
    except Exception:
        pass

    _emit()
    return stats


def enqueue_existing_gaps(
    db: Any,
    store: JobStore,
    *,
    source_id: int,
    mode: Mode,
    limit: int | None = None,
    after_id: int = 0,
    ocr_enabled: bool = False,
    patch_enabled: bool = True,
    settings: Any | None = None,
) -> int | tuple[int, int]:
    """DB-only gap plan. limit/after_id ile chunk'lı arka plan tarama.

    ``limit`` hem yeni kuyruğa yazılan iş sayısını hem de bu çağrıda
    incelenen dosya sayısını sınırlar (en fazla ~2×limit dosya). Böylece
    pending zaten doluyken (added=0) tüm tablo taranmaz.

    Returns:
      limit is None → enqueued (int)  [geri uyum]
      limit set     → (enqueued, last_id)  last_id=0 kaynak bitti
    """
    repair = mode == Mode.REPAIR
    n = 0
    examined = 0
    last_id = int(after_id or 0)
    target = int(limit) if limit is not None and int(limit) > 0 else None
    # Pending dolu archive'da added=0 iken sonsuz tarama olmasın; yine de
    # tamamlanmış önek dosyaları atlamak için target'ın 2 katına izin ver.
    max_examine = None if target is None else max(int(target) * 2, int(target))
    fetch = 300 if target is None else max(50, min(target, 300))
    while True:
        sql_with_size = """
            SELECT id, COALESCE(file_size,0) AS file_size FROM files
            WHERE source_id=? AND status NOT IN ('excluded_internal','missing')
              AND id>?
            ORDER BY id
            LIMIT ?
        """
        sql_id_only = """
            SELECT id FROM files
            WHERE source_id=? AND status NOT IN ('excluded_internal','missing')
              AND id>?
            ORDER BY id
            LIMIT ?
        """
        with db.connect() as conn:
            try:
                rows = conn.execute(
                    sql_with_size, (int(source_id), int(last_id), int(fetch))
                ).fetchall()
            except Exception:
                rows = conn.execute(
                    sql_id_only, (int(source_id), int(last_id), int(fetch))
                ).fetchall()
        if not rows:
            if limit is not None:
                return (n, 0) if n == 0 else (n, last_id)
            return n
        for row in rows:
            fid = int(row["id"])
            last_id = fid
            examined += 1
            report = assess_file(db, fid, require_disk=False)
            report.source_id = int(source_id)
            # Mode A (Eksikleri Tamamla / REPAIR): content-heal bad previews;
            # VALID → skip (no mass re-render).
            if repair:
                try:
                    from core.preview_self_heal.hooks import invalidate_if_bad_preview

                    report = invalidate_if_bad_preview(
                        db, report, allow_source_compare=True
                    )
                except Exception:
                    pass
            jobs = plan_jobs_for_file(
                report,
                mode,
                repair=repair,
                ocr_enabled=bool(ocr_enabled),
                patch_enabled=bool(patch_enabled),
            )
            from dataclasses import replace

            from core.ovd_index import apply_owl_queue_policy

            try:
                sz = int(row["file_size"] or 0)
            except Exception:
                sz = 0
            jobs = [replace(j, file_size=sz) for j in jobs]
            jobs = apply_owl_queue_policy(jobs, db=db, job_store=store)
            reconcile_stale_pendings(store, report)
            # reopen_done only for Preview/Thumbnail physical gaps — never heavy AI.
            light_jobs = [
                j
                for j in jobs
                if j.artifact in (Artifact.PREVIEW, Artifact.THUMBNAIL)
            ]
            heavy_jobs = [
                j
                for j in jobs
                if j.artifact not in (Artifact.PREVIEW, Artifact.THUMBNAIL)
            ]
            reopen_light = mode != Mode.FAST
            added = 0
            if light_jobs:
                added += int(
                    store.enqueue(
                        light_jobs,
                        reopen_permanent=repair,
                        reopen_done=reopen_light,
                        settings=settings,
                    )
                    or 0
                )
            if heavy_jobs:
                added += int(
                    store.enqueue(
                        heavy_jobs,
                        reopen_permanent=repair,
                        reopen_done=False,
                    )
                    or 0
                )
            n += int(added)
            if target is not None and n >= target:
                return n, last_id
            if max_examine is not None and examined >= max_examine:
                return n, last_id
        # limit yoksa sonraki dosya chunk'ına devam (ilk 300'de kesme)


def enqueue_missing_thumbnail_jobs(
    db: Any,
    store: JobStore,
    *,
    source_ids: list[int] | None = None,
    limit: int = 50,
) -> dict[str, int]:
    """Thumb-only backfill: enqueue Artifact.THUMBNAIL without FeaturePreview.

    Selects files with physical_thumbnail_ready!=1. Does not plan PREVIEW
    (avoids forcing 1024px FeaturePreview across the archive). Idempotent via
    JobStore.enqueue. Newer file ids first (recent index priority).
    """
    lim = max(1, int(limit or 50))
    sids = [int(s) for s in (source_ids or []) if int(s or 0) > 0]
    params: list[Any] = []
    sql = """
        SELECT id, source_id, path FROM files
        WHERE status NOT IN ('excluded_internal','missing')
          AND ifnull(physical_thumbnail_ready, 0) != 1
    """
    if sids:
        placeholders = ",".join("?" * len(sids))
        sql += f" AND source_id IN ({placeholders})"
        params.extend(sids)
    # Prefer cheap raster first (avoid NAS TIF/vector timeout storms).
    sql += """
        ORDER BY
          CASE
            WHEN lower(path) LIKE '%.jpg' OR lower(path) LIKE '%.jpeg'
              OR lower(path) LIKE '%.png' OR lower(path) LIKE '%.bmp'
              OR lower(path) LIKE '%.webp' THEN 0
            ELSE 1
          END,
          id DESC
        LIMIT ?
    """
    params.append(lim)
    with db.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
        size_by_id: dict[int, int] = {}
        try:
            ids = [int(r["id"]) for r in rows]
            if ids:
                ph = ",".join("?" * len(ids))
                for r in conn.execute(
                    f"SELECT id, COALESCE(file_size,0) AS file_size FROM files WHERE id IN ({ph})",
                    ids,
                ).fetchall():
                    size_by_id[int(r["id"])] = int(r["file_size"] or 0)
        except Exception:
            size_by_id = {}
    jobs: list[Job] = []
    for row in rows:
        fid = int(row["id"])
        jobs.append(
            Job(
                fid,
                Artifact.THUMBNAIL,
                QueueKind.LIGHT,
                int(row["source_id"] or 0),
                str(row["path"] or ""),
                file_size=int(size_by_id.get(fid, 0) or 0),
            )
        )
    if not jobs:
        return {"queued": 0, "examined": 0}
    added = int(
        store.enqueue(
            jobs,
            reopen_permanent=False,
            reopen_done=True,
            settings=None,
        )
        or 0
    )
    return {"queued": added, "examined": len(jobs)}
