"""Immediate index purge when a watched image disappears from disk.

Folder watch used to debounce into a full scan. This path removes one deleted
file from the index SSOT without waiting for that scan: files/features/FTS
and related joins, JobStore pending/claimed rows, FAISS exclude plus the
rebuild flag, and thumbnail/feature_preview cache files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from core.logger import setup_logger
from core.utils import fs_access_path, normalize_path

logger = setup_logger(__name__)


def job_store_path(settings: Any) -> Path:
    base = Path(str(getattr(settings, "db_path", "") or ""))
    return base.with_name(base.stem + ".v3jobs.db")


def purge_disk_deletes(
    db: Any,
    settings: Any,
    paths: Iterable[str],
    *,
    watched_roots: Iterable[str] | None = None,
    tree_paths: Iterable[str] | None = None,
    job_store: Any | None = None,
) -> dict[str, Any]:
    """Purge indexed files that the given filesystem notifications say are gone.

    ``paths`` may be a deleted file, a deleted directory, or a directory that
    still exists (only direct children missing on disk are removed).
    ``tree_paths`` is the overflow case: every indexed file under that live
    directory that is no longer on disk.
    ``watched_roots`` limits the purge to those trees. ``None`` does not add
    that filter (tests and direct callers).
    """
    roots = [normalize_path(p) for p in (watched_roots or []) if str(p or "").strip()]
    root_filter = list(watched_roots) if watched_roots is not None else None
    if root_filter is not None and not roots:
        return {"removed": 0, "jobs_cancelled": 0, "file_ids": []}

    ids = collect_deleted_file_ids(
        db,
        paths,
        watched_roots=root_filter,
        tree_paths=tree_paths,
    )
    if not ids:
        return {"removed": 0, "jobs_cancelled": 0, "file_ids": []}

    jobs_cancelled = _cancel_jobs(settings, ids, job_store)
    cache_dir = str(getattr(settings, "cache_dir", "") or "").strip() or None
    removed = int(db.purge_file_ids(ids, cache_dir=cache_dir) or 0)
    if removed:
        try:
            db.set_meta("faiss_needs_rebuild", "1")
        except Exception:
            logger.warning("faiss_needs_rebuild flag was not saved", exc_info=True)
    _exclude_faiss(settings, ids)
    if removed:
        logger.info("Purged %s index file(s) deleted on disk", removed)
    return {
        "removed": removed,
        "jobs_cancelled": jobs_cancelled,
        "file_ids": ids,
    }


def collect_deleted_file_ids(
    db: Any,
    paths: Iterable[str],
    *,
    watched_roots: Iterable[str] | None = None,
    tree_paths: Iterable[str] | None = None,
) -> list[int]:
    roots = None
    if watched_roots is not None:
        roots = [normalize_path(p) for p in watched_roots if str(p or "").strip()]
        if not roots:
            return []
    found: list[int] = []
    seen: set[int] = set()

    def add(fid: int) -> None:
        value = int(fid)
        if value > 0 and value not in seen:
            seen.add(value)
            found.append(value)

    for raw in list(paths or []):
        for fid in _ids_for_notified_path(db, str(raw or ""), roots):
            add(fid)
    for raw in list(tree_paths or []):
        for fid in _ids_missing_under_tree(db, str(raw or ""), roots):
            add(fid)
    return found


def _ids_for_notified_path(db: Any, raw: str, roots: list[str] | None) -> list[int]:
    norm = normalize_path(raw)
    if not norm or not _allowed(norm, roots) or _is_fs_root(norm):
        return []
    if _is_dir(norm):
        return [
            fid
            for fid, path in _direct_indexed_files(db, norm)
            if not _is_file(path)
        ]
    ids: list[int] = []
    if not _is_file(norm):
        row = _file_row(db, norm)
        if row is not None:
            ids.append(int(row["id"]))
        parent = os.path.dirname(norm.rstrip("\\/"))
        if parent and parent != norm and _is_dir(parent):
            for fid, path in _indexed_under_prefix(db, norm):
                if not _is_file(path):
                    ids.append(fid)
    return ids


def _ids_missing_under_tree(db: Any, raw: str, roots: list[str] | None) -> list[int]:
    norm = normalize_path(raw)
    if not norm or not _allowed(norm, roots) or _is_fs_root(norm) or not _is_dir(norm):
        return []
    return [fid for fid, path in _indexed_under_prefix(db, norm) if not _is_file(path)]


def _cancel_jobs(settings: Any, file_ids: list[int], job_store: Any | None) -> int:
    store = job_store
    if store is None:
        db_path = str(getattr(settings, "db_path", "") or "").strip()
        if not db_path:
            return 0
        try:
            from core.index_v3.queues import JobStore

            store = JobStore(job_store_path(settings))
        except Exception:
            logger.warning("JobStore cancel skipped", exc_info=True)
            return 0
    try:
        return int(store.cancel_jobs_for_file_ids(file_ids) or 0)
    except Exception:
        logger.warning("JobStore cancel failed", exc_info=True)
        return 0


def _exclude_faiss(settings: Any, file_ids: list[int]) -> None:
    try:
        from core.index_v3.faiss_sync import exclude_file_ids

        exclude_file_ids(settings, file_ids)
    except Exception:
        logger.warning("FAISS exclude failed; rebuild flag remains", exc_info=True)


def _file_row(db: Any, path: str) -> dict[str, Any] | None:
    row = db.get_file_by_path(path)
    if row:
        return row
    with db.connect() as conn:
        found = conn.execute(
            "SELECT id, path FROM files WHERE path = ? COLLATE NOCASE",
            (path,),
        ).fetchone()
    return dict(found) if found else None


def _direct_indexed_files(db: Any, directory: str) -> list[tuple[int, str]]:
    direct, nested = _like_patterns(directory)
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, path FROM files
            WHERE path LIKE ? ESCAPE '\\'
              AND path NOT LIKE ? ESCAPE '\\'
            """,
            (direct, nested),
        ).fetchall()
    return [(int(row["id"]), str(row["path"])) for row in rows]


def _indexed_under_prefix(db: Any, directory: str) -> list[tuple[int, str]]:
    direct, _nested = _like_patterns(directory)
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, path FROM files
            WHERE path LIKE ? ESCAPE '\\'
            """,
            (direct,),
        ).fetchall()
    return [(int(row["id"]), str(row["path"])) for row in rows]


def _like_patterns(directory: str) -> tuple[str, str]:
    """SQL LIKE patterns for ``directory + sep + …`` (direct vs any descendant)."""
    directory = normalize_path(directory).rstrip("\\/")
    sep = "\\" if os.name == "nt" else "/"
    escaped = directory.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    esc_sep = "\\\\" if sep == "\\" else sep
    direct = escaped + esc_sep + "%"
    nested = direct + esc_sep + "%"
    return direct, nested


def _allowed(path: str, roots: list[str] | None) -> bool:
    if roots is None:
        return True
    return any(_under(path, root) for root in roots)


def _under(path: str, root: str) -> bool:
    left = normalize_path(path).casefold().rstrip("\\/")
    right = normalize_path(root).casefold().rstrip("\\/")
    if not left or not right:
        return False
    if left == right:
        return True
    return left.startswith(right + os.sep)


def _is_file(path: str) -> bool:
    try:
        return os.path.isfile(fs_access_path(path))
    except OSError:
        return False


def _is_dir(path: str) -> bool:
    try:
        return os.path.isdir(fs_access_path(path))
    except OSError:
        return False


def _is_fs_root(path: str) -> bool:
    raw = normalize_path(path).rstrip("\\/")
    parent = os.path.dirname(raw)
    return not parent or parent == raw
