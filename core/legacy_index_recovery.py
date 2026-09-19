"""Legacy index recovery helpers.

The release package must never replace a user's existing data directory.  This
module also repairs an older index whose source registry was lost while file
rows still exist.  Recovery is conservative and idempotent.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from core.logger import setup_logger
from core.path_safety import internal_source_reason
from core.utils import normalize_source_root

logger = setup_logger(__name__)


def _db_file_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        con = sqlite3.connect(str(path), timeout=5)
        try:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
            ).fetchone()
            if not row:
                return 0
            return int(con.execute(
                "SELECT COUNT(*) FROM files WHERE status NOT IN ('excluded_internal','missing')"
            ).fetchone()[0] or 0)
        finally:
            con.close()
    except Exception:
        return 0


def _has_sources(path: Path) -> bool:
    try:
        con = sqlite3.connect(str(path), timeout=5)
        try:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sources'"
            ).fetchone()
            return bool(row and con.execute("SELECT 1 FROM sources LIMIT 1").fetchone())
        finally:
            con.close()
    except Exception:
        return False


def _candidate_databases(current: Path) -> list[Path]:
    """Find likely previous project databases without scanning the whole disk."""
    current = current.resolve()
    roots = [current.parent]
    # Typical update layout: new_project/ next to old_project/.
    for p in (current.parent.parent, current.parent.parent.parent):
        if p and p.exists():
            roots.append(p)
    seen: set[Path] = set()
    out: list[Path] = []
    skip = {".git", ".venv", "cache", "__pycache__", "node_modules"}
    for root in roots:
        try:
            iterator = root.rglob("patterns.db")
        except Exception:
            continue
        for p in iterator:
            try:
                rp = p.resolve()
                rel = rp.relative_to(root.resolve())
                if len(rel.parts) > 4 or rp == current or rp in seen:
                    continue
                if any(part.casefold() in skip for part in rel.parts):
                    continue
                if _db_file_count(rp) <= 0:
                    continue
                seen.add(rp)
                out.append(rp)
            except Exception:
                continue
    # Prefer databases with an intact source registry, then larger indexes.
    out.sort(key=lambda p: (_has_sources(p), _db_file_count(p), p.stat().st_mtime), reverse=True)
    return out


def recover_previous_database(settings: Any) -> dict[str, Any]:
    """If the configured DB is empty, point settings at a nearby real index.

    This is only a startup recovery path.  It never overwrites or deletes a DB.
    """
    current = Path(str(getattr(settings, "db_path", ""))).expanduser()
    if _db_file_count(current) > 0:
        return {"recovered": False, "reason": "current_db_has_files", "count": _db_file_count(current)}
    candidates = _candidate_databases(current)
    if not candidates:
        return {"recovered": False, "reason": "no_previous_db"}
    candidate = candidates[0]
    settings.db_path = str(candidate)
    data_dir = candidate.parent
    # Reuse the old vector/cache artifacts when present.  Missing files are fine;
    # the application will fall back to SQL/text search and can rebuild later.
    dino = data_dir / "faiss_dino.index"
    clip = data_dir / "faiss_clip.index"
    cache = data_dir.parent / "cache"
    if dino.exists():
        settings.faiss_dino_path = str(dino)
    if clip.exists():
        settings.faiss_clip_path = str(clip)
    if cache.exists():
        settings.cache_dir = str(cache)
    logger.warning("Önceki index bulundu ve korundu: %s (%d kayıt)", candidate, _db_file_count(candidate))
    return {"recovered": True, "reason": "previous_db", "path": str(candidate), "count": _db_file_count(candidate)}


def _common_root(paths: list[str]) -> str:
    cleaned = [str(Path(p).expanduser()) for p in paths if str(p).strip()]
    if not cleaned:
        return ""
    try:
        root = os.path.commonpath(cleaned)
    except ValueError:
        return ""
    # Keep the host-native path form here. normalize_source_root intentionally
    # canonicalizes Windows separators and is not appropriate for Linux-based
    # test runners.
    root = str(Path(root).resolve())
    # Never auto-register a drive/share root.  That could make a later scan walk
    # the entire machine or NAS.  A meaningful directory is required.
    p = Path(root)
    if len(p.parts) <= 1:
        return ""
    if len(p.parts) == 2 and p.anchor:
        return ""
    return root


def recover_orphan_source_registry(settings: Any) -> dict[str, Any]:
    """Recover legacy indexed files whose source registry is missing.

    Older databases may contain perfectly valid ``files`` rows with
    ``source_id=0`` (the source column did not exist yet), or with a positive
    source id whose registry row was lost during an update.  The V3 status
    engine intentionally excludes orphan source ids from the archive scope, so
    leaving these rows untouched makes a real old index look like "0 files".

    Recovery is conservative:
      * never deletes indexed rows;
      * never changes an existing source row;
      * only considers active, non-missing files;
      * derives a real directory from the indexed paths;
      * reuses an orphan positive id when it is free, otherwise allocates a new
        source id and rewrites only the affected ``source_id`` values.
    """
    from core.db import Database

    db = Database(settings.db_path)
    with db.connect() as con:
        registry_ids = {
            int(r[0])
            for r in con.execute("SELECT id FROM sources").fetchall()
        }
        rows = con.execute(
            """
            SELECT f.id, f.source_id, f.path
            FROM files f
            LEFT JOIN sources s ON s.id = f.source_id
            WHERE f.status NOT IN ('excluded_internal','missing')
              AND (
                    f.source_id = 0
                    OR s.id IS NULL
                  )
            ORDER BY f.source_id, f.id
            """
        ).fetchall()

    if not rows:
        return {
            "recovered": False,
            "reason": "no_orphan_files",
            "sources": 0,
            "files_reassigned": 0,
            "skipped": 0,
        }

    grouped: dict[int, list[tuple[int, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row[1] or 0), []).append(
            (int(row[0]), str(row[2] or ""))
        )

    recovered = 0
    skipped = 0
    files_reassigned = 0
    next_id = max(registry_ids or {0}) + 1

    for old_sid, file_rows in grouped.items():
        paths = [path for _, path in file_rows if path]
        root = _common_root(paths)
        if not root or internal_source_reason(root, settings):
            skipped += 1
            continue

        # A positive legacy id can be preserved if it is not occupied.  The
        # source_id=0 legacy bucket needs a real registry id for V3 scope.
        if old_sid > 0 and old_sid not in registry_ids:
            new_sid = old_sid
        else:
            while next_id in registry_ids:
                next_id += 1
            new_sid = next_id
            next_id += 1

        name = Path(root).name or root
        stype = "server_share" if root.startswith("\\\\") else "local_pc"
        with db.connect() as con:
            con.execute(
                """
                INSERT INTO sources(id, name, source_type, root_path, is_active,
                                    scan_interval_hours, deep_scan_interval_days)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    int(new_sid),
                    f"Kurtarılan: {name}",
                    stype,
                    root,
                    1,
                    int(getattr(settings, "quick_scan_interval_hours", 6) or 6),
                    int(getattr(settings, "deep_scan_interval_days", 7) or 7),
                ),
            )
            # For source_id=0 or an orphan id allocated to a new registry row,
            # bind the legacy records to the recovered source.  Existing
            # registry-owned rows are never touched.
            ids = [fid for fid, _ in file_rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                cur = con.execute(
                    f"UPDATE files SET source_id=?, updated_at=datetime('now') "
                    f"WHERE id IN ({placeholders})",
                    [int(new_sid), *ids],
                )
                files_reassigned += int(cur.rowcount or 0)

        registry_ids.add(int(new_sid))
        recovered += 1
        logger.warning(
            "Eski index kaynağı kurtarıldı: source_id=%s -> %s root=%s files=%d",
            old_sid,
            new_sid,
            root,
            len(file_rows),
        )

    return {
        "recovered": recovered > 0,
        "reason": "orphan_sources",
        "sources": recovered,
        "files_reassigned": files_reassigned,
        "skipped": skipped,
    }
