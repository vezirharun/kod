"""INDEX vs SEARCH / SQLite write boundary (A / B / C).

A — INDEX_FROZEN (permanent flag): index *algorithms*, learned rules, and
    ranking recipes stay intact. It does NOT mean "stop all indexing forever".
    New sources/files continue to index when search write-protection is off.

B — SEARCH PROTECTION: while any thread holds search_session (process-wide),
    block mutation of ranking/index artifacts (features/FAISS/DNA/semantic/
    OCR text/ranking rows). Indexing resumes when no search_session is active.

C — SQLITE WRITE: normal maintenance/status/artifact housekeeping must not
    crash with INDEX_FROZEN_WRITE_BLOCKED. Prefer skip/swallow maintenance
    during search over treating INDEX_FROZEN alone as a global write ban.

Display-only caches (list thumbnail / feature preview image files) are
allowed during search so cache-miss results can still show images.

Production data is never deleted or rebuilt by this module.
"""
from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.logger import setup_logger

logger = setup_logger(__name__)

# A: permanent — algorithms/learned index logic frozen (not "pause workers forever").
INDEX_FROZEN = True

FROZEN_WRITE_CODE = "INDEX_FROZEN_WRITE_BLOCKED"

# UI list/detail görselleri — arama sırasında da üretilebilir (SQLite/FAISS değil).
DISPLAY_CACHE_OPS = frozenset(
    {
        "thumbnail.create",
        "preview.create",
    }
)

# C: housekeeping ops that may be allow-listed by callers; ranking artifacts are not here.
MAINTENANCE_OPS = frozenset(
    {
        "sqlite.maintenance.exclude_internal",
        "sqlite.maintenance.source_stats",
        "sqlite.maintenance.quarantine",
    }
)

_tls = threading.local()
_search_lock = threading.Lock()
_search_active = 0


def write_caller_frame() -> str:
    """First stack frame outside freeze/sqlite wrappers (file:line function)."""
    try:
        for fr in inspect.stack()[1:24]:
            fn = str(fr.filename or "").replace("\\", "/")
            parts = fn.rsplit("/", 2)
            parent = parts[-2] if len(parts) > 1 else ""
            base = parts[-1] if parts else ""
            if parent == "core" and base in {"index_freeze.py", "db.py"}:
                continue
            return f"{fr.filename}:{fr.lineno} {fr.function}"
    except Exception:
        return ""
    return ""


class IndexFrozenWriteBlocked(RuntimeError):
    """Accidental index write during search. Do not swallow."""

    def __init__(
        self,
        op: str,
        module: str = "",
        *,
        sql: str = "",
        caller: str = "",
    ) -> None:
        self.op = op
        self.module = module
        self.sql = sql
        self.caller = caller
        extra = ""
        if caller:
            extra += f" caller={caller}"
        if sql:
            extra += f" sql={sql[:180]!r}"
        super().__init__(f"{FROZEN_WRITE_CODE}: op={op} module={module}{extra}")


def in_search_session() -> bool:
    """This thread is inside search_session (nested wrappers)."""
    return int(getattr(_tls, "search", 0) or 0) > 0


def process_search_active() -> bool:
    """Any thread in this process currently holds search_session."""
    return int(_search_active) > 0


def search_write_protection_active() -> bool:
    """B: True while any search_session is held (this thread or another)."""
    return process_search_active() or in_search_session()


def in_index_write_session() -> bool:
    return int(getattr(_tls, "write", 0) or 0) > 0


@contextmanager
def search_session() -> Iterator[None]:
    """B: enter search write-protection (process-wide)."""
    global _search_active
    _tls.search = int(getattr(_tls, "search", 0) or 0) + 1
    with _search_lock:
        _search_active += 1
    try:
        yield
    finally:
        with _search_lock:
            _search_active = max(0, int(_search_active) - 1)
        _tls.search = max(0, int(getattr(_tls, "search", 0) or 0) - 1)


@contextmanager
def allow_index_writes() -> Iterator[None]:
    """IndexEngine / Indexer only. Does not bypass an active search_session
    (this thread or another thread in the process)."""
    _tls.write = int(getattr(_tls, "write", 0) or 0) + 1
    try:
        yield
    finally:
        _tls.write = max(0, int(getattr(_tls, "write", 0) or 0) - 1)


def guard_index_write(op: str, module: str = "", *, sql: str = "") -> None:
    """B: raise only while search write-protection is active (not for A alone).

    INDEX_FROZEN gates whether protection exists at all. When True, writes are
    blocked during search_session; outside search, indexing/maintenance proceed.
    """
    if not INDEX_FROZEN:
        return
    op_key = str(op or "")
    # Liste/önizleme görsel cache — arama sırasında da yazılabilir.
    if op_key in DISPLAY_CACHE_OPS:
        return
    # C: explicit maintenance ops — do not crash workers mid-search.
    if op_key in MAINTENANCE_OPS:
        return
    # Process-wide search wins over allow_index_writes on any thread.
    if search_write_protection_active():
        pass
    elif in_index_write_session():
        return
    else:
        # Outside search: A does not block normal index/maintenance writes.
        return
    if not module:
        try:
            module = inspect.stack()[1].filename
        except Exception:
            module = ""
    caller = write_caller_frame()
    logger.error(
        "%s op=%s module=%s caller=%s sql=%s",
        FROZEN_WRITE_CODE,
        op,
        module,
        caller,
        (sql or "")[:180],
    )
    raise IndexFrozenWriteBlocked(op, module, sql=sql, caller=caller)


def snapshot_index_artifacts(
    *,
    db_path: str | Path,
    faiss_dino_path: str | Path = "",
    faiss_clip_path: str | Path = "",
    cache_dir: str | Path = "",
) -> dict[str, dict[str, Any]]:
    """Size/mtime/exists for freeze tests. Read-only; does not open SQLite."""
    paths: list[Path] = []
    for raw in (db_path, faiss_dino_path, faiss_clip_path):
        if raw:
            p = Path(raw)
            paths.append(p)
            paths.append(Path(str(p) + "-wal"))
            paths.append(Path(str(p) + "-shm"))
            paths.append(p.with_suffix(".map.npy"))
            paths.append(p.with_name(f"{p.stem}_map.npy"))
    cache = Path(cache_dir) if cache_dir else None
    if cache and cache.is_dir():
        for sub in ("thumbnails", "feature_previews"):
            d = cache / sub
            if d.is_dir():
                paths.extend(sorted(d.glob("*"))[:5000])

    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        key = str(path)
        if not path.exists() or not path.is_file():
            out[key] = {"exists": False, "size": 0, "mtime_ns": 0}
            continue
        st = path.stat()
        out[key] = {
            "exists": True,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    out["_file_count"] = {
        "exists": True,
        "size": len(
            [
                v
                for k, v in out.items()
                if v.get("exists") and not str(k).endswith(("-wal", "-shm"))
            ]
        ),
        "mtime_ns": 0,
    }
    return out


def freeze_fingerprint(snap: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare index artifacts; ignore SQLite WAL/SHM reader sidecars."""
    return {
        k: v
        for k, v in snap.items()
        if k != "_file_count" and not str(k).endswith(("-wal", "-shm"))
    }
