"""Low-priority background archive audit.

Never blocks search/index/startup. Small batches, long sleeps while search
or frozen-index search sessions are active. Suggest-only writes to sidecar DB.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from core.archive_intelligence.consistency import score_file_consistency
from core.archive_intelligence.store import (
    ArchiveIntelligenceStore,
    default_store_path,
)
from core.manual_label_guard import parse_texture_map

logger = logging.getLogger(__name__)

_BATCH = 20
_IDLE_SLEEP = 4.0
_BUSY_SLEEP = 12.0
_STARTUP_DELAY = 8.0


def _search_busy() -> bool:
    try:
        from core.background_index import is_search_active

        if is_search_active():
            return True
    except Exception:
        pass
    try:
        from core.index_freeze import INDEX_FROZEN, process_search_active

        if INDEX_FROZEN and process_search_active():
            return True
    except Exception:
        pass
    return False


class ArchiveIntelligenceScanner:
    def __init__(
        self,
        settings: Any,
        *,
        batch_size: int = _BATCH,
        idle_sleep: float = _IDLE_SLEEP,
        busy_sleep: float = _BUSY_SLEEP,
        startup_delay: float = _STARTUP_DELAY,
    ) -> None:
        self.settings = settings
        self.batch_size = max(5, int(batch_size))
        self.idle_sleep = max(1.0, float(idle_sleep))
        self.busy_sleep = max(self.idle_sleep, float(busy_sleep))
        self.startup_delay = max(0.0, float(startup_delay))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_stats: dict[str, Any] = {}

    def start(self) -> bool:
        if not bool(getattr(self.settings, "archive_intelligence_enabled", True)):
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="VezirArchiveIntelligence",
            daemon=True,
        )
        self._thread.start()
        logger.info("Archive Intelligence arka plan taraması başlatıldı")
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=max(0.1, float(timeout)))
        self._thread = None

    def _store(self) -> ArchiveIntelligenceStore:
        db_path = str(getattr(self.settings, "db_path", "") or "")
        custom = str(getattr(self.settings, "archive_intelligence_db_path", "") or "")
        path = custom or default_store_path(db_path)
        return ArchiveIntelligenceStore(path)

    def _run(self) -> None:
        if self.startup_delay > 0:
            if self._stop.wait(self.startup_delay):
                return
        store = self._store()
        db_path = str(getattr(self.settings, "db_path", "") or "")
        while not self._stop.is_set():
            if _search_busy():
                self.last_stats = {"paused": "search_active"}
                self._stop.wait(self.busy_sleep)
                continue
            try:
                scanned = self._scan_batch(store, db_path)
                self.last_stats = {"scanned": scanned, "cursor": store.get_cursor()}
            except Exception:
                logger.exception("Archive Intelligence batch failed")
                self.last_stats = {"error": True}
            self._stop.wait(self.idle_sleep)

    def _scan_batch(self, store: ArchiveIntelligenceStore, db_path: str) -> int:
        if _search_busy():
            return 0
        from core.db import Database

        db = Database(db_path)
        cursor = store.get_cursor()
        rows: list[dict[str, Any]] = []
        with db.connect() as conn:
            fetched = conn.execute(
                """
                SELECT f.id, f.filename, f.path, f.pattern_family, f.pattern_confidence,
                       f.category_path, f.mtime, f.indexed_at, f.created_at,
                       f.feature_preview_mtime, f.needs_review, fe.texture_map
                FROM files f
                LEFT JOIN features fe ON fe.file_id = f.id
                WHERE f.id > ?
                ORDER BY f.id ASC
                LIMIT ?
                """,
                (int(cursor), int(self.batch_size)),
            ).fetchall()
            rows = [dict(r) for r in fetched]
        if not rows:
            store.set_cursor(0)
            return 0
        if _search_busy():
            return 0
        last_id = cursor
        for rec in rows:
            if self._stop.is_set() or _search_busy():
                break
            fid = int(rec.get("id") or 0)
            last_id = max(last_id, fid)
            tm = parse_texture_map(rec.get("texture_map"))
            report = score_file_consistency(rec, tm)
            if report.kind == "ok":
                store.clear_ok(fid)
            else:
                fields = report.to_store_fields()
                store.upsert_candidate(file_id=fid, status="pending", **fields)
        store.set_cursor(last_id)
        return len(rows)
