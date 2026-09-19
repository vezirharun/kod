"""Legacy index_queue köprüsü — ilerleme otoritesi DEĞİL.

Tek kuyruk = files.light_status / files.heavy_status (core.index_ssot).
Bu sınıf enqueue yazmaz; özetler SSOT'tan okunur. Tablo silinmez (geriye uyum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.db import Database

TASK_TYPES = (
    "thumbnail",
    "feature_preview",
    "hash",
    "patch",
    "texture_map",
    "taxonomy",
    "ai_embedding",
    "ocr",
)

QUEUE_PENDING = "pending"
QUEUE_RUNNING = "running"
QUEUE_DONE = "done"
QUEUE_ERROR = "error"
QUEUE_SKIPPED = "skipped"


@dataclass
class SourceQueueSummary:
    source_id: int
    source_name: str
    indexed: int = 0
    pending: int = 0
    processing_files: int = 0
    queue_pending: int = 0
    queue_error: int = 0
    total_seen: int = 0


class IndexQueueManager:
    def __init__(self, db: Database):
        self.db = db

    def reset_stale_running(self) -> int:
        """Eski index_queue running satırlarını temizle (SSOT değil)."""
        return self.db.reset_index_queue_running()

    def summaries_for_active_sources(self) -> list[SourceQueueSummary]:
        from core.index_ssot import count_light_pending_jobs

        out: list[SourceQueueSummary] = []
        for src in self.db.list_sources(active_only=True):
            sid = int(src["id"])
            stats = self.db.count_files_for_source(sid)
            # File-level heavy_status=pending (ör. DNA eksiği) bütün arşivi
            # complete kuyruğu gibi göstermesin. Açılış resume yalnız gerçek
            # light pending / processing dosyalarına bakılır.
            light_n = int(count_light_pending_jobs(self.db, sid) or 0)
            processing = int(stats.get("processing", 0) or 0)
            out.append(
                SourceQueueSummary(
                    source_id=sid,
                    source_name=src.get("name", ""),
                    indexed=stats.get("indexed", 0),
                    pending=light_n,
                    processing_files=processing,
                    queue_pending=light_n,
                    queue_error=0,
                    total_seen=sum(stats.values()),
                )
            )
        return out

    def sources_with_incomplete_work(self) -> list[SourceQueueSummary]:
        return [
            s
            for s in self.summaries_for_active_sources()
            if s.processing_files > 0 or s.queue_pending > 0 or s.pending > 0
        ]

    def enqueue_file_tasks(self, *args: Any, **kwargs: Any) -> None:
        """No-op — tek kuyruk SSOT."""
        return None

    def enqueue_heavy_after_preview(self, *args: Any, **kwargs: Any) -> None:
        """No-op — Genel AI = DB heavy_status pending."""
        return None

    def enqueue_fast_search_tasks(self, *args: Any, **kwargs: Any) -> None:
        """No-op — tek kuyruk SSOT."""
        return None

    def pending_file_ids(
        self, source_id: int | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        """SSOT light+heavy pending (legacy şekil)."""
        from core.index_ssot import list_fast_jobs, list_general_ai_jobs

        sid = int(source_id or 0)
        jobs = list_fast_jobs(self.db, sid, limit=limit) + list_general_ai_jobs(
            self.db, sid, limit=limit
        )
        return [
            {"file_id": fid, "file_path": path, "filename": name}
            for fid, path, name in jobs[:limit]
        ]

    def mark_task(self, queue_id: int, status: str, error: str = "") -> None:
        """DONE yazmadan önce artifact doğrula — false-done engeli."""
        if status == QUEUE_DONE:
            with self.db.connect() as conn:
                row = conn.execute(
                    "SELECT file_id, COALESCE(task_type, stage) AS task FROM index_queue WHERE id=?",
                    (int(queue_id),),
                ).fetchone()
            if row:
                from core.index_integrity import validate_before_queue_done

                ok, reason = validate_before_queue_done(
                    self.db, int(row["file_id"]), str(row["task"] or "")
                )
                if not ok:
                    self.db.update_index_queue_item(
                        queue_id,
                        status=QUEUE_ERROR,
                        error_message=reason or "artifact_missing",
                    )
                    return
        self.db.update_index_queue_item(queue_id, status=status, error_message=error)

    def complete_file_if_done(self, file_id: int) -> bool:
        """Tüm görevler bittiyse integrity'ye göre indexed veya incomplete."""
        if not self.db.file_queue_all_done(file_id):
            return False
        from core.index_integrity import finalize_file_status

        finalize_file_status(self.db, int(file_id))
        return True
