"""Çok kaynaklı merkezi index — kaynak türleri ve yönetim."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from core.db import Database
from core.logger import setup_logger
from core.path_safety import internal_source_reason
from core.settings import AppSettings
from core.utils import normalize_path, normalize_source_root

logger = setup_logger(__name__)


class SourceType(str, Enum):
    LOCAL_PC = "local_pc"
    SERVER_SHARE = "server_share"
    NAS_BACKUP = "nas_backup"
    SELECTED_FOLDER = "selected_folder"


class SearchScope(str, Enum):
    ALL = "all"
    LOCAL_PC = "local_pc"
    SERVER_SHARE = "server_share"
    NAS_BACKUP = "nas_backup"
    SELECTED_FOLDER = "selected_folder"
    SELECTED_SOURCES = "selected_sources"
    QUICK_FOLDER = "quick_folder"
    CUSTOMER = "customer"


class ScanMode(str, Enum):
    QUICK = "quick"  # mtime + size — hızlı güncelleme
    DEEP = "deep"  # partial hash + thumbnail doğrulama — haftalık


SOURCE_TYPE_LABELS = {
    SourceType.LOCAL_PC: "Bu PC",
    SourceType.SERVER_SHARE: "Sunucu Paylaşımı",
    SourceType.NAS_BACKUP: "NAS Yedek",
    SourceType.SELECTED_FOLDER: "Seçili Klasör",
}

SOURCE_TYPE_LABELS_TR = {e.value: SOURCE_TYPE_LABELS[e] for e in SourceType}

SEARCH_SCOPE_LABELS = {
    SearchScope.ALL: "Tüm kaynaklarda ara",
    SearchScope.SELECTED_SOURCES: "Seçili kaynaklarda ara",
    SearchScope.LOCAL_PC: "Sadece bu PC",
    SearchScope.SERVER_SHARE: "Sadece sunucu",
    SearchScope.NAS_BACKUP: "Sadece NAS",
    SearchScope.SELECTED_FOLDER: "Sadece seçili klasör",
    SearchScope.QUICK_FOLDER: "Bu klasörde hızlı ara",
    SearchScope.CUSTOMER: "Sadece seçili müşteri",
}


@dataclass
class SearchFilter:
    """Arama kapsam filtresi."""

    scope: str = SearchScope.ALL.value
    customer: str = ""
    source_id: int = 0
    source_ids: list[int] = field(default_factory=list)
    folder_path: str = ""

    def source_types(self) -> list[str] | None:
        if self.scope == SearchScope.ALL.value:
            return None
        if self.scope in (
            SearchScope.CUSTOMER.value,
            SearchScope.SELECTED_FOLDER.value,
            SearchScope.SELECTED_SOURCES.value,
            SearchScope.QUICK_FOLDER.value,
        ):
            return None
        if self.scope in (
            SearchScope.LOCAL_PC.value,
            SearchScope.SERVER_SHARE.value,
            SearchScope.NAS_BACKUP.value,
        ):
            return [self.scope]
        return None


def detect_source_type(path: str) -> str:
    """Yoldan kaynak türünü tahmin et."""
    norm = normalize_source_root(path)
    upper = norm.upper()
    if norm.startswith("\\\\"):
        nas_keywords = ("NAS", "BACKUP", "YEDEK", "ARCHIVE", "ARŞIV", "ARSIV")
        if any(k in upper for k in nas_keywords):
            return SourceType.NAS_BACKUP.value
        return SourceType.SERVER_SHARE.value
    return SourceType.LOCAL_PC.value


class SourceManager:
    """Kaynak CRUD ve tarama planlama."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        run_maintenance: bool = True,
        defer_db: bool = False,
    ):
        self.settings = settings
        from core.index_freeze import (
            IndexFrozenWriteBlocked,
            search_write_protection_active,
        )

        # B: skip maintenance during search (avoid mid-search ranking churn).
        if run_maintenance and search_write_protection_active():
            run_maintenance = False
        self._defer_db = bool(defer_db)
        self._db: Database | None = None
        self.quarantined_sources: list[dict[str, Any]] = []
        self.excluded_internal_files = 0
        self._maintenance_pending = bool(run_maintenance)
        if not self._defer_db:
            self._ensure_db()

    @property
    def db(self) -> Database:
        return self._ensure_db()

    @db.setter
    def db(self, value: Database | None) -> None:
        self._db = value

    def _ensure_db(self) -> Database:
        if self._db is None:
            self._db = Database(self.settings.db_path)
        if self._maintenance_pending:
            from core.index_freeze import IndexFrozenWriteBlocked

            self._maintenance_pending = False
            try:
                self.quarantined_sources = self._quarantine_internal_sources()
                self.excluded_internal_files = self._exclude_generated_artifacts()
                self._reconcile_source_counts()
            except IndexFrozenWriteBlocked:
                logger.info(
                    "SourceManager maintenance skipped (search write-protection)"
                )
                self.quarantined_sources = []
                self.excluded_internal_files = 0
        return self._db

    def list_sources(self, active_only: bool = False) -> list[dict[str, Any]]:
        return self.db.list_sources(active_only=active_only)

    def get_source(self, source_id: int) -> dict[str, Any] | None:
        return self.db.get_source(source_id)

    def add_source(
        self,
        name: str,
        root_path: str,
        source_type: str | None = None,
        is_active: bool = True,
        scan_interval_hours: int | None = None,
        deep_scan_interval_days: int | None = None,
    ) -> int:
        root = normalize_source_root(root_path)
        reason = internal_source_reason(root, self.settings)
        if reason:
            raise ValueError(reason)
        stype = source_type or detect_source_type(root)
        if not name:
            name = os.path.basename(root.rstrip("\\/")) or root
        return self.db.upsert_source(
            {
                "name": name,
                "source_type": stype,
                "root_path": root,
                "is_active": 1 if is_active else 0,
                "scan_interval_hours": scan_interval_hours
                or self.settings.quick_scan_interval_hours,
                "deep_scan_interval_days": deep_scan_interval_days
                or self.settings.deep_scan_interval_days,
            }
        )

    def _is_internal_storage(self, root_path: str) -> bool:
        return bool(internal_source_reason(root_path, self.settings))

    def _quarantine_internal_sources(self) -> list[dict[str, Any]]:
        quarantined: list[dict[str, Any]] = []
        for source in self.db.list_sources(active_only=True):
            reason = internal_source_reason(source.get("root_path", ""), self.settings)
            if not reason:
                continue
            blocked = {**source, "is_active": 0, "cache_status": "blocked_internal"}
            self.db.upsert_source(blocked)
            quarantined.append({**blocked, "blocked_reason": reason})
            logger.warning(
                "Teknik kaynak pasifleştirildi: %s (%s)",
                source.get("root_path"),
                reason,
            )
        return quarantined

    def _reconcile_source_counts(self) -> None:
        """Keep the source table consistent with files already owned by source_id."""
        for source in self.db.list_sources():
            self.update_source_stats(int(source["id"]))
            reason = internal_source_reason(source.get("root_path", ""), self.settings)
            if reason:
                refreshed = self.db.get_source(int(source["id"])) or source
                self.db.upsert_source(
                    {
                        **refreshed,
                        "is_active": 0,
                        "cache_status": "blocked_internal",
                    }
                )

    def _exclude_generated_artifacts(self) -> int:
        from core.index_freeze import IndexFrozenWriteBlocked

        project_root = os.path.dirname(os.path.abspath(self.settings.cache_dir))
        protected = [
            self.settings.cache_dir,
            os.path.dirname(os.path.abspath(self.settings.db_path)),
            os.path.join(project_root, ".venv"),
        ]
        active_roots = [
            normalize_source_root(src.get("root_path", "")).casefold()
            for src in self.db.list_sources(active_only=True)
        ]
        safe_to_exclude: list[str] = []
        for root in protected:
            key = normalize_source_root(root).casefold()
            explicitly_selected = any(
                source == key or source.startswith(key + "\\")
                for source in active_roots
            )
            if not explicitly_selected:
                safe_to_exclude.append(root)
        try:
            count = self.db.exclude_internal_artifacts(safe_to_exclude)
        except IndexFrozenWriteBlocked:
            # C: search started mid-maintenance — no crash, retry later.
            logger.info("exclude_internal skipped (search write-protection)")
            return 0
        if count:
            logger.warning("Teknik uygulama dosyaları arama dışına alındı: %d", count)
        return count

    def update_source(self, source_id: int, **fields: Any) -> None:
        src = self.db.get_source(source_id)
        if not src:
            return
        merged = {**src, **fields}
        self.db.upsert_source(merged)

    def toggle_active(self, source_id: int, active: bool) -> None:
        self.update_source(source_id, is_active=1 if active else 0)

    def remove_source(self, source_id: int) -> None:
        """Kaynak satırını sil — dosya kayıtlarına dokunmaz (yetim bırakabilir)."""
        self.db.delete_source(source_id)

    def detach_source(self, source_id: int) -> None:
        """Kaynağı arama listesinden kaldır — orijinal dosyalara dokunmaz."""
        self.toggle_active(source_id, False)

    def purge_orphan_indexes(
        self,
        cache_dir: str,
        progress_callback=None,
    ) -> dict[str, Any]:
        """Yetim (sources'ta olmayan source_id) file/index kayıtlarını temizle.

        Açık işlem — otomatik index/kuyruk YOK. Orijinal disk dosyalarına dokunmaz.
        """
        from core.index_v3.scope import resolve_index_scope

        def emit(**extra):
            if progress_callback:
                progress_callback(extra)

        scope = resolve_index_scope(self.db, [])
        orphan_ids = list(scope.orphan_source_ids)
        if not orphan_ids:
            emit(stage="done", message="Yetim kayıt yok", files_removed=0)
            return {"files_removed": 0, "orphan_source_ids": []}

        total_removed = 0
        for sid in orphan_ids:
            emit(
                stage="purge_orphan",
                message=f"Yetim source_id={sid} temizleniyor…",
                source_id=sid,
            )
            artifact_paths = self.db.list_file_artifact_paths(source_id=sid)
            cache_removed = self.db.remove_cache_artifacts(artifact_paths, cache_dir)
            removed = self.db.purge_source_records(sid, cache_dir=cache_dir)
            total_removed += int(removed or 0)
            _ = cache_removed

        self.db.set_meta("faiss_needs_rebuild", "1")
        emit(stage="done", message="Yetim kayıtlar temizlendi", files_removed=total_removed)
        return {
            "files_removed": total_removed,
            "orphan_source_ids": orphan_ids,
        }
    def purge_source_index(
        self,
        source_id: int,
        cache_dir: str,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Kaynağın index/cache kayıtlarını temizle.
        Orijinal dosyaları silmez. FAISS rebuild gerekir.
        """

        def emit(**extra):
            if progress_callback:
                progress_callback(extra)

        emit(stage="count", message="Kayıt sayısı hesaplanıyor…")
        counts = self.db.count_files_for_source(source_id)
        total_files = sum(int(v) for v in counts.values())

        rows = self.db.list_files_for_source(source_id)
        artifact_paths = self.db.list_file_artifact_paths(source_id=source_id)
        from core.preview_cache import FeaturePreviewCache
        from core.thumbnailer import Thumbnailer

        thumbnailer = Thumbnailer(
            cache_dir,
            max_edge=self.settings.thumbnail_max_edge,
            fmt=self.settings.thumbnail_format,
        )
        previews = FeaturePreviewCache(
            cache_dir,
            self.settings.feature_preview_max_edge,
            self.settings.thumbnail_format,
        )
        for row in rows:
            source_path = str(row.get("path") or "")
            if source_path:
                artifact_paths.extend(
                    [
                        str(thumbnailer.thumbnail_path_for(source_path)),
                        str(previews.preview_path_for(source_path)),
                    ]
                )
        cache_removed = self.db.remove_cache_artifacts(artifact_paths, cache_dir)
        emit(stage="db", message="Index kayıtları siliniyor…", total=total_files)
        removed = self.db.purge_source_records(source_id)
        self.db.delete_source(source_id)
        self.db.set_meta("faiss_needs_rebuild", "1")
        vector_files_removed = 0
        for raw in (self.settings.faiss_dino_path, self.settings.faiss_clip_path):
            index_path = Path(raw)
            for path in (index_path, index_path.with_suffix(".map.npy")):
                try:
                    if path.is_file():
                        path.unlink()
                        vector_files_removed += 1
                except OSError:
                    logger.warning("Vector index silinemedi: %s", path)

        emit(stage="done", message="Tamamlandı", files_removed=removed)
        return {
            "source_id": source_id,
            "files_removed": removed,
            "cache_files_removed": cache_removed,
            "vector_files_removed": vector_files_removed,
            "file_ids": list(range(removed)),
        }

    def purge_missing_index(
        self,
        source_id: int | None = None,
        progress_callback=None,
    ) -> dict[str, Any]:
        def emit(**extra):
            if progress_callback:
                progress_callback(extra)

        emit(stage="count", message="Silinmiş kayıtlar sayılıyor…")
        pending = self.db.count_missing_files(source_id)
        if pending == 0:
            emit(stage="done", message="Silinecek kayıt yok", files_removed=0)
            return {"files_removed": 0, "source_id": source_id or 0}

        emit(stage="db", message="Silinmiş kayıtlar temizleniyor…", total=pending)
        removed = self.db.purge_missing_files(
            source_id=source_id,
            cache_dir=self.settings.cache_dir,
        )
        emit(stage="done", message="Tamamlandı", files_removed=removed)
        return {"files_removed": removed, "source_id": source_id or 0}

    def migrate_legacy_archive_root(self) -> int | None:
        """Eski tek kök ayarından kaynak oluştur."""
        if not self.settings.archive_root:
            return None
        existing = self.db.get_source_by_path(self.settings.archive_root)
        if existing:
            return existing["id"]
        sources = self.db.list_sources()
        if sources:
            return None
        sid = self.add_source(
            name="Ana Arşiv",
            root_path=self.settings.archive_root,
            source_type=detect_source_type(self.settings.archive_root),
        )
        logger.info("Eski archive_root kaynağa taşındı: id=%d", sid)
        return sid

    def sources_due_for_scan(self) -> list[tuple[dict[str, Any], str]]:
        """Taranması gereken kaynakları (kaynak, scan_mode) olarak döndür."""
        due: list[tuple[dict[str, Any], str]] = []
        now = datetime.now(timezone.utc)
        for src in self.db.list_sources(active_only=True):
            mode = self._scan_mode_for_source(src, now)
            if mode:
                due.append((src, mode))
        return due

    def _scan_mode_for_source(self, src: dict[str, Any], now: datetime) -> str | None:
        if not src.get("is_active"):
            return None
        last_deep = _parse_iso(src.get("last_deep_scan_at") or src.get("last_scan_at"))
        last_quick = _parse_iso(
            src.get("last_quick_scan_at") or src.get("last_scan_at")
        )
        deep_days = (
            src.get("deep_scan_interval_days") or self.settings.deep_scan_interval_days
        )
        quick_hours = (
            src.get("scan_interval_hours") or self.settings.quick_scan_interval_hours
        )

        if last_deep is None or (now - last_deep) >= timedelta(days=deep_days):
            return ScanMode.DEEP.value
        if last_quick is None or (now - last_quick) >= timedelta(hours=quick_hours):
            return ScanMode.QUICK.value
        return None

    def update_source_stats(self, source_id: int) -> None:
        stats = self.db.count_files_for_source(source_id)
        cache_status = self._cache_status_for_source(source_id, stats)
        self.db.update_source_stats(
            source_id,
            file_count=stats.get("indexed", 0) + stats.get("pending", 0),
            error_count=stats.get("error", 0),
            cache_status=cache_status,
        )

    def _cache_status_for_source(self, source_id: int, stats: dict[str, int]) -> str:
        total = sum(stats.values())
        indexed = stats.get("indexed", 0)
        errors = stats.get("error", 0)
        if total == 0:
            return "empty"
        if errors > 0 and indexed == 0:
            return "error"
        if errors > 0:
            return "partial"
        if indexed < total:
            return "partial"
        return "ok"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None
