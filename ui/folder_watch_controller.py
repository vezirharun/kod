"""Kaynak klasörlerini izle — dosya ekleme/silme sonrası hızlı tarama tetikle."""

from __future__ import annotations

import os

from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer, Signal

from core.logger import setup_logger
from core.settings import AppSettings
from core.sources import SourceManager
from core.utils import normalize_path, normalize_source_root

logger = setup_logger(__name__)


class FolderWatchController(QObject):
    """Aktif kaynak köklerini izler; değişiklikte debounce ile quick scan ister."""

    scan_requested = Signal(int, str)  # source_id, reason

    def __init__(
        self, settings: AppSettings, source_manager: SourceManager, parent=None
    ):
        super().__init__(parent)
        self.settings = settings
        self.source_manager = source_manager
        self._watcher = QFileSystemWatcher(self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._pending_paths: set[str] = set()
        self._root_to_source: dict[str, int] = {}
        self._watcher.directoryChanged.connect(self._on_directory_changed)
        self._debounce.timeout.connect(self._flush_pending)
        self._apply_debounce_interval()

    def discover_watch_paths(self) -> list[tuple[str, int]]:
        """Probe local/network paths outside the Qt event loop."""
        if not getattr(self.settings, "folder_watch_enabled", True):
            return []
        entries: list[tuple[str, int]] = []
        seen: set[str] = set()
        for src in self.source_manager.list_sources(active_only=True):
            root = normalize_source_root(src.get("root_path", ""))
            if not root or _is_remote_path(root) or not os.path.isdir(root):
                continue
            norm = normalize_path(root)
            key = norm.casefold()
            if key not in seen:
                entries.append((norm, int(src["id"])))
                seen.add(key)
        quick = normalize_path(getattr(self.settings, "quick_search_folder", "") or "")
        if (
            quick and quick.casefold() not in seen
            and not _is_remote_path(quick) and os.path.isdir(quick)
        ):
            src = self.source_manager.db.get_source_by_path(quick)
            entries.append((quick, int(src["id"]) if src else 0))
        return entries

    def apply_watch_paths(self, entries: list[tuple[str, int]]) -> None:
        current = self._watcher.directories()
        if current:
            self._watcher.removePaths(current)
        self._root_to_source.clear()
        for path, source_id in entries:
            if self._watcher.addPath(path):
                self._root_to_source[path.casefold()] = int(source_id)

    def _apply_debounce_interval(self) -> None:
        sec = max(
            1.0, float(getattr(self.settings, "folder_watch_debounce_sec", 3.0) or 3.0)
        )
        self._debounce.setInterval(int(sec * 1000))

    def set_enabled(self, enabled: bool) -> None:
        self.settings.folder_watch_enabled = bool(enabled)
        if not enabled:
            self.apply_watch_paths([])

    def refresh_watch_paths(self) -> None:
        self._apply_debounce_interval()
        current = self._watcher.directories()
        if current:
            self._watcher.removePaths(current)
        self._root_to_source.clear()
        if not getattr(self.settings, "folder_watch_enabled", True):
            return

        for src in self.source_manager.list_sources(active_only=True):
            root = normalize_source_root(src.get("root_path", ""))
            if not root or not os.path.isdir(root):
                continue
            norm = normalize_path(root)
            key = norm.casefold()
            if key in self._root_to_source:
                continue
            if self._watcher.addPath(norm):
                self._root_to_source[key] = int(src["id"])
                logger.debug(
                    "Klasör izleniyor: %s (kaynak %s)", norm, src.get("name", "")
                )

        quick = normalize_path(getattr(self.settings, "quick_search_folder", "") or "")
        if quick and os.path.isdir(quick):
            key = quick.casefold()
            if key not in self._root_to_source:
                src = self.source_manager.db.get_source_by_path(quick)
                sid = int(src["id"]) if src else 0
                if self._watcher.addPath(quick):
                    self._root_to_source[key] = sid

    def _on_directory_changed(self, path: str) -> None:
        if not getattr(self.settings, "folder_watch_enabled", True):
            return
        self._pending_paths.add(normalize_path(path))
        self._debounce.start()

    def _resolve_source_id(self, changed_path: str) -> int:
        norm = normalize_path(changed_path).casefold()
        if norm in self._root_to_source:
            return self._root_to_source[norm]
        best_id = 0
        best_len = -1
        for root, sid in self._root_to_source.items():
            if norm.startswith(root.rstrip("\\/") + "\\") or norm == root:
                if len(root) > best_len:
                    best_len = len(root)
                    best_id = sid
        return best_id

    def _flush_pending(self) -> None:
        if not self._pending_paths:
            return
        source_ids: set[int] = set()
        for path in self._pending_paths:
            sid = self._resolve_source_id(path)
            if sid:
                source_ids.add(sid)
        self._pending_paths.clear()
        for sid in sorted(source_ids):
            logger.info("Folder change quick scan: source %d", sid)
            self.scan_requested.emit(sid, "folder_watch")
        if not source_ids:
            logger.debug("Folder change did not match a source")


def _is_remote_path(path: str) -> bool:
    if path.startswith("\\\\"):
        return True
    if os.name != "nt":
        return False
    drive, _ = os.path.splitdrive(path)
    if not drive:
        return False
    try:
        import ctypes
        return int(ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")) == 4
    except Exception:
        return False

