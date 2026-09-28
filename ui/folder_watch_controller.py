"""Kaynak klasörlerini izle — silinen dosyayı hemen index'ten düşür, eklemeyi tara."""

from __future__ import annotations

import os
import queue

from PySide6.QtCore import QFileSystemWatcher, QObject, QThread, QTimer, Signal

from core.logger import setup_logger
from core.settings import AppSettings
from core.sources import SourceManager
from core.utils import normalize_path, normalize_source_root
from core.win_subtree_watch import SubtreeDeleteWatch

logger = setup_logger(__name__)


class _DiskPurgeThread(QThread):
    """DB/cache/job/FAISS purge off the UI thread."""

    finished_count = Signal(int)

    def __init__(
        self,
        db,
        settings,
        paths: list[str],
        watched_roots: list[str],
        tree_paths: list[str],
    ) -> None:
        super().__init__()
        self._db = db
        self._settings = settings
        self._paths = list(paths)
        self._roots = list(watched_roots)
        self._trees = list(tree_paths)

    def run(self) -> None:
        removed = 0
        try:
            from core.disk_delete_purge import purge_disk_deletes

            result = purge_disk_deletes(
                self._db,
                self._settings,
                self._paths,
                watched_roots=self._roots,
                tree_paths=self._trees,
            )
            removed = int(result.get("removed") or 0)
        except Exception:
            logger.exception("instant disk-delete purge failed")
        self.finished_count.emit(removed)


class FolderWatchController(QObject):
    """Aktif kaynak köklerini izler.

    Silinen bir dosya index'ten hemen düşer. Yeni dosyalar debounce ile quick
    scan ister. Windows'ta alt klasörler ReadDirectoryChangesW ile izlenir.
    """

    scan_requested = Signal(int, str)  # source_id, reason
    files_purged = Signal(int)

    def __init__(
        self, settings: AppSettings, source_manager: SourceManager, parent=None
    ):
        super().__init__(parent)
        self.settings = settings
        self.source_manager = source_manager
        self._watcher = QFileSystemWatcher(self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._delete_timer = QTimer(self)
        self._delete_timer.setSingleShot(True)
        self._delete_timer.setInterval(100)
        self._native_poll = QTimer(self)
        self._native_poll.setInterval(50)
        self._pending_paths: set[str] = set()
        self._deleted_paths: set[str] = set()
        self._tree_paths: set[str] = set()
        self._watch_roots: list[str] = []
        self._root_to_source: dict[str, int] = {}
        self._native_q: queue.SimpleQueue = queue.SimpleQueue()
        self._purge_busy = False
        self._purge_worker: _DiskPurgeThread | None = None
        self._native = SubtreeDeleteWatch(
            self._queue_native_removed, self._queue_native_overflow
        )
        self._watcher.directoryChanged.connect(self._on_directory_changed)
        self._debounce.timeout.connect(self._flush_pending)
        self._delete_timer.timeout.connect(self._flush_deleted_paths)
        self._native_poll.timeout.connect(self._drain_native_queue)
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
        self._watch_roots = []
        for path, source_id in entries:
            norm = normalize_path(path)
            if norm and os.path.isdir(norm):
                self._watch_roots.append(norm)
            if self._watcher.addPath(path):
                self._root_to_source[norm.casefold() if norm else path.casefold()] = int(
                    source_id
                )
        self._sync_native_watch()

    def _apply_debounce_interval(self) -> None:
        sec = max(
            1.0, float(getattr(self.settings, "folder_watch_debounce_sec", 3.0) or 3.0)
        )
        self._debounce.setInterval(int(sec * 1000))

    def set_enabled(self, enabled: bool) -> None:
        self.settings.folder_watch_enabled = bool(enabled)
        if not enabled:
            self._deleted_paths.clear()
            self._tree_paths.clear()
            self.apply_watch_paths([])

    def refresh_watch_paths(self) -> None:
        self._apply_debounce_interval()
        current = self._watcher.directories()
        if current:
            self._watcher.removePaths(current)
        self._root_to_source.clear()
        self._watch_roots = []
        if not getattr(self.settings, "folder_watch_enabled", True):
            self._sync_native_watch()
            return

        for src in self.source_manager.list_sources(active_only=True):
            root = normalize_source_root(src.get("root_path", ""))
            if not root or not os.path.isdir(root):
                continue
            norm = normalize_path(root)
            key = norm.casefold()
            if key in self._root_to_source:
                continue
            if os.path.isdir(norm):
                self._watch_roots.append(norm)
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
                if os.path.isdir(quick) and quick not in self._watch_roots:
                    self._watch_roots.append(quick)
                if self._watcher.addPath(quick):
                    self._root_to_source[key] = sid
        self._sync_native_watch()

    def _sync_native_watch(self) -> None:
        """Recursive Windows delete watch. Qt still watches the source roots."""
        if not getattr(self.settings, "folder_watch_enabled", True):
            self._native.stop()
            self._native_poll.stop()
            return
        running = self._native.set_roots(list(self._watch_roots))
        if running:
            self._native_poll.start()
        else:
            self._native_poll.stop()
            if os.name == "nt" and self._watch_roots:
                logger.warning(
                    "Recursive delete watch did not start; "
                    "only direct children of watched folders are purged immediately"
                )

    def _queue_native_removed(self, path: str) -> None:
        self._native_q.put((str(path or ""), False))

    def _queue_native_overflow(self, root: str) -> None:
        self._native_q.put((str(root or ""), True))

    def _drain_native_queue(self) -> None:
        changed = False
        while True:
            try:
                path, tree = self._native_q.get_nowait()
            except queue.Empty:
                break
            norm = normalize_path(path)
            if not norm:
                continue
            if tree:
                self._tree_paths.add(norm)
                self._pending_paths.add(norm)
                self._debounce.start()
            else:
                self._deleted_paths.add(norm)
            changed = True
        if changed:
            self._delete_timer.start()

    def note_filesystem_removal(self, path: str) -> None:
        """Queue an immediate purge for a deleted path or a changed directory."""
        if not getattr(self.settings, "folder_watch_enabled", True):
            return
        norm = normalize_path(path)
        if not norm:
            return
        self._deleted_paths.add(norm)
        self._delete_timer.start()

    def _on_directory_changed(self, path: str) -> None:
        if not getattr(self.settings, "folder_watch_enabled", True):
            return
        norm = normalize_path(path)
        self._pending_paths.add(norm)
        self._debounce.start()
        self.note_filesystem_removal(norm)

    def _flush_deleted_paths(self) -> None:
        if self._purge_busy:
            self._delete_timer.start()
            return
        paths = list(self._deleted_paths)
        trees = list(self._tree_paths)
        self._deleted_paths.clear()
        self._tree_paths.clear()
        if not paths and not trees:
            return
        self._purge_busy = True
        worker = _DiskPurgeThread(
            self.source_manager.db,
            self.settings,
            paths,
            list(self._watch_roots),
            trees,
        )
        worker.finished_count.connect(self._on_disk_purge_finished)
        worker.finished.connect(worker.deleteLater)
        self._purge_worker = worker
        worker.start()

    def _on_disk_purge_finished(self, removed: int) -> None:
        self._purge_busy = False
        self._purge_worker = None
        count = int(removed or 0)
        if count > 0:
            logger.info("Disk delete purged %s index file(s)", count)
            self.files_purged.emit(count)
        if self._deleted_paths or self._tree_paths:
            self._delete_timer.start()

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

