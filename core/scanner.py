"""Ağ klasörlerini recursive tarayan modül."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from core.logger import log_skipped_file, setup_logger
from core.settings import SUPPORTED_EXTENSIONS
from core.utils import customer_from_path, normalize_path, safe_stat

logger = setup_logger(__name__)

IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".appledouble",
        "__macosx",
        ".spotlight-v100",
        ".trashes",
        "#recycle",
        "@recycle",
        "$recycle.bin",
        "recycler",
        "system volume information",
        ".snapshot",
        "@eadir",
    }
)


def is_ignored_artifact_path(path: str) -> bool:
    """Return True for OS metadata files that only look like supported images."""
    normalized = str(path or "").replace("/", "\\")
    parts = [part.casefold() for part in normalized.split("\\") if part]
    name = parts[-1] if parts else ""
    return (
        any(part in IGNORED_DIRECTORY_NAMES for part in parts[:-1])
        or name.startswith("._")
        or name in {".ds_store", "thumbs.db", "desktop.ini"}
    )


@dataclass
class ScanResult:
    path: str
    filename: str
    customer: str
    file_size: int
    mtime: float
    source_id: int = 0
    is_new: bool = True
    is_changed: bool = False


class ArchiveScanner:
    def __init__(
        self,
        archive_root: str,
        timeout_sec: int = 30,
        retry_count: int = 3,
        excluded_roots: list[str] | None = None,
    ):
        self.archive_root = normalize_path(archive_root)
        self.timeout_sec = timeout_sec
        self.retry_count = retry_count
        self.excluded_roots = {
            os.path.normcase(os.path.normpath(str(path))).rstrip("\\/")
            for path in (excluded_roots or [])
            if path
        }
        self.diagnostics = {
            "total_dirs_seen": 0,
            "total_files_seen": 0,
            "supported_images_found": 0,
            "skipped_unsupported": 0,
            "failed_files": 0,
            "failed_dirs": 0,
            "first_50_errors": [],
            "skipped_internal_dirs": 0,
            "skipped_os_artifacts": 0,
        }

    def iter_files(
        self,
        customer_filter: str = "",
        subfolder: str = "",
        source_id: int = 0,
    ) -> Iterator[ScanResult]:
        """Arşivdeki desteklenen dosyaları yield et."""
        root = Path(self.archive_root)
        if subfolder:
            root = root / subfolder
        if not root.exists():
            logger.error("Arşiv kökü bulunamadı: %s", root)
            return

        if customer_filter:
            root = root / customer_filter
            if not root.exists():
                logger.warning("Müşteri klasörü bulunamadı: %s", root)
                return
            yield from self._walk(root, customer_filter, source_id)
        else:
            yield from self._walk(root, "", source_id)

    def _walk(
        self, root: Path, fixed_customer: str, source_id: int = 0
    ) -> Iterator[ScanResult]:
        def on_error(exc: OSError) -> None:
            self.diagnostics["failed_dirs"] += 1
            self._remember_error(
                str(getattr(exc, "filename", root)), str(exc), "directory"
            )

        try:
            for dirpath, dirnames, filenames in os.walk(
                root, topdown=True, onerror=on_error
            ):
                self.diagnostics["total_dirs_seen"] += 1
                kept: list[str] = []
                for dirname in dirnames:
                    child = os.path.normcase(
                        os.path.normpath(os.path.join(dirpath, dirname))
                    ).rstrip("\\/")
                    blocked_name = dirname.casefold() in {
                        ".venv",
                        ".pytest_cache",
                        "__pycache__",
                        *IGNORED_DIRECTORY_NAMES,
                    }
                    blocked_root = any(
                        child == root_key or child.startswith(root_key + os.sep)
                        for root_key in self.excluded_roots
                    )
                    if blocked_name or blocked_root:
                        self.diagnostics["skipped_internal_dirs"] += 1
                    else:
                        kept.append(dirname)
                dirnames[:] = kept
                for name in filenames:
                    self.diagnostics["total_files_seen"] += 1
                    full = normalize_path(Path(dirpath) / name)
                    if is_ignored_artifact_path(full):
                        self.diagnostics["skipped_os_artifacts"] += 1
                        continue
                    ext = Path(name).suffix.lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        self.diagnostics["skipped_unsupported"] += 1
                        continue
                    self.diagnostics["supported_images_found"] += 1
                    stat = safe_stat(full, self.timeout_sec, self.retry_count)
                    if stat is None:
                        self.diagnostics["failed_files"] += 1
                        self._remember_error(
                            full, "Dosya okunamadı veya ağ zaman aşımı", "stat"
                        )
                        log_skipped_file(full, "Dosya erişilemedi veya bulunamadı")
                        continue
                    customer = fixed_customer or customer_from_path(
                        full, self.archive_root
                    )
                    yield ScanResult(
                        path=full,
                        filename=name,
                        customer=customer,
                        file_size=stat.st_size,
                        mtime=stat.st_mtime,
                        source_id=source_id,
                    )
        except OSError as exc:
            self.diagnostics["failed_dirs"] += 1
            self._remember_error(str(root), str(exc), "walk")
            logger.error("Tarama hatası %s: %s", root, exc)

    def _remember_error(self, path: str, error: str, stage: str) -> None:
        errors = self.diagnostics["first_50_errors"]
        if len(errors) < 50:
            errors.append({"path": path, "stage": stage, "error": error})

    def scan_all(
        self,
        customer_filter: str = "",
        progress_callback: Callable[[ScanResult, int], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> list[ScanResult]:
        results: list[ScanResult] = []
        count = 0
        for item in self.iter_files(customer_filter=customer_filter):
            if stop_check and stop_check():
                break
            results.append(item)
            count += 1
            if progress_callback:
                progress_callback(item, count)
        return results

    @staticmethod
    def needs_reindex(
        existing: dict | None,
        file_size: int,
        mtime: float,
        partial_hash: str = "",
        new_partial_hash: str = "",
        deep_scan: bool = False,
    ) -> bool:
        """Dosyanın yeniden indexlenmesi gerekip gerekmediğini kontrol et."""
        if existing is None:
            return True
        if existing.get("status") == "missing":
            return True
        if existing.get("file_size") != file_size:
            return True
        if abs(existing.get("mtime", 0) - mtime) > 0.5:
            return True
        if (
            deep_scan
            and partial_hash
            and new_partial_hash
            and partial_hash != new_partial_hash
        ):
            return True
        # SSOT: light/heavy done kayıtlarını status tuzağıyla bozma.
        # pending/error yalnız henüz light_done olmayanlarda yeniden kuyruk gerekçesi.
        light = str(existing.get("light_status") or "")
        if light not in ("done",) and existing.get("status") in (
            "pending",
            "error",
            "changed",
            "processing",
        ):
            return True
        if deep_scan and existing.get("status") == "indexed":
            thumb = existing.get("thumbnail_path", "")
            if thumb and not os.path.exists(thumb):
                return True
        return False

    @staticmethod
    def content_changed(
        existing: dict | None,
        file_size: int,
        mtime: float,
    ) -> bool:
        """Güvenli envanter: yeni / missing / size|mtime — done sıfırlamaz."""
        from core.index_ssot import archive_content_changed

        return archive_content_changed(existing, file_size, mtime)
