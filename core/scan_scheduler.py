"""Açılışta ve periyodik incremental tarama planlayıcı."""

from __future__ import annotations

import os
from typing import Callable

from core.indexer import Indexer
from core.logger import setup_logger
from core.settings import AppSettings
from core.sources import ScanMode, SourceManager

logger = setup_logger(__name__)


class ScanScheduler:
    """Kaynak bazlı hızlı/derin tarama planlar ve çalıştırır."""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.source_manager = SourceManager(settings)

    def pending_scans(self) -> list[dict]:
        """UI için bekleyen taramaları listele."""
        items = []
        for src, mode in self.source_manager.sources_due_for_scan():
            items.append(
                {
                    "source_id": src["id"],
                    "source_name": src["name"],
                    "root_path": src["root_path"],
                    "scan_mode": mode,
                    "scan_mode_label": (
                        "Derin kontrol"
                        if mode == ScanMode.DEEP.value
                        else "Hızlı güncelleme"
                    ),
                }
            )
        return items

    def should_auto_scan_on_startup(self) -> bool:
        if not self.settings.auto_scan_on_startup:
            return False
        return len(self.pending_scans()) > 0

    def run_due_scans(
        self,
        progress_callback: Callable[[dict], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
        customer_filter: str = "",
    ) -> dict:
        """Sadece zamanı gelen kaynakları incremental tarar."""
        self.source_manager.migrate_legacy_archive_root()
        due = self.source_manager.sources_due_for_scan()
        if not due:
            logger.info("Taranması gereken kaynak yok — atlanıyor")
            return {"skipped_all": True, "sources_scanned": 0}

        indexer = Indexer(self.settings)
        combined: dict = {
            "sources_scanned": 0,
            "scanned": 0,
            "processed": 0,
            "errors": 0,
            "skipped": 0,
            "new": 0,
            "changed": 0,
            "source_results": [],
        }

        for src, mode in due:
            if stop_check and stop_check():
                break
            if not os.path.exists(src.get("root_path", "")):
                logger.warning("Kaynak erişilemiyor, tarama atlandı.")
                continue
            logger.info(
                "Kaynak taranıyor: %s [%s] mode=%s",
                src["name"],
                src["root_path"],
                mode,
            )
            try:
                stats = indexer.run_source(
                    source=src,
                    scan_mode=mode,
                    progress_callback=progress_callback,
                    customer_filter=customer_filter,
                )
            except OSError as exc:
                logger.warning("Kaynak erişilemiyor, tarama atlandı. (%s)", exc)
                continue
            except Exception as exc:
                logger.warning(
                    "Kaynak tarama hatası, atlandı: %s — %s", src.get("name"), exc
                )
                continue
            combined["sources_scanned"] += 1
            for k in ("scanned", "processed", "errors", "skipped", "new", "changed"):
                combined[k] += stats.get(k, 0)
            combined["source_results"].append(
                {
                    "source_id": src["id"],
                    "source_name": src["name"],
                    "scan_mode": mode,
                    **stats,
                }
            )

        return combined

    def run_source(
        self,
        source_id: int,
        scan_mode: str = ScanMode.QUICK.value,
        progress_callback: Callable[[dict], None] | None = None,
        customer_filter: str = "",
    ) -> dict:
        """Tek kaynağı zorla tara."""
        src = self.source_manager.get_source(source_id)
        if not src:
            raise ValueError(f"Kaynak bulunamadı: {source_id}")
        if not os.path.exists(src.get("root_path", "")):
            logger.warning("Kaynak erişilemiyor, tarama atlandı.")
            return {
                "source_id": source_id,
                "skipped_all": True,
                "scanned": 0,
                "processed": 0,
                "errors": 0,
            }
        indexer = Indexer(self.settings)
        try:
            return indexer.run_source(
                source=src,
                scan_mode=scan_mode,
                progress_callback=progress_callback,
                customer_filter=customer_filter,
            )
        except OSError as exc:
            logger.warning("Kaynak erişilemiyor, tarama atlandı. (%s)", exc)
            return {
                "source_id": source_id,
                "skipped_all": True,
                "scanned": 0,
                "processed": 0,
                "errors": 0,
            }

    def run_all_active(
        self,
        scan_mode: str = ScanMode.QUICK.value,
        progress_callback: Callable[[dict], None] | None = None,
        customer_filter: str = "",
    ) -> dict:
        """Tüm aktif kaynakları tara (manuel index)."""
        self.source_manager.migrate_legacy_archive_root()
        sources = self.source_manager.list_sources(active_only=True)
        if not sources:
            raise ValueError("Aktif kaynak tanımlı değil")

        indexer = Indexer(self.settings)
        combined: dict = {
            "sources_scanned": 0,
            "scanned": 0,
            "processed": 0,
            "errors": 0,
            "skipped": 0,
            "source_results": [],
        }
        for src in sources:
            stats = indexer.run_source(
                source=src,
                scan_mode=scan_mode,
                progress_callback=progress_callback,
                customer_filter=customer_filter,
            )
            combined["sources_scanned"] += 1
            for k in ("scanned", "processed", "errors", "skipped"):
                combined[k] += stats.get(k, 0)
            combined["source_results"].append({"source_id": src["id"], **stats})
        return combined
