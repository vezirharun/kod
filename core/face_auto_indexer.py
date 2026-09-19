"""Always-on face indexing bridge.

The face index is a secondary index. It follows the authoritative pattern DB and
never blocks the pattern index or UI. It scans missing/changed raster files in
small batches and can repair split identities after enough observations exist.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from core.face_scanner import FaceIndexScanner

logger = logging.getLogger(__name__)


class FaceAutoIndexer:
    def __init__(self, settings: Any, *, interval_sec: float = 5.0, batch_size: int = 120):
        self.settings = settings
        self.interval_sec = max(2.0, float(interval_sec))
        self.batch_size = max(20, int(batch_size))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._scanner: FaceIndexScanner | None = None
        self._last_reconcile = 0.0
        self._last_merge = 0.0
        self._last_log = 0.0

    def start(self) -> None:
        if not bool(getattr(self.settings, "face_index_enabled", False)):
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="FaceAutoIndexer", daemon=True)
        self._thread.start()
        logger.info("Yüz zekâsı arka plan indeksi başlatıldı.")

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)

    def _run(self) -> None:
        try:
            self._scanner = FaceIndexScanner(
                getattr(self.settings, "db_path", ""),
                getattr(self.settings, "face_db_path", ""),
                threshold=float(getattr(self.settings, "face_identity_threshold", 0.56) or 0.56),
                min_margin=float(getattr(self.settings, "face_identity_min_margin", 0.035) or 0.035),
            )
        except Exception:
            logger.exception("Yüz zekâsı arka plan indeksi kurulamadı.")
            return

        dependency_wait = 30.0
        try:
            # Repair already-existing V8 identities immediately; this is not
            # limited to newly indexed files.
            self._scanner.reconcile()
            merged = self._scanner.store.consolidate_identities(merge_threshold=0.70, max_pairs=1500)
            if merged:
                logger.info("Mevcut yüz kimlikleri ilk açılışta birleştirildi: %s", merged)
            self._last_merge = time.monotonic()
        except Exception:
            logger.debug("Yüz ilk kimlik onarımı atlandı", exc_info=True)

        while not self._stop.is_set():
            try:
                result = self._scanner.scan_incremental(batch_size=self.batch_size)
                if result.get("dependency_missing"):
                    self._stop.wait(dependency_wait)
                    continue
                now = time.monotonic()
                if now - self._last_reconcile >= 120.0:
                    self._scanner.reconcile()
                    self._last_reconcile = now
                if result.get("scanned", 0) and now - self._last_merge >= 300.0:
                    merged = self._scanner.store.consolidate_identities(merge_threshold=0.70, max_pairs=1500)
                    self._last_merge = now
                    if merged:
                        logger.info("Yüz kimlik kümeleri birleştirildi: %s", merged)
                if result.get("scanned", 0) or result.get("errors", 0):
                    if now - self._last_log >= 15.0:
                        logger.info("Yüz indeksi: taranan=%s yüz=%s hata=%s toplam=%s kişi=%s",
                                    result.get("scanned", 0), result.get("faces", 0),
                                    result.get("errors", 0), result.get("stats", {}).get("faces", 0),
                                    result.get("stats", {}).get("persons", 0))
                        self._last_log = now
                # Drain quickly while the pattern index is actively adding files;
                # otherwise keep CPU/NAS pressure low.
                self._stop.wait(0.25 if result.get("scanned", 0) else self.interval_sec)
            except Exception:
                logger.exception("Yüz zekâsı arka plan döngüsünde hata")
                self._stop.wait(10.0)
