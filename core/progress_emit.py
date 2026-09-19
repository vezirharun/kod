"""Index ilerleme — UI donmasını önlemek için throttle."""

from __future__ import annotations

import time
from typing import Any, Callable

_FORCE_STAGES = frozenset(
    {
        "scan_start",
        "queue_prepare",
        "backlog_only",
        "process_batch_start",
        "heavy_backlog_start",
        "ai_loading",
        "fast_lane_complete",
        "source_complete",
        "complete",
    }
)


class ThrottledProgress:
    """Arka plan thread'den UI'ya en fazla ~4 güncelleme/saniye gönder."""

    def __init__(
        self,
        callback: Callable[[dict], None] | None,
        *,
        min_interval: float = 0.25,
        scan_every: int = 250,
        process_every: int = 20,
    ):
        self._callback = callback
        self._min_interval = min_interval
        self._scan_every = max(1, scan_every)
        self._process_every = max(1, process_every)
        self._last_emit = 0.0
        self._scanned = 0

    def emit(self, stage: str, **extra: Any) -> None:
        if not self._callback:
            return
        now = time.perf_counter()
        force = stage in _FORCE_STAGES
        if stage == "scan":
            self._scanned = int(extra.get("scanned", self._scanned + 1) or 0)
            if not force and self._scanned % self._scan_every != 0:
                if (now - self._last_emit) < self._min_interval * 2:
                    return
        if stage in ("process", "process_done"):
            jobs_done = int(extra.get("jobs_done", 0) or 0)
            jobs_total = int(extra.get("jobs_total", 0) or 0)
            # İlk dosyalar ve her N. dosya — UI "donmuş" görünmesin
            every = self._process_every if jobs_done > 5 else 1
            if (
                not force
                and jobs_done > 0
                and jobs_done < jobs_total
                and jobs_done % every != 0
                and (now - self._last_emit) < self._min_interval
            ):
                return
        if not force and (now - self._last_emit) < self._min_interval:
            return
        self._last_emit = now
        self._callback({"stage": stage, **extra})
