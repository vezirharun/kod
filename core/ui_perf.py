"""UI FPS ölçer + donma dedektörü (kanıtlı stack trace)."""

from __future__ import annotations

import faulthandler
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from core.logger import setup_logger

logger = setup_logger(__name__)

FREEZE_INTERACT_SEC = 0.10
FREEZE_WARNING_SEC = 0.25
FREEZE_CRITICAL_SEC = 0.50
FREEZE_THRESHOLD_SEC = 0.50  # critical dump threshold (was 1.0s)
WATCHDOG_POLL_SEC = 0.25
FREEZE_DUMP_COOLDOWN_SEC = 5.0


@dataclass
class UiPerfSnapshot:
    ui_fps: float = 60.0
    search_queue: int = 0
    preview_queue: int = 0
    thumbnails_per_sec: float = 0.0
    last_freeze_sec: float = 0.0
    freeze_count: int = 0


class UiPerfMonitor(QObject):
    """Ana thread heartbeat + arka plan watchdog."""

    updated = Signal(object)

    def __init__(self, log_dir: str | Path | None = None, parent=None):
        super().__init__(parent)
        self._log_dir = Path(log_dir or "data/logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._freeze_log = self._log_dir / "ui_freeze_tracebacks.log"
        self._heartbeat = time.perf_counter()
        self._heartbeat_lock = threading.Lock()
        self._frame_times: list[float] = []
        self._thumb_times: list[float] = []
        self._search_queue = 0
        self._preview_queue = 0
        self._freeze_count = 0
        self._last_freeze = 0.0
        self._watchdog_warned = False
        self._watchdog_interact_noted = False
        self._last_freeze_dump_at = 0.0
        self._stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="ui-freeze-watchdog", daemon=True
        )
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._tick_frame)
        self._fps_timer.start(16)
        self._emit_timer = QTimer(self)
        self._emit_timer.timeout.connect(self._emit_snapshot)
        self._emit_timer.start(500)
        self._watchdog.start()

    def touch_heartbeat(self) -> None:
        with self._heartbeat_lock:
            self._heartbeat = time.perf_counter()

    def set_search_queue(self, count: int) -> None:
        self._search_queue = max(0, int(count))

    def set_preview_queue(self, count: int) -> None:
        self._preview_queue = max(0, int(count))

    def record_thumbnail_loaded(self) -> None:
        now = time.perf_counter()
        self._thumb_times.append(now)
        cutoff = now - 1.0
        self._thumb_times = [t for t in self._thumb_times if t >= cutoff]

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> UiPerfSnapshot:
        now = time.perf_counter()
        cutoff = now - 1.0
        frames = [t for t in self._frame_times if t >= cutoff]
        thumbs = [t for t in self._thumb_times if t >= cutoff]
        fps = len(frames) if frames else 60.0
        return UiPerfSnapshot(
            ui_fps=float(min(999, fps)),
            search_queue=self._search_queue,
            preview_queue=self._preview_queue,
            thumbnails_per_sec=float(len(thumbs)),
            last_freeze_sec=self._last_freeze,
            freeze_count=self._freeze_count,
        )

    def _tick_frame(self) -> None:
        self.touch_heartbeat()
        self._frame_times.append(time.perf_counter())

    def _emit_snapshot(self) -> None:
        self.updated.emit(self.snapshot())

    def _watchdog_loop(self) -> None:
        while not self._stop.wait(WATCHDOG_POLL_SEC):
            with self._heartbeat_lock:
                gap = time.perf_counter() - self._heartbeat
            if gap < FREEZE_INTERACT_SEC:
                self._watchdog_warned = False
                self._watchdog_interact_noted = False
                continue
            blocked_ms = gap * 1000.0
            if gap < FREEZE_WARNING_SEC:
                if not self._watchdog_interact_noted:
                    self._watchdog_interact_noted = True
                    logger.info(
                        "[UI-WATCHDOG] blocked=%.0fms level=interact",
                        blocked_ms,
                    )
                continue
            if gap < FREEZE_CRITICAL_SEC:
                if not self._watchdog_warned:
                    self._watchdog_warned = True
                    logger.warning(
                        "[UI-WATCHDOG] blocked=%.0fms level=warning",
                        blocked_ms,
                    )
                continue
            self._freeze_count += 1
            self._last_freeze = gap
            self._watchdog_warned = False
            self._watchdog_interact_noted = False
            logger.critical(
                "[UI-WATCHDOG] blocked=%.0fms level=critical",
                blocked_ms,
            )
            now = time.perf_counter()
            if now - self._last_freeze_dump_at >= FREEZE_DUMP_COOLDOWN_SEC:
                self._last_freeze_dump_at = now
                self._log_freeze(gap)
            else:
                self.touch_heartbeat()

    def _log_freeze(self, gap_sec: float) -> None:
        header = (
            f"\n{'=' * 72}\n"
            f"UI FREEZE detected gap={gap_sec:.2f}s at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        logger.critical("UI freeze %.2fs — stack trace → %s", gap_sec, self._freeze_log)
        try:
            with open(self._freeze_log, "a", encoding="utf-8") as fh:
                fh.write(header)
                faulthandler.dump_traceback(file=fh, all_threads=True)
                fh.write("\n")
        except OSError as exc:
            logger.error("Freeze log yazılamadı: %s", exc)
        try:
            with open(self._freeze_log, "a", encoding="utf-8") as fh:
                fh.write("Python stack (main thread fallback):\n")
                traceback.print_stack(file=fh)
                fh.write("\n")
        except OSError:
            pass
        self.touch_heartbeat()
