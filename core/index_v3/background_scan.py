"""Arka plan full scan — index drain'den bağımsız, sistemi kilitlemez.

Görevler (periyodik, chunk'lı):
  1) physical_* stale bayrak reconcile
  2) DB gap → JobStore enqueue (Mode.COMPLETE: light+heavy gap)
  3) Seyrek disk walk → missing (silinen dosyalar)

Index worker yalnız kuyruğu işler; full scan bitmesini BEKLEMEZ.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.index_v3.discovery import (
    discover_source,
    enqueue_existing_gaps,
    source_root_is_dir,
)
from core.index_v3.physical_reconcile import reconcile_stale_physical_flags
from core.index_v3.queues import JobStore
from core.index_v3.types import Mode


@dataclass
class BackgroundScanStats:
    cycles: int = 0
    gaps_enqueued: int = 0
    physical_fixed: int = 0
    missing_marked: int = 0
    purged_missing: int = 0
    new_found: int = 0
    changed_found: int = 0
    unchanged_seen: int = 0
    source_errors: int = 0
    last_error: str = ""
    phase: str = "idle"
    last_cycle_at: float = 0.0


class BackgroundIndexScan:
    """Düşük öncelikli döngü — kısa chunk, aralarda sleep."""

    def __init__(
        self,
        db: Any,
        store: JobStore,
        *,
        settings: Any | None = None,
        gap_interval_sec: float = 90.0,
        walk_interval_sec: float = 900.0,
        chunk_size: int = 80,
        chunk_pause_sec: float = 0.05,
    ) -> None:
        self.db = db
        self.store = store
        self.settings = settings
        self.gap_interval_sec = float(gap_interval_sec)
        self.walk_interval_sec = float(walk_interval_sec)
        self.chunk_size = max(10, int(chunk_size))
        self.chunk_pause_sec = float(chunk_pause_sec)
        self.stats = BackgroundScanStats()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        # None = henüz disk walk yapılmadı. 0.0 + monotonic() < interval
        # açılışta (PC uptime < 15 dk) walk'ı sessizce atlıyordu.
        self._last_walk: float | None = None

    def request_stop(self) -> None:
        self._stop.set()
        self._pause.set()

    def pause(self) -> None:
        self._pause.clear()

    def resume(self) -> None:
        self._pause.set()

    def _wait(self) -> bool:
        while not self._pause.is_set():
            if self._stop.is_set():
                return False
            self._pause.wait(0.2)
        return not self._stop.is_set()

    def run_once(
        self,
        *,
        source_provider: Callable[[], list[dict[str, Any]]],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        force_walk: bool = True,
        purge_missing: bool = False,
    ) -> dict[str, Any]:
        """Elle 'Sistemi baştan tara' — bir tam cycle (walk zorunlu)."""
        prev_walk = self._last_walk
        try:
            self._one_cycle(
                source_provider, progress_callback, force_walk=force_walk
            )
            if purge_missing:
                self._purge_missing_records()
        finally:
            if not force_walk:
                self._last_walk = prev_walk
        self.stats.cycles += 1
        self.stats.last_cycle_at = time.time()
        self.stats.phase = "idle"
        return {
            **self._public_stats(),
            "manual": True,
        }

    def _public_stats(self) -> dict[str, Any]:
        marked = int(self.stats.missing_marked or 0)
        purged = int(self.stats.purged_missing or 0)
        deleted = purged if purged > 0 else marked
        return {
            "cycles": self.stats.cycles,
            "gaps_enqueued": self.stats.gaps_enqueued,
            "physical_fixed": self.stats.physical_fixed,
            "missing_marked": marked,
            "deleted_files": deleted,
            "purged_missing": purged,
            "new_found": self.stats.new_found,
            "changed_found": self.stats.changed_found,
            "unchanged_seen": self.stats.unchanged_seen,
            "source_errors": self.stats.source_errors,
            "last_error": self.stats.last_error,
        }

    def _purge_missing_records(self) -> None:
        try:
            n = int(self.db.purge_missing_files() or 0)
            self.stats.purged_missing += n
        except Exception:
            try:
                with self.db.connect() as conn:
                    cur = conn.execute("DELETE FROM files WHERE status='missing'")
                    self.stats.purged_missing += int(cur.rowcount or 0)
            except Exception as exc:
                self.stats.last_error = f"purge:{exc}"[:300]

    def run_until_stopped(
        self,
        *,
        source_provider: Callable[[], list[dict[str, Any]]],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        purge_missing: bool = False,
    ) -> dict[str, Any]:
        while not self._stop.is_set():
            if not self._wait():
                break
            try:
                self._one_cycle(source_provider, progress_callback, force_walk=False)
                if purge_missing:
                    self._purge_missing_records()
            except Exception as exc:
                self.stats.last_error = str(exc)[:300]
                self.stats.phase = "error"
            self.stats.cycles += 1
            self.stats.last_cycle_at = time.time()
            self.stats.phase = "idle"
            end = time.monotonic() + self.gap_interval_sec
            while time.monotonic() < end:
                if self._stop.is_set():
                    break
                if not self._pause.is_set():
                    self._pause.wait(0.2)
                    continue
                time.sleep(0.25)
        return self._public_stats()

    def _emit(
        self, cb: Callable[[dict[str, Any]], None] | None, **extra: Any
    ) -> None:
        if cb is None:
            return
        try:
            cb(
                {
                    "phase": self.stats.phase,
                    **self._public_stats(),
                    **extra,
                }
            )
        except Exception:
            pass

    def _one_cycle(
        self,
        source_provider: Callable[[], list[dict[str, Any]]],
        progress_callback: Callable[[dict[str, Any]], None] | None,
        *,
        force_walk: bool = False,
    ) -> None:
        sources = list(source_provider() or [])
        source_ids = [int(s["id"]) for s in sources if int(s.get("id") or 0) > 0]
        if not source_ids:
            return

        if not self._wait():
            return
        self.stats.phase = "physical_reconcile"
        self._emit(progress_callback)
        try:
            self.stats.physical_fixed += int(
                reconcile_stale_physical_flags(
                    self.db,
                    source_ids,
                    cache_dir=str(getattr(self.settings, "cache_dir", "") or "")
                    or None,
                )
                or 0
            )
        except Exception as exc:
            self.stats.last_error = f"physical:{exc}"[:300]

        # Mode B — candidate-only preview self-heal (no full-archive decode).
        if not self._wait():
            return
        self.stats.phase = "preview_self_heal"
        self._emit(progress_callback)
        try:
            from core.preview_self_heal.hooks import heal_candidates_for_sources

            busy = False
            try:
                busy = int(self.store.count_pending() or 0) > 200
            except Exception:
                busy = False
            heal = heal_candidates_for_sources(
                self.db,
                self.store,
                source_ids,
                limit=max(10, min(self.chunk_size, 40)),
                settings=self.settings,
                index_busy=busy,
            )
            self.stats.gaps_enqueued += int(heal.get("enqueued") or 0)
        except Exception as exc:
            self.stats.last_error = f"preview_heal:{exc}"[:300]

        if not self._wait():
            return
        self.stats.phase = "gap_scan"
        self._emit(progress_callback)
        for sid in source_ids:
            if not self._wait():
                return
            after_id = 0
            while not self._stop.is_set():
                if not self._wait():
                    return
                n, last_id = enqueue_existing_gaps(
                    self.db,
                    self.store,
                    source_id=int(sid),
                    mode=Mode.COMPLETE,
                    limit=self.chunk_size,
                    after_id=after_id,
                    ocr_enabled=bool(
                        getattr(self.settings, "ocr_enabled", False)
                    ),
                    patch_enabled=bool(
                        getattr(self.settings, "auto_patch_after_ai_final", True)
                    ),
                )
                self.stats.gaps_enqueued += int(n)
                self._emit(progress_callback, source_id=sid, after_id=last_id)
                if not last_id:
                    break
                after_id = int(last_id)
                if self.chunk_pause_sec > 0:
                    time.sleep(self.chunk_pause_sec)

        if not self._wait():
            return
        self.stats.phase = "autonomous_learn"
        self._emit(progress_callback)
        try:
            from core.autonomous_learn import run_autonomous_pass

            db_path = str(getattr(self.settings, "db_path", "") or "")
            if db_path:
                run_autonomous_pass(
                    self.db,
                    db_path,
                    limit=max(40, int(self.chunk_size)),
                )
        except Exception as exc:
            self.stats.last_error = f"autonomous:{exc}"[:300]

        now = time.monotonic()
        never_walked = self._last_walk is None
        due = never_walked or (now - float(self._last_walk) >= self.walk_interval_sec)
        if not force_walk and not due:
            return
        if not self._wait():
            return
        self.stats.phase = "disk_walk"
        self._emit(progress_callback)
        self._last_walk = now
        for src in sources:
            if not self._wait():
                return
            sid = int(src.get("id") or 0)
            root = str(src.get("root_path") or "").strip()
            if not sid or not root or not source_root_is_dir(root):
                self.stats.source_errors += 1
                continue
            try:
                st = discover_source(
                    self.db,
                    self.store,
                    source_id=sid,
                    root_path=root,
                    mode=Mode.FAST,
                    mark_missing=True,
                    ocr_enabled=bool(
                        getattr(self.settings, "ocr_enabled", False)
                    ),
                    patch_enabled=bool(
                        getattr(self.settings, "auto_patch_after_ai_final", True)
                    ),
                    settings=self.settings,
                )
                self.stats.new_found += int(st.inserted or 0)
                self.stats.changed_found += int(st.changed or 0)
                self.stats.unchanged_seen += int(st.unchanged or 0)
                self.stats.missing_marked += int(st.missing_marked or 0)
                self.stats.purged_missing += int(st.purged_missing or 0)
                self.stats.gaps_enqueued += int(st.jobs_enqueued or 0)
            except Exception as exc:
                self.stats.source_errors += 1
                self.stats.last_error = f"walk:{exc}"[:300]
            self._emit(progress_callback, source_id=sid)
            if self.chunk_pause_sec > 0:
                time.sleep(self.chunk_pause_sec)
