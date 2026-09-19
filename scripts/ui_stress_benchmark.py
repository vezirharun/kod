"""UI Stress Test & Performance Benchmark

Gerçek MainWindow üzerinde etkileşim stresi + ölçüm.

Calistir:
  python scripts/ui_stress_benchmark.py --searches 20 --save
  python scripts/ui_stress_benchmark.py --searches 10 --offscreen
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if os.environ.get("UI_STRESS_OFFSCREEN", "").strip() in ("1", "true", "yes"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, QSize, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea

from core.settings import AppSettings
from core.ui_perf import UiPerfMonitor


@dataclass
class StressReport:
    timestamp: str = ""
    duration_s: float = 0.0
    phase_startup_s: float = 0.0
    phase_stress_s: float = 0.0
    searches_requested: int = 20
    searches_started: int = 0
    searches_finished: int = 0
    search_durations_ms: list[float] = field(default_factory=list)
    avg_search_ms: float = 0.0
    max_search_ms: float = 0.0
    samples: int = 0
    avg_ui_fps: float = 0.0
    min_ui_fps: float = 0.0
    max_event_lag_ms: float = 0.0
    avg_event_lag_ms: float = 0.0
    p95_event_lag_ms: float = 0.0
    max_main_thread_block_ms: float = 0.0
    blocks_over_100ms: int = 0
    frame_drops: int = 0
    freeze_count_1s: int = 0
    anr_detected: bool = False
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_rss_mb: float = 0.0
    max_rss_mb: float = 0.0
    panel_switches: int = 0
    scroll_steps: int = 0
    filter_changes: int = 0
    resize_events: int = 0
    errors: list[str] = field(default_factory=list)
    pass_criteria: dict[str, bool] = field(default_factory=dict)
    overall_pass: bool = False
    notes: list[str] = field(default_factory=list)
    startup_max_block_ms: float = 0.0
    stress_max_block_ms: float = 0.0

    def evaluate(self) -> None:
        # Başarı: stress fazında ana thread <100ms (startup AI GIL ayrı not)
        self.anr_detected = self.freeze_count_1s > 0
        stress_ok = self.stress_max_block_ms < 100.0
        self.pass_criteria = {
            "main_thread_block_under_100ms_during_stress": stress_ok,
            "no_anr_freeze_1s": self.freeze_count_1s == 0,
            "ui_stayed_interactive": self.avg_ui_fps >= 10.0 and self.samples > 20,
            "searches_completed": self.searches_finished >= max(
                1, min(5, self.searches_requested // 2)
            ),
        }
        self.overall_pass = all(self.pass_criteria.values())
        if self.startup_max_block_ms >= 100:
            self.notes.append(
                f"Startup/AI-load max block={self.startup_max_block_ms:.0f}ms "
                "(GIL — stress fazından ayrı ölçülür)"
            )
        if self.blocks_over_100ms:
            self.notes.append(
                f"Total gaps>100ms={self.blocks_over_100ms} "
                f"(stress_max={self.stress_max_block_ms:.1f}ms)"
            )

    def print_report(self) -> None:
        w = 64
        print("=" * w)
        print("  UI STRESS TEST & PERFORMANCE BENCHMARK")
        print(f"  {self.timestamp}")
        print("=" * w)
        print(f"  Duration.............. {self.duration_s:.1f}s")
        print(f"  Startup / Stress...... {self.phase_startup_s:.1f}s / {self.phase_stress_s:.1f}s")
        print(f"  Searches.............. {self.searches_finished}/{self.searches_requested}")
        print(f"  Avg / Max search...... {self.avg_search_ms:.0f} / {self.max_search_ms:.0f} ms")
        print("-" * w)
        print(f"  Avg / Min UI FPS...... {self.avg_ui_fps:.1f} / {self.min_ui_fps:.1f}")
        print(f"  Avg / P95 event lag... {self.avg_event_lag_ms:.1f} / {self.p95_event_lag_ms:.1f} ms")
        print(f"  Max block (all)....... {self.max_main_thread_block_ms:.1f} ms")
        print(f"  Max block (startup)... {self.startup_max_block_ms:.1f} ms")
        print(f"  Max block (stress).... {self.stress_max_block_ms:.1f} ms")
        print(f"  Blocks >100ms......... {self.blocks_over_100ms}")
        print(f"  Frame drops (>33ms)... {self.frame_drops}")
        print(f"  Freeze/ANR (>=1s)..... {self.freeze_count_1s}  anr={self.anr_detected}")
        print("-" * w)
        print(f"  Avg / Max CPU......... {self.avg_cpu_percent:.1f}% / {self.max_cpu_percent:.1f}%")
        print(f"  Avg / Max RSS......... {self.avg_rss_mb:.0f} / {self.max_rss_mb:.0f} MB")
        print(f"  Panel / Scroll / Filt. {self.panel_switches} / {self.scroll_steps} / {self.filter_changes}")
        print(f"  Resize events......... {self.resize_events}")
        print("-" * w)
        for k, v in self.pass_criteria.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  OVERALL............... {'PASS' if self.overall_pass else 'FAIL'}")
        for n in self.notes:
            print(f"  note: {n}")
        for e in self.errors[:8]:
            print(f"  error: {e[:140]}")
        print("=" * w)


def _patch_message_boxes() -> None:
    """Stress sırasında modal dialog UI'yi kilitlemesin."""

    def _ok(*_a, **_k):
        return QMessageBox.StandardButton.Yes

    def _exec(self, *_a, **_k):  # noqa: ANN001
        return QMessageBox.StandardButton.Yes

    QMessageBox.question = staticmethod(_ok)  # type: ignore[assignment]
    QMessageBox.information = staticmethod(_ok)  # type: ignore[assignment]
    QMessageBox.warning = staticmethod(_ok)  # type: ignore[assignment]
    QMessageBox.critical = staticmethod(_ok)  # type: ignore[assignment]
    QMessageBox.exec = _exec  # type: ignore[method-assign,assignment]


class EventLoopLagProbe(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._last = time.perf_counter()
        self.gaps_ms: list[float] = []
        self.gap_times: list[float] = []  # absolute time of each sample
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(16)

    def _on_tick(self) -> None:
        now = time.perf_counter()
        gap = (now - self._last) * 1000.0
        self._last = now
        self.gaps_ms.append(gap)
        self.gap_times.append(now)

    def stop(self) -> None:
        self._timer.stop()


class ResourceSampler(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cpu: list[float] = []
        self.rss_mb: list[float] = []
        self._proc = None
        self._prev_cpu = None
        self._prev_t = time.perf_counter()
        self._win_handle = None
        self._psapi = None
        self._pmc_cls = None
        try:
            import psutil

            self._proc = psutil.Process(os.getpid())
            self._prev_cpu = self._proc.cpu_times()
        except Exception:
            self._proc = None
            try:
                import ctypes
                from ctypes import wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                self._pmc_cls = PROCESS_MEMORY_COUNTERS
                self._psapi = ctypes.windll.psapi
                self._win_handle = ctypes.windll.kernel32.GetCurrentProcess()
                self._ctypes = ctypes
            except Exception:
                self._win_handle = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._sample)
        self._timer.start(500)

    def _sample(self) -> None:
        try:
            if self._proc is not None:
                now = time.perf_counter()
                cur = self._proc.cpu_times()
                if self._prev_cpu is not None:
                    dt = max(1e-6, now - self._prev_t)
                    used = (cur.user - self._prev_cpu.user) + (
                        cur.system - self._prev_cpu.system
                    )
                    self.cpu.append(min(800.0, (used / dt) * 100.0))
                self._prev_cpu = cur
                self._prev_t = now
                self.rss_mb.append(float(self._proc.memory_info().rss) / (1024 * 1024))
                return
            if self._win_handle is not None and self._pmc_cls is not None:
                pmc = self._pmc_cls()
                pmc.cb = self._ctypes.sizeof(pmc)
                ok = self._psapi.GetProcessMemoryInfo(
                    self._win_handle, self._ctypes.byref(pmc), pmc.cb
                )
                if ok:
                    self.rss_mb.append(float(pmc.WorkingSetSize) / (1024 * 1024))
        except Exception:
            pass

    def stop(self) -> None:
        self._timer.stop()


class UiStressHarness(QObject):
    def __init__(self, window, *, searches: int = 20, parent=None):
        super().__init__(parent)
        self.win = window
        self.searches = max(1, int(searches))
        self.report = StressReport(searches_requested=self.searches)
        self._t0 = time.perf_counter()
        self._stress_t0: float | None = None
        self._search_started_at: float | None = None
        self._query_i = 0
        self._queries = [
            "leopard", "floral", "stripe", "geometric", "paisley",
            "plaid", "animal", "cicek", "cizgi", "desen",
            "kahverengi", "siyah", "gucci", "lv", "baroque",
            "lace", "dot", "check", "tropical", "marble",
        ]
        self._panel_cycle = [
            "filters", "inspector", "source", "filters", "health", "category", "format"
        ]
        self._panel_i = 0
        self._step = 0
        self._fps_samples: list[float] = []
        self._done = False
        self._stress_active = False

        self.lag = EventLoopLagProbe(self)
        self.res = ResourceSampler(self)
        # MainWindow kendi UiPerfMonitor'unu taşır — onu kullan
        self.perf = getattr(self.win, "_ui_perf", None)
        if self.perf is None:
            self.perf = UiPerfMonitor(log_dir=ROOT / "data" / "logs", parent=self)
        self.perf._freeze_count = 0
        self.perf.touch_heartbeat()
        self.perf.updated.connect(self._on_perf)

        # Suppress filter→auto-search storms during stress
        try:
            self.win.filter_panel.filter_research.disconnect()
            self.win.filter_panel.filter_changed.disconnect()
            self.win.filter_panel.panel_settings_changed.disconnect()
        except Exception:
            pass

        self._driver = QTimer(self)
        self._driver.timeout.connect(self._tick_scenario)
        self._warmup_quiet_since: float | None = None
        self._warmup_deadline = time.perf_counter() + 100.0

        # AI/GIL soğuk yükleme bitsin → sonra stress ölçümü
        QTimer.singleShot(800, self._kick_ai_warmup)
        QTimer.singleShot(2500, self._warmup_poll)
        QTimer.singleShot(max(300_000, 60_000 + self.searches * 5000), self._finish)

    def _kick_ai_warmup(self) -> None:
        """İlk aramada UI'yi kilitlememek için modelleri önceden yükle (worker thread)."""
        import threading

        settings = getattr(self.win, "settings", None)

        def _load() -> None:
            try:
                from core.search_engine import SearchEngine

                eng = SearchEngine(settings, load_ai=False)
                eng.ensure_ai_loaded()
            except Exception as exc:
                self.report.errors.append(f"ai_warmup:{exc}")

        threading.Thread(target=_load, daemon=True, name="ui-stress-ai-warmup").start()
        self.report.notes.append("AI warmup thread started")

    def _on_perf(self, snap) -> None:
        if self._stress_active:
            self._fps_samples.append(float(snap.ui_fps))
        self.report.freeze_count_1s = max(
            self.report.freeze_count_1s, int(snap.freeze_count)
        )

    def _recent_max_gap_ms(self, window_s: float = 1.5) -> float:
        if not self.lag.gaps_ms or not self.lag.gap_times:
            return 9999.0
        cutoff = time.perf_counter() - window_s
        recent = [g for g, t in zip(self.lag.gaps_ms, self.lag.gap_times) if t >= cutoff]
        return max(recent) if recent else 9999.0

    def _warmup_poll(self) -> None:
        if self._done or self._stress_active:
            return
        max_gap = self._recent_max_gap_ms(1.5)
        now = time.perf_counter()
        if max_gap < 80.0:
            if self._warmup_quiet_since is None:
                self._warmup_quiet_since = now
            elif now - self._warmup_quiet_since >= 2.0:
                self.report.notes.append(
                    f"Warmup quiet OK after {now - self._t0:.1f}s (gap<{max_gap:.0f}ms)"
                )
                self._begin_stress()
                return
        else:
            self._warmup_quiet_since = None
        if now >= self._warmup_deadline:
            self.report.notes.append(
                f"Warmup timeout after {now - self._t0:.1f}s (recent_gap={max_gap:.0f}ms)"
            )
            self._begin_stress()
            return
        QTimer.singleShot(400, self._warmup_poll)

    def _begin_stress(self) -> None:
        self._stress_active = True
        self._stress_t0 = time.perf_counter()
        self.report.phase_startup_s = round(self._stress_t0 - self._t0, 2)
        # Warmup false-positive freeze sayacını temizle
        if self.perf is not None:
            self.perf._freeze_count = 0
            self.perf.touch_heartbeat()
        self.report.freeze_count_1s = 0
        # Stress ölçümüne sadece bundan sonraki gap'ler
        self.lag.gaps_ms.clear()
        self.lag.gap_times.clear()
        self.lag._last = time.perf_counter()
        try:
            self.win.filter_panel.slider_threshold.setValue(45)
            self.win.settings.similarity_threshold = 0.45
        except Exception:
            pass
        self._driver.start(70)
        QTimer.singleShot(100, self._start_next_search)

    def _start_next_search(self) -> None:
        if self._done or self._query_i >= self.searches:
            return
        q = self._queries[self._query_i % len(self._queries)]
        self._query_i += 1
        self.report.searches_started += 1
        self._search_started_at = time.perf_counter()
        try:
            self.win.search_header.txt_search.setText(q)
            # Direct worker — skip modal confirmations
            from core.search_models import SearchQuery

            query = SearchQuery(mode="text", text=q, threshold=0.45, fast_only=False)
            self.win._start_search_worker(query)
        except Exception as exc:
            self.report.errors.append(f"search_start:{exc}")
        QTimer.singleShot(120, self._poll_search_done)

    def _poll_search_done(self) -> None:
        if self._done:
            return
        worker = getattr(self.win, "_search_worker", None)
        if worker and worker.isRunning():
            QTimer.singleShot(180, self._poll_search_done)
            return
        if self._search_started_at is not None:
            self.report.search_durations_ms.append(
                (time.perf_counter() - self._search_started_at) * 1000.0
            )
            self.report.searches_finished += 1
            self._search_started_at = None
        if self._query_i < self.searches:
            QTimer.singleShot(200, self._start_next_search)
        else:
            QTimer.singleShot(600, self._finish)

    def _tick_scenario(self) -> None:
        if self._done or not self._stress_active:
            return
        self._step += 1
        app = QApplication.instance()
        try:
            if self._step % 3 == 0:
                key = self._panel_cycle[self._panel_i % len(self._panel_cycle)]
                self._panel_i += 1
                if hasattr(self.win, "_set_panel_visible"):
                    self.win._set_panel_visible(key, True)
                fp = getattr(self.win, "filter_panel", None)
                if fp and hasattr(fp, "tabs") and fp.tabs.count():
                    fp.tabs.setCurrentIndex(self._step % fp.tabs.count())
                self.report.panel_switches += 1

            if self._step % 2 == 0:
                rp = getattr(self.win, "results_panel", None)
                if rp is not None:
                    for name in ("_scroll", "scroll", "list_widget", "_list"):
                        w = getattr(rp, name, None)
                        if w is None:
                            continue
                        bar = getattr(w, "verticalScrollBar", lambda: None)()
                        if bar is not None:
                            bar.setValue(
                                (bar.value() + 110) % max(1, bar.maximum() + 1)
                            )
                            self.report.scroll_steps += 1
                    sc = rp.findChild(QScrollArea) if hasattr(rp, "findChild") else None
                    if sc is not None:
                        bar = sc.verticalScrollBar()
                        bar.setValue((bar.value() + 95) % max(1, bar.maximum() + 1))
                        self.report.scroll_steps += 1

            if self._step % 5 == 0:
                fp = getattr(self.win, "filter_panel", None)
                if fp and hasattr(fp, "slider_threshold"):
                    fp.slider_threshold.blockSignals(True)
                    fp.slider_threshold.setValue(35 + (self._step * 2) % 55)
                    fp.slider_threshold.blockSignals(False)
                    self.report.filter_changes += 1

            if self._step % 6 == 0:
                self.win.resize(QSize(1080 + (self._step * 19) % 420, 680 + (self._step * 11) % 260))
                self.report.resize_events += 1

            if app:
                app.processEvents()
        except Exception as exc:
            self.report.errors.append(f"scenario:{exc}")

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._driver.stop()
        now = time.perf_counter()
        if self._stress_t0:
            self.report.phase_stress_s = round(now - self._stress_t0, 2)
        self.lag.stop()
        self.res.stop()
        try:
            # MainWindow monitor'unu durdurma — thumbnail callback kırılır
            if self.perf is not None and self.perf is not getattr(self.win, "_ui_perf", None):
                self.perf.stop()
        except Exception:
            pass

        gaps = list(self.lag.gaps_ms)
        times = list(self.lag.gap_times)
        if len(gaps) > 8:
            gaps = gaps[4:]
            times = times[4:]
        self.report.samples = len(gaps)
        stress_start = self._stress_t0 or self._t0
        startup_blocks: list[float] = []
        stress_blocks: list[float] = []
        if gaps:
            ordered = sorted(gaps)
            self.report.max_event_lag_ms = round(max(gaps), 2)
            self.report.avg_event_lag_ms = round(sum(gaps) / len(gaps), 2)
            self.report.p95_event_lag_ms = round(ordered[int(len(ordered) * 0.95)], 2)
            self.report.max_main_thread_block_ms = round(max(0.0, max(gaps) - 16.0), 2)
            self.report.blocks_over_100ms = sum(1 for g in gaps if g > 100.0)
            self.report.frame_drops = sum(1 for g in gaps if g > 33.0)
            for g, t in zip(gaps, times):
                block = max(0.0, g - 16.0)
                if t < stress_start:
                    startup_blocks.append(block)
                else:
                    stress_blocks.append(block)
            self.report.startup_max_block_ms = round(max(startup_blocks or [0.0]), 2)
            self.report.stress_max_block_ms = round(max(stress_blocks or [0.0]), 2)

        if self._fps_samples:
            self.report.avg_ui_fps = round(
                sum(self._fps_samples) / len(self._fps_samples), 1
            )
            self.report.min_ui_fps = round(min(self._fps_samples), 1)

        snap = self.perf.snapshot()
        self.report.freeze_count_1s = max(
            self.report.freeze_count_1s, int(snap.freeze_count)
        )
        if self.res.cpu:
            self.report.avg_cpu_percent = round(sum(self.res.cpu) / len(self.res.cpu), 1)
            self.report.max_cpu_percent = round(max(self.res.cpu), 1)
        if self.res.rss_mb:
            self.report.avg_rss_mb = round(sum(self.res.rss_mb) / len(self.res.rss_mb), 1)
            self.report.max_rss_mb = round(max(self.res.rss_mb), 1)
        durs = self.report.search_durations_ms
        if durs:
            self.report.avg_search_ms = round(sum(durs) / len(durs), 1)
            self.report.max_search_ms = round(max(durs), 1)

        self.report.duration_s = round(now - self._t0, 2)
        self.report.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report.evaluate()
        try:
            self.report.print_report()
        except Exception as exc:
            print("print_report failed:", exc)
        try:
            out = ROOT / "data" / "reports" / "ui_stress_benchmark.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(asdict(self.report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("Saved:", out)
        except Exception as exc:
            print("save failed:", exc)
        QTimer.singleShot(50, QApplication.instance().quit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--searches", type=int, default=20)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--offscreen", action="store_true")
    args = parser.parse_args()
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    _patch_message_boxes()
    app = QApplication.instance() or QApplication(sys.argv)

    # Startup loadunu hafiflet (UI stress ölçümüne odak)
    settings = AppSettings.load()
    settings.resume_index_on_startup = False
    settings.auto_scan_on_startup = False
    # Settings dosyasını kalıcı yazma — sadece bu process
    AppSettings.load = staticmethod(lambda: settings)  # type: ignore[method-assign]

    from ui.main_window import MainWindow

    win = MainWindow()
    win._resume_index_suppressed = True
    win.resize(1280, 860)
    win.show()
    app.processEvents()

    harness = UiStressHarness(win, searches=args.searches)
    app._stress_harness = harness  # type: ignore[attr-defined]
    app._stress_win = win  # type: ignore[attr-defined]
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
