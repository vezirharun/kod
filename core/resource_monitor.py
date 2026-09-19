"""Sistem kaynak diagnostik — 5 sn aralıkla log (P1 darboğaz tespiti)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from core.logger import setup_logger

logger = setup_logger(__name__)

_CREATIVE_PROCESSES = (
    "photoshop.exe",
    "illustrator.exe",
    "coreldr.exe",
    "coreldraw.exe",
    "chrome.exe",
    "msedge.exe",
    "cursor.exe",
    "code.exe",
)


class ResourceMonitor:
    """Index sırasında CPU/RAM/handle/thread/kuyruk raporu üretir."""

    def __init__(
        self,
        *,
        interval_sec: float = 5.0,
        log_dir: str | Path | None = None,
        queue_snapshot: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.interval_sec = max(2.0, float(interval_sec))
        self.log_path = Path(log_dir or "data/logs") / "resource_monitor.jsonl"
        self.queue_snapshot = queue_snapshot
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._extra: dict[str, Any] = {}

    def set_extra(self, **kwargs: Any) -> None:
        self._extra.update(kwargs)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._loop,
            name="vezir-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("Resource monitor started → %s", self.log_path)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                report = self.sample()
                line = json.dumps(report, ensure_ascii=False)
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                logger.warning(
                    "[RESOURCE] cpu=%.0f%% ram=%.0fMB handles=%s threads=%s "
                    "io_r=%.1fMB/s io_w=%.1fMB/s creative=%s queue=%s workers=%s",
                    report.get("cpu_percent", 0),
                    report.get("rss_mb", 0),
                    report.get("num_handles", "?"),
                    report.get("num_threads", "?"),
                    report.get("disk_read_mb_s", 0),
                    report.get("disk_write_mb_s", 0),
                    report.get("creative_apps", []),
                    report.get("queue", {}),
                    (report.get("queue") or {}).get(
                        "workers", report.get("extra", {}).get("workers", "?")
                    ),
                )
            except Exception as exc:
                logger.debug("Resource monitor sample failed: %s", exc)

    def sample(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "ts": time.time(),
            "pid": os.getpid(),
            "extra": dict(self._extra),
        }
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            report["cpu_percent"] = float(proc.cpu_percent(interval=0.05))
            mem = proc.memory_info()
            report["rss_mb"] = round(mem.rss / (1024 * 1024), 1)
            report["num_threads"] = int(proc.num_threads())
            try:
                report["num_handles"] = int(proc.num_handles())
            except (AttributeError, psutil.Error):
                report["num_handles"] = None
            try:
                io1 = proc.io_counters()
                time.sleep(0.15)
                io2 = proc.io_counters()
                dt = 0.15
                report["disk_read_mb_s"] = round(
                    (io2.read_bytes - io1.read_bytes) / (1024 * 1024) / dt, 2
                )
                report["disk_write_mb_s"] = round(
                    (io2.write_bytes - io1.write_bytes) / (1024 * 1024) / dt, 2
                )
            except (AttributeError, psutil.Error):
                report["disk_read_mb_s"] = 0.0
                report["disk_write_mb_s"] = 0.0
            try:
                vm = psutil.virtual_memory()
                report["system_ram_percent"] = float(vm.percent)
                report["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
            except psutil.Error:
                pass
            creative = []
            for p in psutil.process_iter(["name"]):
                name = (p.info.get("name") or "").lower()
                if name in _CREATIVE_PROCESSES:
                    creative.append(name)
            report["creative_apps"] = sorted(set(creative))
        except Exception as exc:
            report["error"] = str(exc)
            report["num_threads"] = threading.active_count()

        # GPU — opsiyonel (nvidia-smi yoksa atla)
        report["gpu_util_percent"] = _sample_gpu_util()

        if self.queue_snapshot:
            try:
                report["queue"] = self.queue_snapshot() or {}
            except Exception:
                report["queue"] = {}
        return report


_monitor: ResourceMonitor | None = None
_monitor_lock = threading.Lock()
_last_io: tuple[float, int, int] | None = None


def _sample_gpu_util() -> float | None:
    try:
        import shutil
        import subprocess

        if not shutil.which("nvidia-smi"):
            return None
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=1.5,
            stderr=subprocess.DEVNULL,
        )
        first = out.decode("utf-8", errors="ignore").strip().splitlines()[0]
        return float(first.strip())
    except Exception:
        return None


def start_resource_monitor(
    *,
    log_dir: str | Path | None = None,
    queue_snapshot: Callable[[], dict[str, Any]] | None = None,
    interval_sec: float = 5.0,
) -> ResourceMonitor:
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = ResourceMonitor(
                interval_sec=interval_sec,
                log_dir=log_dir,
                queue_snapshot=queue_snapshot,
            )
        _monitor.start()
        return _monitor


def stop_resource_monitor() -> None:
    global _monitor
    with _monitor_lock:
        if _monitor is not None:
            _monitor.stop()


def get_resource_monitor() -> ResourceMonitor | None:
    return _monitor
