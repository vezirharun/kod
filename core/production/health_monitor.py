"""Sistem sağlığı — 5 sn aralıkla CPU/RAM/GPU/I/O/kuyruk/FAISS/SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from core.logger import setup_logger
from core.resource_monitor import ResourceMonitor, start_resource_monitor

logger = setup_logger(__name__)

_lock = threading.Lock()
_last_snapshot: dict[str, Any] = {}
_health_thread: threading.Thread | None = None
_health_stop = threading.Event()


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        return 0.0
    return round(total / (1024 * 1024), 1)


def _sqlite_status(db_path: str) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "wal": False, "locked": False}
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        mode = conn.execute("PRAGMA journal_mode").fetchone()
        out["wal"] = bool(mode and str(mode[0]).lower() == "wal")
        conn.execute("SELECT 1")
        out["ok"] = True
        conn.close()
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            out["locked"] = True
        out["error"] = str(exc)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _faiss_status(settings) -> dict[str, Any]:
    from core.faiss_store import FaissStore

    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    dino = Path(settings.faiss_dino_path)
    clip = Path(settings.faiss_clip_path)
    return {
        "available": store.available,
        "dino_exists": dino.exists(),
        "clip_exists": clip.exists(),
        "dino_mb": round(dino.stat().st_size / (1024 * 1024), 2) if dino.exists() else 0,
        "clip_mb": round(clip.stat().st_size / (1024 * 1024), 2) if clip.exists() else 0,
    }


def queues_from_status(status: dict[str, Any] | None) -> dict[str, int]:
    """Health/RESOURCE kuyruk görünümü.

    ``index_active`` = taze claim (gerçek çalışan iş).
    Eski mapping ``queue_pending`` = Genel AI kalan dosya (total - ai_final
    - failed_permanent) idi; çalışan worker sayısı değildi.
    """
    s = dict(status or {})
    claimed = int(s.get("claimed_jobs") or s.get("processing") or 0)
    light_jobs = s.get("pending_light_jobs")
    if light_jobs is None:
        light_jobs = 0
    return {
        "index_active": claimed,
        "SOURCE_TOTAL": int(s.get("total") or s.get("archive_total") or 0),
        "LIGHT_PENDING": int(light_jobs),
        "LIGHT_PROCESSING": claimed,
        "LIGHT_DONE": int(s.get("light_done") or s.get("fast_completed") or 0),
        "LIGHT_FAILED": int(
            s.get("light_failed") or s.get("failed_permanent_files") or 0
        ),
        "GENERAL_AI_PENDING": int(
            s.get("general_remaining") or s.get("queue_pending") or 0
        ),
        "GENERAL_AI_PROCESSING": claimed,
        "GENERAL_AI_COMPLETED": int(
            s.get("general_completed") or s.get("ai_final_ready") or 0
        ),
        "INDEX_ACTIVE": claimed,
        "pending_full": int(s.get("pending_heavy_jobs") or 0),
        "ocr_pending": int(s.get("pending_ocr") or 0),
        "embedding_pending": int(s.get("pending_embedding") or 0),
        "thumbnail_pending": int(
            s.get("pending_light_jobs") or s.get("pending_preview") or 0
        ),
        "workers": claimed,
    }


def collect_health_snapshot(settings, *, use_cache_only: bool = False) -> dict[str, Any]:
    """Sağlık raporu — mümkünse RAM/status cache; Search ile SQLite yarışmaz."""
    global _last_snapshot
    from core.app_status import get_cached_status
    from core.resource_monitor import ResourceMonitor as RM

    cached = get_last_health_snapshot()
    status_ram = get_cached_status()
    if use_cache_only and cached:
        log_dir = Path(settings.db_path).parent / "logs"
        base = RM(log_dir=log_dir).sample()
        snap = {**cached, **base}
        snap["ts"] = time.time()
        snap["system_cpu_percent"] = base.get(
            "system_cpu_percent", base.get("cpu_percent", 0)
        )
        snap["system_ram_percent"] = base.get("system_ram_percent", 0)
        snap["threads"] = int(base.get("num_threads", 0) or 0)
        snap["handles"] = base.get("num_handles")
        snap["disk_read_mb_s"] = base.get("disk_read_mb_s", 0)
        snap["disk_write_mb_s"] = base.get("disk_write_mb_s", 0)
        if status_ram:
            snap["queues"] = queues_from_status(status_ram)
        snap["from_snapshot_cache"] = True
        with _lock:
            _last_snapshot = snap
        return snap

    log_dir = Path(settings.db_path).parent / "logs"
    base = RM(log_dir=log_dir).sample()
    cache_dir = Path(
        getattr(settings, "cache_dir", "")
        or Path(settings.db_path).parent.parent / "cache"
    )
    thumb_dir = cache_dir / "thumbnails"

    # Dashboard / kuyruk: önce StatusWorker RAM cache
    dash: dict[str, Any] = {}
    if status_ram:
        dash = dict(status_ram)
    else:
        try:
            from core.db import Database

            from core.index_ssot import count_lanes

            db = Database(settings.db_path)
            dash = db.count_index_pipeline_dashboard()
            lanes = count_lanes(db)
            queue_active = int(
                lanes["light_queue_display"] + lanes["heavy_ready_pending"]
            )
            pending_full = int(lanes["heavy_ready_pending"])
            dash = {**dash, **lanes}
        except Exception as exc:
            logger.debug("health db fallback skipped: %s", exc)
            dash = {}

    snap = {
        "ts": time.time(),
        **base,
        "system_cpu_percent": base.get("system_cpu_percent", base.get("cpu_percent", 0)),
        "system_ram_percent": base.get("system_ram_percent", 0),
        "gpu_util_percent": base.get("gpu_util_percent"),
        "nas_io_note": "network I/O via indexer throttle",
        "sqlite": _sqlite_status(settings.db_path),
        "faiss": _faiss_status(settings),
        "queues": queues_from_status(dash),
        "pipeline": dash,
        "cache_mb": {
            "total": _dir_size_mb(cache_dir),
            "thumbnails": _dir_size_mb(thumb_dir),
        },
        "threads": int(base.get("num_threads", 0) or 0),
        "handles": base.get("num_handles"),
        "disk_read_mb_s": base.get("disk_read_mb_s", 0),
        "disk_write_mb_s": base.get("disk_write_mb_s", 0),
        "from_snapshot_cache": False,
    }
    with _lock:
        _last_snapshot = snap
    return snap


def get_last_health_snapshot() -> dict[str, Any]:
    with _lock:
        return dict(_last_snapshot)


def _health_loop(
    settings,
    log_path: Path,
    interval_sec: float,
    on_snapshot: Callable[[dict[str, Any]], None] | None,
) -> None:
    while not _health_stop.wait(interval_sec):
        try:
            # Search sırasında snapshot cache + status RAM; ağır SQLite yok
            snap = collect_health_snapshot(settings, use_cache_only=bool(_last_snapshot))
            if on_snapshot:
                on_snapshot(snap)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("health loop: %s", exc)


def start_health_monitor(
    settings,
    *,
    interval_sec: float = 5.0,
    on_snapshot: Callable[[dict[str, Any]], None] | None = None,
) -> ResourceMonitor:
    """5 sn aralıkla health snapshot + jsonl log."""
    global _health_thread
    log_dir = Path(settings.db_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    health_log = log_dir / "health_monitor.jsonl"

    # İlk snapshot (cache doldur)
    try:
        collect_health_snapshot(settings, use_cache_only=False)
    except Exception:
        pass

    mon = start_resource_monitor(
        log_dir=log_dir,
        queue_snapshot=lambda: (
            get_last_health_snapshot().get("queues")
            or collect_health_snapshot(settings, use_cache_only=True).get("queues", {})
        ),
        interval_sec=interval_sec,
    )

    _health_stop.clear()
    if _health_thread is None or not _health_thread.is_alive():
        _health_thread = threading.Thread(
            target=_health_loop,
            args=(settings, health_log, interval_sec, on_snapshot),
            name="vezir-health-monitor",
            daemon=True,
        )
        _health_thread.start()
    return mon


def stop_health_monitor() -> None:
    _health_stop.set()
