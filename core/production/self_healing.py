"""Otomatik onarım — yalnızca eksik/bozuk parçalar."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from core.db import Database
from core.index_queue_manager import IndexQueueManager
from core.logger import setup_logger
from core.production.index_verify import repair_index, verify_index

logger = setup_logger(__name__)

_HANDLE_LEAK_THRESHOLD = 8000
_THREAD_STALL_THRESHOLD = 120


def _read_recent_health(log_dir: Path, limit: int = 30) -> list[dict[str, Any]]:
    path = log_dir / "health_monitor.jsonl"
    if not path.exists():
        path = log_dir / "resource_monitor.jsonl"
    if not path.exists():
        return []
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def detect_issues(settings) -> dict[str, Any]:
    """Mevcut durumda tespit edilen sorunlar."""
    db = Database(settings.db_path)
    log_dir = Path(settings.db_path).parent / "logs"
    issues: list[str] = []
    details: dict[str, Any] = {}

    dash = db.count_index_pipeline_dashboard()
    verify = verify_index(settings, sample_limit=2000)

    if int(dash.get("pending_preview", 0) or 0) > 0:
        issues.append("missing_thumbnail")
    if int(dash.get("pending_embedding", 0) or 0) > 0:
        issues.append("missing_embedding")
    if int(dash.get("pending_ocr", 0) or 0) > 0:
        issues.append("missing_ocr")
    if int(dash.get("pending_pattern_dna", 0) or 0) > 0:
        issues.append("missing_pattern_dna")
    if int(dash.get("pending_semantic_tag", 0) or 0) > 0:
        issues.append("missing_semantic")

    # SQLite lock
    try:
        import sqlite3

        conn = sqlite3.connect(settings.db_path, timeout=1)
        conn.execute("SELECT 1")
        conn.close()
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            issues.append("sqlite_lock")
            details["sqlite"] = str(exc)

    samples = _read_recent_health(log_dir)
    if samples:
        handles = [s.get("handles") or s.get("num_handles") for s in samples if s.get("handles") or s.get("num_handles")]
        if handles and max(int(h) for h in handles if h) > _HANDLE_LEAK_THRESHOLD:
            issues.append("handle_leak")
            details["max_handles"] = max(int(h) for h in handles if h)

        threads = [s.get("threads") or s.get("num_threads") for s in samples[-5:]]
        if threads and all(int(t or 0) > 80 for t in threads):
            issues.append("thread_pressure")
            details["threads"] = threads[-1]

        from core.index_ssot import count_lanes

        lanes = count_lanes(db)
        queue_active = int(
            lanes["light_queue_display"] + lanes["heavy_ready_pending"]
        )
        proc_stuck = int(lanes["light_processing"] + lanes["heavy_processing"])
        if proc_stuck > 0:
            issues.append("stale_processing_files")
            details["processing_stuck"] = proc_stuck
            details["ssot_queue"] = queue_active

    for key, count in (verify.get("issues") or {}).items():
        if int(count or 0) > 0 and key == "thumbnail":
            if "broken_thumbnail" not in issues:
                issues.append("broken_thumbnail")

    details["verify"] = verify.get("issues")
    details["dashboard"] = dash
    return {"issues": issues, "details": details}


def run_self_healing(settings) -> dict[str, Any]:
    """Tespit edilen sorunları hedefli onar."""
    from core.index_maintenance import purge_broken_index_records

    detected = detect_issues(settings)
    actions: dict[str, Any] = {"detected": detected["issues"], "steps": []}
    db = Database(settings.db_path)
    queue = IndexQueueManager(db)

    if "sqlite_lock" in detected["issues"]:
        actions["steps"].append({"sqlite": "wait_retry", "note": "kilit geçici olabilir"})

    if "stale_processing_files" in detected["issues"] or "thread_pressure" in detected["issues"]:
        from core.index_ssot import reset_stale_processing

        reset_q = queue.reset_stale_running()
        ssot_reset = reset_stale_processing(db)
        purge = purge_broken_index_records(settings)
        actions["steps"].append(
            {"queue_reset": reset_q, "ssot_reset": ssot_reset, "purge": purge}
        )

    repair_targets = []
    issue_map = {
        "missing_thumbnail": "thumbnail",
        "broken_thumbnail": "thumbnail",
        "missing_embedding": "embedding",
        "missing_ocr": "ocr",
        "missing_pattern_dna": "pattern_dna",
        "missing_semantic": "semantic",
    }
    for iss in detected["issues"]:
        t = issue_map.get(iss)
        if t and t not in repair_targets:
            repair_targets.append(t)

    if repair_targets:
        actions["steps"].append(
            repair_index(settings, only=repair_targets)
        )

    actions["steps"].append({"cache": _trim_if_ram_high(settings)})
    logger.info("Self-healing tamamlandı: %s", actions)
    return actions


def _trim_if_ram_high(settings) -> dict[str, Any]:
    from core.production.cache_optimizer import optimize_cache

    try:
        import psutil

        if psutil.virtual_memory().percent >= 85:
            return optimize_cache(settings)
    except Exception:
        pass
    return {"skipped": True, "reason": "ram_ok"}
