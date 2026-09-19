"""Performans benchmark — tek tuş rapor."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from core.db import Database
from core.logger import setup_logger

logger = setup_logger(__name__)


def run_performance_benchmark(settings) -> dict[str, Any]:
    db = Database(settings.db_path)
    dash = db.count_index_pipeline_dashboard()
    total = max(1, int(dash.get("total_searchable", 0) or 1))

    report: dict[str, Any] = {"ts": time.time(), "metrics": {}}

    # SQLite yazma hızı
    t0 = time.perf_counter()
    writes = 0
    try:
        conn = sqlite3.connect(settings.db_path, timeout=5)
        for i in range(200):
            conn.execute("SELECT 1")
            writes += 1
        conn.close()
    except Exception as exc:
        report["metrics"]["sqlite_writes_per_sec"] = {"error": str(exc)}
    else:
        dt = max(0.001, time.perf_counter() - t0)
        report["metrics"]["sqlite_reads_per_sec"] = round(writes / dt, 1)

    # Index pipeline oranları (mevcut arşivden türetilmiş)
    preview_ready = int(dash.get("preview_ready", 0) or 0)
    ocr_ready = int(dash.get("ocr_ready", 0) or 0)
    emb_ready = int(dash.get("embedding_ready", 0) or 0)
    dna_ready = int(dash.get("pattern_dna_ready", 0) or 0)
    sem_ready = int(dash.get("semantic_tag_ready", 0) or 0)

    report["metrics"]["thumbnail_coverage_pct"] = round(100 * preview_ready / total, 1)
    report["metrics"]["ocr_coverage_pct"] = round(100 * ocr_ready / total, 1)
    report["metrics"]["embedding_coverage_pct"] = round(100 * emb_ready / total, 1)
    report["metrics"]["pattern_dna_coverage_pct"] = round(100 * dna_ready / total, 1)
    report["metrics"]["semantic_coverage_pct"] = round(100 * sem_ready / total, 1)

    # Cache/NAS IO proxy
    cache_dir = Path(settings.cache_dir)
    cache_mb = 0.0
    if cache_dir.exists():
        try:
            cache_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        except OSError:
            pass
    report["metrics"]["cache_mb"] = round(cache_mb, 1)
    report["metrics"]["index_total"] = total
    report["dashboard"] = dash

    logger.info("Benchmark: %s", report["metrics"])
    return report
