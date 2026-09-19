"""Merkezi loglama ve hata raporlama."""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.settings import DEFAULT_DATA_DIR

LOG_DIR = DEFAULT_DATA_DIR / "logs"
APP_LOG = LOG_DIR / "app.log"
ERROR_CSV = LOG_DIR / "error_files.csv"
SKIPPED_CSV = LOG_DIR / "skipped_files.csv"
INDEX_REPORT = LOG_DIR / "index_report.json"

_logger: logging.Logger | None = None


def setup_logger(name: str = "vezir") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        fh = logging.FileHandler(APP_LOG, encoding="utf-8")
    except OSError:
        # Windows can temporarily lock app.log while another app instance is
        # shutting down. Logging must never prevent the application starting.
        fallback = Path(tempfile.gettempdir()) / "vezir_pattern_search.log"
        try:
            fh = logging.FileHandler(fallback, encoding="utf-8")
        except OSError:
            fh = None
    if fh is not None:
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    _logger = logger
    return logger


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    target = path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_header = not target.exists()
        with open(target, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    except OSError:
        # Diagnostics are best-effort; a locked report must not stop indexing.
        setup_logger().warning("Tanı CSV dosyası yazılamadı: %s", target)


def log_error_file(file_path: str, error: str, stage: str = "") -> None:
    _append_csv(
        ERROR_CSV,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_path": file_path,
            "stage": stage,
            "error": error,
        },
        ["timestamp", "file_path", "stage", "error"],
    )
    setup_logger().error("Dosya hatası [%s] %s: %s", stage, file_path, error)


def log_skipped_file(file_path: str, reason: str) -> None:
    _append_csv(
        SKIPPED_CSV,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_path": file_path,
            "reason": reason,
        },
        ["timestamp", "file_path", "reason"],
    )
    setup_logger().info("Atlandı: %s — %s", file_path, reason)


def write_index_report(report: dict[str, Any]) -> None:
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    targets = [INDEX_REPORT, Path(tempfile.gettempdir()) / "vezir_index_report.json"]
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return
        except OSError:
            continue
    setup_logger().warning("İndeks raporu yazılamadı")


def write_source_scan_report(report: dict[str, Any]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    targets = [
        DEFAULT_DATA_DIR / "reports" / f"source_scan_{stamp}.json",
        Path(tempfile.gettempdir())
        / "vezir_pattern_search_reports"
        / f"source_scan_{stamp}.json",
    ]
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return target
        except OSError:
            continue
    setup_logger().warning("Kaynak tarama raporu yazılamadı")
    return targets[-1]
