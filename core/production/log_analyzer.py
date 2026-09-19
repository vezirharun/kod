"""Log analizi — son 24 saat özet."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.logger import setup_logger

logger = setup_logger(__name__)

_PATTERNS = {
    "ocr_errors": re.compile(r"ocr|tesseract", re.I),
    "sqlite_locks": re.compile(r"sqlite.*lock|database is locked", re.I),
    "handle_leaks": re.compile(r"handle.?leak|num_handles", re.I),
    "crashes": re.compile(r"critical|yakalanmamış|traceback|crash", re.I),
}


def _parse_ts(obj: dict[str, Any]) -> float | None:
    ts = obj.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    for key in ("time", "timestamp"):
        v = obj.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def analyze_logs_24h(settings) -> dict[str, Any]:
    log_dir = Path(settings.db_path).parent / "logs"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    cutoff_ts = cutoff.timestamp()

    counts = {k: 0 for k in _PATTERNS}
    cpu_samples: list[float] = []
    ram_samples: list[float] = []

    for path in sorted(log_dir.glob("*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        for name, pat in _PATTERNS.items():
                            if pat.search(line):
                                counts[name] += 1
                        continue
                    ts = _parse_ts(obj)
                    if ts is not None and ts < cutoff_ts:
                        continue
                    if "cpu_percent" in obj or "system_cpu_percent" in obj:
                        cpu_samples.append(
                            float(obj.get("system_cpu_percent") or obj.get("cpu_percent") or 0)
                        )
                    if "rss_mb" in obj:
                        ram_samples.append(float(obj["rss_mb"]))
                    if obj.get("sqlite", {}).get("locked"):
                        counts["sqlite_locks"] += 1
                    handles = obj.get("handles") or obj.get("num_handles")
                    if handles and int(handles) > 8000:
                        counts["handle_leaks"] += 1
        except OSError:
            continue

    for path in sorted(log_dir.glob("*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name, pat in _PATTERNS.items():
                counts[name] += len(pat.findall(text))
        except OSError:
            continue

    return {
        "period_hours": 24,
        "ocr_errors": counts["ocr_errors"],
        "sqlite_locks": counts["sqlite_locks"],
        "handle_leaks": counts["handle_leaks"],
        "crashes": counts["crashes"],
        "avg_cpu_percent": round(sum(cpu_samples) / len(cpu_samples), 1) if cpu_samples else None,
        "avg_ram_mb": round(sum(ram_samples) / len(ram_samples), 1) if ram_samples else None,
        "samples": len(cpu_samples),
    }
