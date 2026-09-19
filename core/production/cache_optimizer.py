"""RAM dolunca eski cache girdilerini LRU ile temizle."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from core.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_MAX_CACHE_MB = 4096
_RAM_PRESSURE_PCT = 82


def _list_cache_files(cache_dir: Path) -> list[tuple[Path, float, int]]:
    items: list[tuple[Path, float, int]] = []
    if not cache_dir.exists():
        return items
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            p = Path(root) / name
            try:
                st = p.stat()
                items.append((p, st.st_mtime, st.st_size))
            except OSError:
                pass
    return items


def optimize_cache(
    settings,
    *,
    max_cache_mb: float | None = None,
    ram_pressure_pct: float = _RAM_PRESSURE_PCT,
) -> dict[str, Any]:
    cache_dir = Path(settings.cache_dir)
    limit_mb = float(
        max_cache_mb
        or getattr(settings, "cache_max_mb", 0)
        or _DEFAULT_MAX_CACHE_MB
    )

    ram_high = False
    try:
        import psutil

        ram_high = psutil.virtual_memory().percent >= ram_pressure_pct
    except Exception:
        pass

    files = _list_cache_files(cache_dir)
    total_mb = sum(s for _p, _t, s in files) / (1024 * 1024)
    if total_mb <= limit_mb and not ram_high:
        return {"removed": 0, "freed_mb": 0.0, "total_mb": round(total_mb, 1), "skipped": True}

    target_mb = limit_mb * 0.75 if ram_high else limit_mb * 0.9
    files.sort(key=lambda x: x[1])  # oldest first
    removed = 0
    freed = 0
    for path, _mtime, size in files:
        if total_mb - (freed / (1024 * 1024)) <= target_mb:
            break
        try:
            path.unlink(missing_ok=True)
            removed += 1
            freed += size
        except OSError:
            pass

    result = {
        "removed": removed,
        "freed_mb": round(freed / (1024 * 1024), 2),
        "total_mb_before": round(total_mb, 1),
        "ram_pressure": ram_high,
        "ts": time.time(),
    }
    logger.info("Cache optimize: %s", result)
    return result
