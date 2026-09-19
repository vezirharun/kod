"""Program açılışında otomatik kurtarma — yarım kalan index."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.db import Database
from core.index_queue_manager import IndexQueueManager
from core.logger import setup_logger

logger = setup_logger(__name__)


def _recovery_path(settings) -> Path:
    return Path(settings.db_path).parent / "recovery.json"


def run_startup_recovery(settings) -> dict[str, Any]:
    """Elektrik kesintisi / çökme sonrası kuyruk ve işlem durumunu sıfırla."""
    if not getattr(settings, "production_recovery_enabled", True):
        return {"skipped": True}

    db = Database(settings.db_path)
    queue = IndexQueueManager(db)
    result: dict[str, Any] = {
        "ts": time.time(),
        "queue_reset": queue.reset_stale_running(),
    }

    if hasattr(db, "purge_broken_index_records"):
        try:
            result["purge"] = db.purge_broken_index_records()
        except Exception as exc:
            result["purge_error"] = str(exc)

    from core.index_ssot import count_lanes, reset_stale_processing

    result["ssot_reset"] = reset_stale_processing(db)
    lanes = count_lanes(db)
    pending = int(lanes["light_queue_display"] + lanes["heavy_ready_pending"])
    incomplete = len(queue.sources_with_incomplete_work())
    result["pending_queue"] = pending
    result["incomplete_sources"] = incomplete

    path = _recovery_path(settings)
    try:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        result["write_error"] = str(exc)

    logger.info("Startup recovery: %s", result)
    return result
