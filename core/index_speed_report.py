"""Index hız raporu — light/full süre ve throughput."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.settings import DEFAULT_DATA_DIR


@dataclass
class IndexSpeedSession:
    mode: str = "standard"
    started_at: float = field(default_factory=time.perf_counter)
    light_started_at: float = 0.0
    light_finished_at: float = 0.0
    full_started_at: float = 0.0
    full_finished_at: float = 0.0
    total_files: int = 0
    light_processed: int = 0
    full_processed: int = 0
    thumbnails: int = 0
    hashes: int = 0
    errors: int = 0
    skipped_heavy_format: int = 0
    pending_full_queue: int = 0
    preview_generated: int = 0
    ai_from_cache: int = 0
    original_reopened: int = 0
    report_path: str = ""

    def begin_light(self) -> None:
        self.light_started_at = time.perf_counter()

    def end_light(self) -> None:
        self.light_finished_at = time.perf_counter()

    def begin_full(self) -> None:
        self.full_started_at = time.perf_counter()

    def end_full(self) -> None:
        self.full_finished_at = time.perf_counter()

    def light_seconds(self) -> float:
        if not self.light_started_at:
            return 0.0
        end = self.light_finished_at or time.perf_counter()
        return max(0.0, end - self.light_started_at)

    def full_seconds(self) -> float:
        if not self.full_started_at:
            return 0.0
        end = self.full_finished_at or time.perf_counter()
        return max(0.0, end - self.full_started_at)

    def to_dict(self) -> dict[str, Any]:
        light_s = self.light_seconds()
        full_s = self.full_seconds()
        # Yalnizca basariyla islenen dosyalar (hatalar haric) hiz payi olur.
        proc = max(0, self.light_processed + self.full_processed)
        denom = light_s + full_s
        thumb_s = self.thumbnails / light_s if light_s > 0 else 0.0
        hash_s = self.hashes / light_s if light_s > 0 else 0.0
        files_s = proc / denom if denom > 0 and proc > 0 else 0.0
        return {
            "mode": self.mode,
            "total_files": self.total_files,
            "light_processed": self.light_processed,
            "full_processed": self.full_processed,
            "light_index_seconds": round(light_s, 3),
            "full_index_seconds": round(full_s, 3),
            "files_per_second": round(files_s, 3),
            "thumbnails_per_second": round(thumb_s, 3),
            "hashes_per_second": round(hash_s, 3),
            "errors": self.errors,
            "skipped_heavy_format": self.skipped_heavy_format,
            "pending_full_queue": self.pending_full_queue,
            "preview_generated": self.preview_generated,
            "ai_from_cache": self.ai_from_cache,
            "original_reopened": self.original_reopened,
            "report_path": self.report_path,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def write_index_speed_report(
    session: IndexSpeedSession,
    output_dir: str | Path | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}"
    targets = []
    if output_dir:
        targets.append(Path(output_dir) / f"index_speed_report_{stamp}.json")
    targets.extend(
        [
            DEFAULT_DATA_DIR / "reports" / f"index_speed_report_{stamp}.json",
            Path.cwd() / "reports" / f"index_speed_report_{stamp}.json",
        ]
    )
    payload = session.to_dict()
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            session.report_path = str(target)
            payload["report_path"] = session.report_path
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return target
        except OSError:
            continue
    return targets[0]
