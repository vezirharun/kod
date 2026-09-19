"""FAST recovery for sources 8/9 — Global Preview on stuck light gaps."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.ui_bridge import count_v3_ssot
from core.settings import AppSettings
from core.sources import SourceManager


def main() -> None:
    settings = AppSettings.load()
    db = Database(settings.db_path)
    job_db = Path(settings.db_path).with_name("patterns.v3jobs.db")
    sm = SourceManager(settings)
    sources: list[dict] = []
    for sid in (8, 9):
        for s in sm.list_sources(active_only=False):
            if int(s.get("id") or 0) == sid:
                sources.append(
                    {
                        "id": sid,
                        "root_path": str(s.get("root_path") or ""),
                    }
                )
                break
    if len(sources) != 2:
        raise SystemExit(f"expected sources 8+9, got {sources!r}")

    log_dir = ROOT / "data" / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fast_recovery_89.log"
    source_ids = [8, 9]

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    baseline = count_v3_ssot(db, source_ids)
    log(f"BASELINE {json.dumps(baseline, ensure_ascii=False)}")

    last_tick = 0.0

    def progress_cb(*_args, **_kwargs) -> None:
        nonlocal last_tick
        now = time.monotonic()
        if now - last_tick < 30.0:
            return
        last_tick = now
        c = count_v3_ssot(db, source_ids)
        gap = int(c.get("total", 0)) - int(c.get("preview", 0))
        log(
            f"PROGRESS preview={c.get('preview')} thumb={c.get('thumbnail')} "
            f"hash={c.get('hash')} gap={gap} light_q={c.get('light_queue')}"
        )

    eng = IndexEngineV3(
        db,
        job_db_path=job_db,
        settings=settings,
        use_real_extractors=True,
    )
    t0 = time.perf_counter()
    log("FAST start sources=8,9 walk_disk=False")
    report = eng.run(
        mode=Mode.FAST,
        sources=sources,
        walk_disk=False,
        progress_callback=progress_cb,
    )
    elapsed = time.perf_counter() - t0
    after = count_v3_ssot(db, source_ids)
    gap = int(after.get("total", 0)) - int(after.get("preview", 0))
    log(f"DONE elapsed={elapsed:.1f}s gap={gap} worker={json.dumps(report.worker)}")
    log(f"AFTER {json.dumps(after, ensure_ascii=False)}")

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT preview_status, COUNT(*) n
            FROM files
            WHERE source_id IN (8,9) AND COALESCE(physical_preview_ready,0)=0
            GROUP BY preview_status
            ORDER BY n DESC
            """
        ).fetchall()
        log(f"REMAINING {json.dumps([dict(r) for r in rows], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
