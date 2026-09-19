#!/usr/bin/env python
"""Vezir Pattern Search V3 index diagnostic.

Usage:
    python scripts/diagnose_index_v3.py --db data/patterns.db --source-id 1

Reports the DB artifact pools and V3 job states without changing the DB.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from core.db import Database
from core.index_v3.queues import JobStore
from core.index_v3.ui_bridge import count_v3_ssot
from core.index_v3.types import QueueKind


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--source-id", type=int, action="append", default=[])
    args = ap.parse_args()

    db = Database(args.db)
    sids = args.source_id or None
    pools = count_v3_ssot(db, sids)
    print("=== V3 INDEX DURUMU ===")
    for k in (
        "total", "preview", "thumbnail", "light_complete",
        "hash", "metadata", "dino", "clip", "texture",
        "semantic", "dna", "patch", "ocr", "ai_final",
        "waiting_preview",
    ):
        if k in pools:
            print(f"{k:18} {pools[k]}")

    job_db = Path(args.db).with_name(Path(args.db).stem + ".v3jobs.db")
    store = JobStore(job_db)
    print("\n=== KUYRUK ===")
    for q in QueueKind:
        print(
            f"{q.value:10} pending={store.count_pending(q, source_ids=sids)} "
            f"claimed={store.count_claimed(q, source_ids=sids)} "
            f"retry={store.count_pending_retry(q, source_ids=sids)}"
        )
    print(f"\nV3 job DB: {job_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
