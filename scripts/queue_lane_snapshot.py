"""Read-only queue/lane snapshot. Does not start index or write files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.db import Database
from core.index_v3.queues import JobStore
from core.index_v3.types import QueueKind
from core.settings import AppSettings


def main() -> None:
    s = AppSettings.load()
    db = Database(s.db_path)
    jobp = Path(s.db_path).with_name(Path(s.db_path).stem + ".v3jobs.db")
    print("jobs_db", jobp.exists(), jobp)
    store = JobStore(jobp) if jobp.exists() else None
    with db.connect() as conn:
        rows = list(
            conn.execute(
                "SELECT id, is_active, file_count, name, root_path FROM sources"
            )
        )
    print("SOURCES")
    for r in rows:
        print(tuple(r))
    with db.connect() as conn:
        for sid in (8, 9, 10, 11):
            tot = conn.execute(
                "SELECT COUNT(id) FROM files WHERE source_id=? "
                "AND status NOT IN ('missing','excluded_internal')",
                (sid,),
            ).fetchone()[0]
            light = conn.execute(
                "SELECT COUNT(id) FROM files WHERE source_id=? AND light_status='done'",
                (sid,),
            ).fetchone()[0]
            heavy = conn.execute(
                "SELECT COUNT(id) FROM files WHERE source_id=? AND heavy_status='done'",
                (sid,),
            ).fetchone()[0]
            print(
                f"files sid={sid} total={tot} light_done={light} heavy_done={heavy}"
            )
    if not store:
        return
    for sid in (8, 9, 10, 11):
        print(
            "jobs sid=%s light=%s preview=%s heavy=%s repair=%s"
            % (
                sid,
                store.count_pending(QueueKind.LIGHT, source_ids=[sid]),
                store.count_pending(QueueKind.PREVIEW, source_ids=[sid]),
                store.count_pending(QueueKind.HEAVY, source_ids=[sid]),
                store.count_pending(QueueKind.REPAIR, source_ids=[sid]),
            )
        )


if __name__ == "__main__":
    main()
