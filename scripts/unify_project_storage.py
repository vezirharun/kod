"""One-shot: unify production DB under project data/ + cache/."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from core.db import Database
from core.index_v3.discovery import enqueue_existing_gaps
from core.index_v3.physical_reconcile import reconcile_stale_physical_flags
from core.index_v3.queues import JobStore
from core.index_v3.types import Mode
from core.index_v3.ui_bridge import count_v3_ssot
from core.settings import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, PROJECT_ROOT

LEGACY_DATA = Path(r"C:\mnt\data\work_v12424\data")
LEGACY_CACHE = Path(r"C:\mnt\data\work_v12424\cache")
DEST_DB = DEFAULT_DATA_DIR / "patterns.db"
DEST_CACHE = DEFAULT_CACHE_DIR


def _copy_missing(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in src_dir.iterdir():
        if not src.is_file():
            continue
        dst = dst_dir / src.name
        if dst.is_file() and dst.stat().st_size > 0:
            continue
        shutil.copy2(src, dst)
        n += 1
    return n


def _rewrite_cache_paths(db_path: Path, new_cache: Path) -> int:
    olds = [
        "/mnt/data/work_v12424/cache",
        r"\mnt\data\work_v12424\cache",
        r"C:\mnt\data\work_v12424\cache",
        str(LEGACY_CACHE),
    ]
    new = str(new_cache)
    n = 0
    con = sqlite3.connect(str(db_path))
    try:
        for col in ("feature_preview_path", "thumbnail_path"):
            for old in olds:
                cur = con.execute(
                    f"UPDATE files SET {col}=replace({col}, ?, ?) "
                    f"WHERE instr({col}, ?) > 0",
                    (old, new, old),
                )
                n += int(cur.rowcount or 0)
        con.commit()
    finally:
        con.close()
    return n


def main() -> None:
    src_db = LEGACY_DATA / "patterns.db"
    if not src_db.is_file():
        raise SystemExit(f"legacy db missing: {src_db}")

    stub = DEFAULT_DATA_DIR / "patterns.stub.bak.db"
    if DEST_DB.is_file() and DEST_DB.stat().st_size < 1_000_000 and not stub.is_file():
        shutil.copy2(DEST_DB, stub)

    print("backup_sqlite", src_db, "->", DEST_DB)
    src = sqlite3.connect(str(src_db))
    dst = sqlite3.connect(str(DEST_DB))
    with dst:
        src.backup(dst)
    src.close()
    dst.close()

    copied_fp = _copy_missing(LEGACY_CACHE / "feature_previews", DEST_CACHE / "feature_previews")
    copied_th = _copy_missing(LEGACY_CACHE / "thumbnails", DEST_CACHE / "thumbnails")
    print("copied_missing_previews", copied_fp, "thumbs", copied_th)

    rewritten = _rewrite_cache_paths(DEST_DB, DEST_CACHE)
    print("rewritten_path_rows", rewritten)

    db = Database(str(DEST_DB))
    relink = reconcile_stale_physical_flags(
        db, [1, 2], cache_dir=str(DEST_CACHE)
    )
    print("relinked", relink)
    snap = count_v3_ssot(db, [1, 2])
    print(
        "ssot",
        {k: snap.get(k) for k in ("total", "preview", "thumbnail", "dino", "openclip", "hash")},
    )

    store = JobStore(DEST_DB.with_name(DEST_DB.stem + ".v3jobs.db"))
    total_jobs = 0
    after_id = 0
    while True:
        n, last = enqueue_existing_gaps(
            db,
            store,
            source_id=1,
            mode=Mode.GENERAL_AI,
            limit=2000,
            after_id=after_id,
        )
        total_jobs += int(n)
        if not last:
            break
        after_id = int(last)
    pending = store.count_pending(source_ids=[1])
    print("general_ai_enqueued", total_jobs, "pending", pending)
    print("project_root", PROJECT_ROOT)


if __name__ == "__main__":
    main()
