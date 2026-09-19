"""Count EPS/PDF thumb path existence vs DB."""
import sqlite3
from pathlib import Path

c = sqlite3.connect("data/patterns.db")
c.row_factory = sqlite3.Row

for label, like in [("eps", "%.eps"), ("pdf", "%.pdf"), ("tif", "%.tif"), ("jpg", "%.jpg")]:
    rows = c.execute(
        "SELECT thumbnail_path, path, feature_preview_path FROM files "
        "WHERE lower(filename) LIKE ?",
        (like,),
    ).fetchall()
    n = len(rows)
    thumb_db = sum(1 for r in rows if (r["thumbnail_path"] or "").strip())
    thumb_hit = 0
    thumb_miss = 0
    src_ok = 0
    for r in rows:
        tp = Path(r["thumbnail_path"] or "")
        if tp.is_file() and tp.stat().st_size > 0:
            thumb_hit += 1
        elif (r["thumbnail_path"] or "").strip():
            thumb_miss += 1
        if r["path"] and Path(r["path"]).is_file():
            src_ok += 1
    print(
        f"{label}: n={n} thumb_db={thumb_db} thumb_hit={thumb_hit} "
        f"thumb_db_miss_file={thumb_miss} src_ok={src_ok}"
    )
