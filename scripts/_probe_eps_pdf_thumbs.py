"""Find sample EPS/PDF and probe render."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
db = ROOT / "data" / "patterns.db"
c = sqlite3.connect(str(db))
c.row_factory = sqlite3.Row

for ext in (".eps", ".pdf", ".tif", ".tiff", ".webp"):
    row = c.execute(
        "SELECT id, path, filename, thumbnail_path, feature_preview_path, "
        "thumbnail_status, preview_status FROM files "
        "WHERE lower(filename) LIKE ? OR lower(path) LIKE ? LIMIT 3",
        (f"%{ext}", f"%{ext}"),
    ).fetchall()
    print("===", ext, "n=", len(row))
    for r in row:
        print(
            dict(
                id=r["id"],
                filename=r["filename"],
                thumb=r["thumbnail_path"],
                thumb_st=r["thumbnail_status"],
                prev=r["feature_preview_path"],
                prev_st=r["preview_status"],
                path_exists=Path(r["path"] or "").is_file() if r["path"] else False,
            )
        )
