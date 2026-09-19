from pathlib import Path
import sqlite3
from core.settings import AppSettings

s = AppSettings.load()
con = sqlite3.connect(f"file:{Path(s.db_path).as_posix()}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
for fid in (158547, 158548):
    r = con.execute(
        "SELECT id, status, source_id, physical_preview_ready, "
        "length(ocr_text) AS ocr_n, filename FROM files WHERE id=?",
        (fid,),
    ).fetchone()
    print(dict(r))
print("--- ocr like ---")
for r in con.execute(
    "SELECT id, status, filename FROM files WHERE ocr_text LIKE '%GUCCI%' LIMIT 10"
):
    print(tuple(r))
con.close()
