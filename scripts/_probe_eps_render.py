from pathlib import Path
import sqlite3
from PIL import Image

from core.preview_renderer import render_preview_for_index
from core.settings import AppSettings
from core.thumbnailer import Thumbnailer

c = sqlite3.connect("data/patterns.db")
c.row_factory = sqlite3.Row
rows = c.execute(
    "SELECT id, filename, path, thumbnail_path, feature_preview_path "
    "FROM files WHERE lower(filename) LIKE '%.eps' LIMIT 30"
).fetchall()
miss = broken = ok = 0
for r in rows:
    tp = Path(r["thumbnail_path"] or "")
    if not tp.is_file() or tp.stat().st_size <= 0:
        miss += 1
        print("MISS", r["id"], r["filename"])
        continue
    try:
        with Image.open(tp) as im:
            im.verify()
        ok += 1
    except Exception as e:
        broken += 1
        print("BROKEN", r["id"], r["filename"], e, "size", tp.stat().st_size)
print("sample ok", ok, "miss", miss, "broken", broken)
print(
    "total eps",
    c.execute(
        "SELECT COUNT(*) FROM files WHERE lower(filename) LIKE '%.eps'"
    ).fetchone()[0],
)

# EPS without usable thumb
rows2 = c.execute(
    "SELECT id, filename, path, thumbnail_path FROM files "
    "WHERE lower(filename) LIKE '%.eps' LIMIT 200"
).fetchall()
need = None
for r in rows2:
    tp = Path(r["thumbnail_path"] or "")
    if not tp.is_file() or tp.stat().st_size <= 0:
        need = r
        break
    try:
        with Image.open(tp) as im:
            im.load()
    except Exception:
        need = r
        break
if need is None:
    need = rows[0]
print("probe file", need["id"], need["filename"], need["path"])
s = AppSettings.load()
print("cache_dir", s.cache_dir)
rr = render_preview_for_index(need["path"], s.cache_dir)
print("render", rr)
th = Thumbnailer(s.cache_dir, s.thumbnail_max_edge, s.thumbnail_format)
# delete thumb to force recreate
out = th.thumbnail_path_for(need["path"])
if out.is_file():
    out.unlink()
    print("deleted", out)
tr = th.create(need["path"])
print("thumb create", tr)
