from pathlib import Path
import sqlite3

from core.settings import AppSettings
from core.thumbnailer import Thumbnailer
from core.preview_cache import FeaturePreviewCache
from core.preview_renderer import render_preview_for_index, resolve_display_image_path
from core.thumb_resolve import resolve_thumb_path

c = sqlite3.connect("data/patterns.db")
c.row_factory = sqlite3.Row
s = AppSettings.load()

# Pick one EPS / PDF / JPG
samples = {}
for key, like in [("eps", "%.eps"), ("pdf", "%.pdf"), ("jpg", "%.jpg"), ("tif", "%.tif")]:
    r = c.execute(
        "SELECT id, path, filename, thumbnail_path, feature_preview_path "
        "FROM files WHERE lower(filename) LIKE ? LIMIT 1",
        (like,),
    ).fetchone()
    samples[key] = dict(r) if r else None

results = {}
for key, rec in samples.items():
    if not rec:
        results[key] = {"status": "olculemedi"}
        continue
    path = rec["path"]
    # force miss thumb
    th = Thumbnailer(s.cache_dir, s.thumbnail_max_edge, s.thumbnail_format)
    out = th.thumbnail_path_for(path)
    if out.is_file():
        out.unlink()
    tr = th.create(path)
    # display resolve
    disp = resolve_display_image_path(path, s.cache_dir)
    # resolve with db paths
    resolved, st = resolve_thumb_path(
        rec["thumbnail_path"] or str(out),
        cache_dir=s.cache_dir,
        feature_preview_path=rec["feature_preview_path"] or "",
    )
    results[key] = {
        "file": rec["filename"],
        "src_exists": Path(path).is_file(),
        "create_ok": tr.success,
        "create_err": tr.error,
        "thumb_path": tr.thumbnail_path,
        "thumb_exists": bool(tr.thumbnail_path and Path(tr.thumbnail_path).is_file()),
        "display_ok": disp.success,
        "display_renderer": disp.renderer,
        "resolve_status": st,
        "resolve_path_ok": bool(resolved and Path(resolved).is_file()),
    }

print(results)
