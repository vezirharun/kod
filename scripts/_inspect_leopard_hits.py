import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.db import Database
from core.settings import AppSettings

p = Path("data/reports/search_debug_20260814_200538.json")
d = json.loads(p.read_text(encoding="utf-8"))
rows = d.get("top_100_breakdown") or []
print("query", d.get("query_path"))
print("query_animal", d.get("query_animal_print_type"))
print("rows", len(rows))
print("animal", Counter(r.get("animal_print_type") or "" for r in rows).most_common(8))
print("family", Counter(r.get("pattern_family") or "" for r in rows).most_common(8))
print("category", Counter(r.get("category") or "" for r in rows).most_common(8))

settings = AppSettings.load()
db = Database(settings.db_path)
ids = [int(r["file_id"]) for r in rows[:40]]
thumbs = {}
with db.connect() as conn:
    q = ",".join("?" * len(ids))
    for rec in conn.execute(
        f"SELECT id, filename, thumbnail_path, feature_preview_path, path FROM files WHERE id IN ({q})",
        ids,
    ):
        thumbs[int(rec["id"])] = dict(rec)

out = Path("data/reports/leopard_inspect_sample.json")
sample = []
for i, r in enumerate(rows[:40], 1):
    rec = thumbs.get(int(r["file_id"]), {})
    item = {
        "rank": i,
        "file_id": r["file_id"],
        "score": r.get("score_percent"),
        "filename": r.get("filename"),
        "animal": r.get("animal_print_type"),
        "family": r.get("pattern_family"),
        "category": r.get("category"),
        "reason": (r.get("reason") or "")[:120],
        "thumbnail_path": rec.get("thumbnail_path") or "",
        "preview_path": rec.get("feature_preview_path") or "",
        "path": rec.get("path") or r.get("path") or "",
    }
    sample.append(item)
    print(
        f"{i:02d} {item['score']:5.1f} {item['animal']:8} {item['filename']}"
    )
out.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
