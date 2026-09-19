import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.db import Database
from core.settings import AppSettings

db = Database(AppSettings.load().db_path)

def parse_tm(raw):
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}

want = {".ai", ".pdf", ".eps"}
fam_by_ext = {e: Counter() for e in want}
animal_by_ext = {e: Counter() for e in want}
samples = {e: [] for e in want}

with db.connect() as conn:
    for r in conn.execute(
        """
        SELECT f.path, f.pattern_family, fe.texture_map
        FROM files f LEFT JOIN features fe ON fe.file_id=f.id
        WHERE f.source_id IN (8,9)
        """
    ):
        e = Path(r["path"]).suffix.lower()
        if e not in want:
            continue
        tm = parse_tm(r["texture_map"])
        pf = (r["pattern_family"] or tm.get("pattern_family") or "") or "(empty)"
        ap = str(tm.get("animal_print_type") or "") or "(empty)"
        fam_by_ext[e][pf] += 1
        animal_by_ext[e][ap] += 1
        if len(samples[e]) < 5 and pf not in ("(empty)",):
            samples[e].append((Path(r["path"]).name, pf, ap))

for e in (".eps", ".ai", ".pdf"):
    print("====", e)
    print(" family", fam_by_ext[e].most_common(8))
    print(" animal", animal_by_ext[e].most_common(8))
    print(" samples", samples[e])
