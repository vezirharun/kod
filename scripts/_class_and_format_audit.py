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
print("search_top100_exts", Counter(Path(r.get("path") or "").suffix.lower() for r in rows).most_common())
print("search_displayed", (d.get("stats") or {}).get("displayed"))
print(
    "search_eps_ai_pdf",
    [
        Path(r.get("path") or "").name
        for r in rows
        if Path(r.get("path") or "").suffix.lower() in {".eps", ".ai", ".pdf"}
    ],
)

s = AppSettings.load()
db = Database(s.db_path)


def ext_of(path: str) -> str:
    return Path(path or "").suffix.lower()


def parse_tm(raw):
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


with db.connect() as conn:
    fam = Counter()
    animal = Counter()
    leopard_ext = Counter()
    ready_by_ext = Counter()
    all_ext = Counter()
    vector_all = Counter()
    vector_ready = Counter()
    vector_leopard = []
    name_leo_vector = []
    unlabeled_ready = 0
    labeled = 0
    for r in conn.execute(
        """
        SELECT f.path, f.pattern_family,
               COALESCE(f.physical_preview_ready,0) prev,
               fe.texture_map
        FROM files f
        LEFT JOIN features fe ON fe.file_id=f.id
        WHERE f.source_id IN (8,9)
        """
    ):
        e = ext_of(r["path"])
        all_ext[e] += 1
        if r["prev"]:
            ready_by_ext[e] += 1
        tm = parse_tm(r["texture_map"])
        pf = (r["pattern_family"] or tm.get("pattern_family") or "").strip()
        ap = str(tm.get("animal_print_type") or "").strip()
        if pf:
            fam[pf] += 1
            labeled += 1
        elif r["prev"]:
            unlabeled_ready += 1
        if ap:
            animal[ap] += 1
        name = Path(r["path"]).name.lower()
        if e in {".eps", ".ai", ".pdf", ".psd", ".svg"}:
            vector_all[e] += 1
            if r["prev"]:
                vector_ready[e] += 1
            if "leopard" in name or "leopar" in name:
                name_leo_vector.append(
                    (e, Path(r["path"]).name, int(r["prev"]), pf, ap)
                )
        if ap == "leopard":
            leopard_ext[e] += 1
            if e in {".eps", ".ai", ".pdf", ".psd", ".svg", ".cdr"}:
                vector_leopard.append((e, Path(r["path"]).name, int(r["prev"])))

print("ALL_EXT", all_ext.most_common(12))
print("READY_EXT", ready_by_ext.most_common(12))
print("FAMILY", fam.most_common(15))
print("ANIMAL", animal.most_common(12))
print("LEOPARD_EXT", leopard_ext.most_common())
print("vector_all", dict(vector_all))
print("vector_ready", dict(vector_ready))
print("vector_classified_leopard", len(vector_leopard))
for item in vector_leopard[:25]:
    print(" VEC_LEO", item)
print("filename_leo_vector", name_leo_vector, "count", len(name_leo_vector))
print("labeled", labeled, "unlabeled_ready", unlabeled_ready)
