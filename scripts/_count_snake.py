import json
import sqlite3
from pathlib import Path

from core.settings import AppSettings

s = AppSettings.load()
con = sqlite3.connect(f"file:{Path(s.db_path).as_posix()}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

WHERE = (
    "(f.status='indexed' OR COALESCE(f.physical_preview_ready,0)=1) "
    "AND f.status NOT IN ('missing','excluded_internal')"
)


def n(sql, params=()):
    return int(con.execute(sql, params).fetchone()[0])


print("SEARCHABLE", n(f"SELECT COUNT(*) FROM files f WHERE {WHERE}"))
print("INDEXED", n("SELECT COUNT(*) FROM files WHERE status='indexed'"))

print("\n--- filename ---")
for pat in (
    "%snake%",
    "%yilan%",
    "%yılan%",
    "%python%",
    "%serpent%",
    "%snakeskin%",
):
    c = n(
        f"SELECT COUNT(*) FROM files f WHERE {WHERE} AND lower(f.filename) LIKE ?",
        (pat,),
    )
    print(pat, c)

print("\n--- ocr ---")
for pat in ("%snake%", "%yilan%", "%python%", "%serpent%"):
    c = n(
        f"SELECT COUNT(*) FROM files f WHERE {WHERE} AND lower(COALESCE(f.ocr_text,'')) LIKE ?",
        (pat,),
    )
    print("ocr", pat, c)

print("\n--- pattern fields ---")
print(
    "pattern_family=animal_print",
    n(f"SELECT COUNT(*) FROM files f WHERE {WHERE} AND f.pattern_family='animal_print'"),
)
for pt in ("snake", "leopard", "zebra", "tiger"):
    print(
        f"pattern_type={pt}",
        n(f"SELECT COUNT(*) FROM files f WHERE {WHERE} AND f.pattern_type=?", (pt,)),
    )

print("\n--- texture_map animal_print_type ---")
for apt in ("snake", "leopard", "zebra", "tiger", "crocodile"):
    c = n(
        f"""
        SELECT COUNT(*) FROM files f
        LEFT JOIN features fe ON fe.file_id=f.id
        WHERE {WHERE}
          AND (fe.texture_map LIKE ? OR fe.texture_map LIKE ?)
        """,
        (f'%"animal_print_type": "{apt}"%', f'%"animal_print_type":"{apt}"%'),
    )
    print("tm", apt, c)

print(
    "\ntexture_map contains snake",
    n(
        f"""
        SELECT COUNT(*) FROM files f
        LEFT JOIN features fe ON fe.file_id=f.id
        WHERE {WHERE} AND lower(COALESCE(fe.texture_map,'')) LIKE '%snake%'
        """
    ),
)

print("\n--- union list ---")
rows = con.execute(
    f"""
    SELECT f.id, f.status, f.filename, f.pattern_family, f.pattern_type,
           length(COALESCE(f.ocr_text,'')) AS ocr_n
    FROM files f
    LEFT JOIN features fe ON fe.file_id=f.id
    WHERE {WHERE}
      AND (
        lower(f.filename) LIKE '%snake%'
        OR lower(f.filename) LIKE '%yilan%'
        OR f.filename LIKE '%yılan%'
        OR lower(f.filename) LIKE '%python%'
        OR lower(f.filename) LIKE '%serpent%'
        OR f.pattern_type='snake'
        OR lower(COALESCE(f.ocr_text,'')) LIKE '%snake%'
        OR lower(COALESCE(f.ocr_text,'')) LIKE '%yilan%'
        OR lower(COALESCE(fe.texture_map,'')) LIKE '%"animal_print_type": "snake"%'
        OR lower(COALESCE(fe.texture_map,'')) LIKE '%"animal_print_type":"snake"%'
      )
    """
).fetchall()
print("union", len(rows))
for r in rows:
    print(dict(r))

print("\n--- sample texture_map for snake-ish ---")
for r in rows[:5]:
    tm = con.execute(
        "SELECT texture_map FROM features WHERE file_id=?", (r["id"],)
    ).fetchone()
    raw = (tm["texture_map"] if tm else "") or ""
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {}
    print(
        r["id"],
        {
            "pattern_family": obj.get("pattern_family"),
            "animal_print_type": obj.get("animal_print_type"),
            "semantic_tags": obj.get("semantic_tags"),
        },
    )

print("\n--- missing snake-named images metadata ---")
ids = [153068, 153072, 153081, 156008, 157861, 159217, 153080, 152949, 153039]
q = f"""
SELECT f.id, f.filename, f.pattern_family, f.pattern_type,
       json_extract(fe.texture_map, '$.pattern_family') AS tm_fam,
       json_extract(fe.texture_map, '$.animal_print_type') AS tm_apt,
       json_extract(fe.texture_map, '$.semantic_tags.motif') AS motif,
       json_extract(fe.texture_map, '$.semantic_tags.family') AS sem_fam
FROM files f
LEFT JOIN features fe ON fe.file_id=f.id
WHERE f.id IN ({",".join("?" * len(ids))})
"""
for r in con.execute(q, ids):
    print(dict(r))

print("\n--- animal_print in texture_map family ---")
print(
    "tm pattern_family animal_print",
    n(
        f"""
        SELECT COUNT(*) FROM files f
        LEFT JOIN features fe ON fe.file_id=f.id
        WHERE {WHERE}
          AND (fe.texture_map LIKE '%"pattern_family": "animal_print"%'
               OR fe.texture_map LIKE '%"pattern_family":"animal_print"%')
        """
    ),
)
con.close()
