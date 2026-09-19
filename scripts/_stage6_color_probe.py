"""READ-ONLY Stage 6 color evidence probe."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "patterns.db"


def _named(dc) -> bool:
    if not isinstance(dc, list) or not dc:
        return False
    return all(isinstance(x, str) for x in dc)


def probe(db_path: Path) -> dict:
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    out: dict = {"db": str(db_path)}
    out["files"] = c.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]
    out["dominant_colors"] = c.execute(
        "SELECT COUNT(*) AS n FROM features WHERE dominant_colors IS NOT NULL "
        "AND dominant_colors NOT IN ('', '[]')"
    ).fetchone()["n"]
    out["color_hist"] = c.execute(
        "SELECT COUNT(*) AS n FROM features WHERE color_hist IS NOT NULL "
        "AND length(color_hist) > 0"
    ).fetchone()["n"]

    sample_n = has_ev = named_dna = fam = 0
    rows = c.execute(
        "SELECT dominant_colors, texture_map FROM features "
        "WHERE dominant_colors IS NOT NULL AND dominant_colors NOT IN ('', '[]') "
        "LIMIT 3000"
    ).fetchall()
    for r in rows:
        sample_n += 1
        try:
            tm = json.loads(r["texture_map"] or "{}")
        except Exception:
            tm = {}
        if not isinstance(tm, dict):
            tm = {}
        ev = tm.get("color_evidence") if isinstance(tm.get("color_evidence"), dict) else {}
        if ev.get("detected_colors"):
            has_ev += 1
        dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
        if _named(dna.get("dominant_colors")):
            named_dna += 1
        if tm.get("color_family"):
            fam += 1
    out["sample"] = sample_n
    out["sample_color_evidence"] = has_ev
    out["sample_named_dna"] = named_dna
    out["sample_color_family"] = fam

    # concept positives (learning)
    try:
        concepts = c.execute(
            "SELECT canonical, COUNT(*) AS n FROM concept_examples "
            "WHERE label='positive' OR polarity='positive' OR kind='positive' "
            "GROUP BY canonical"
        ).fetchall()
    except Exception:
        try:
            concepts = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%concept%'"
            ).fetchall()
            out["concept_tables"] = [dict(x) for x in concepts]
            concepts = []
        except Exception:
            concepts = []
    for row in concepts:
        can = str(row["canonical"] if "canonical" in row.keys() else row[0])
        if any(
            x in can.lower()
            for x in ("tiger", "leopard", "snake", "yilan", "rose", "floral", "gul")
        ):
            out.setdefault("positives", {})[can] = (
                row["n"] if "n" in row.keys() else row[1]
            )

    cols = {r[1] for r in c.execute("PRAGMA table_info(files)")}
    path_col = "path" if "path" in cols else ("rel_path" if "rel_path" in cols else None)
    name_col = "filename" if "filename" in cols else ("name" if "name" in cols else None)
    out["file_columns"] = sorted(cols)
    if path_col:
        for kw in ("tiger", "leopard", "snake", "yilan", "rose", "floral"):
            if name_col:
                sql = (
                    f"SELECT COUNT(*) AS n FROM files WHERE lower({path_col}) LIKE ? "
                    f"OR lower({name_col}) LIKE ?"
                )
                n = c.execute(sql, (f"%{kw}%", f"%{kw}%")).fetchone()["n"]
            else:
                sql = f"SELECT COUNT(*) AS n FROM files WHERE lower({path_col}) LIKE ?"
                n = c.execute(sql, (f"%{kw}%",)).fetchone()["n"]
            out.setdefault("path_hits", {})[kw] = n

    c.close()
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DB
    print(json.dumps(probe(path), ensure_ascii=False, indent=2))
