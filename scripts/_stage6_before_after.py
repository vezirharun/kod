"""Before/after Stage-6 small backfill metrics (learning untouched)."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.color_evidence import backfill_color_evidence_db  # noqa: E402


def count_ev(db: Path) -> int:
    c = sqlite3.connect(str(db))
    n = 0
    for (tm,) in c.execute(
        "SELECT texture_map FROM features WHERE dominant_colors IS NOT NULL "
        "AND dominant_colors NOT IN ('','[]') LIMIT 5000"
    ):
        try:
            d = json.loads(tm or "{}")
        except Exception:
            continue
        ev = d.get("color_evidence") if isinstance(d, dict) else None
        if isinstance(ev, dict) and ev.get("detected_colors"):
            n += 1
    c.close()
    return n


def positives(mem: Path) -> dict[str, int]:
    c = sqlite3.connect(str(mem))
    c.row_factory = sqlite3.Row
    rows = c.execute(
        """
        SELECT cr.canonical, COUNT(*) AS n
        FROM concept_examples ce
        JOIN concept_registry cr ON cr.id=ce.concept_id
        WHERE ce.role='positive'
        GROUP BY cr.canonical
        """
    ).fetchall()
    c.close()
    want = ("Tiger", "Leopard", "Snake Skin", "Rose", "Floral", "yılan", "dudak", "kalp")
    # encoding-safe: match by normalize ascii-ish
    out = {}
    for r in rows:
        can = str(r["canonical"])
        if can in want or can.lower() in {w.lower() for w in want}:
            out[can] = int(r["n"])
    # also grab tiger/leopard etc by lowercase contains
    for r in rows:
        can = str(r["canonical"])
        low = can.lower()
        if any(k in low for k in ("tiger", "leopard", "snake", "rose", "floral", "dudak", "kalp")):
            out[can] = int(r["n"])
    return out


def learning_events(mem: Path) -> int:
    c = sqlite3.connect(str(mem))
    n = c.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]
    c.close()
    return int(n)


def main() -> None:
    pdb = ROOT / "data" / "patterns.db"
    mem = ROOT / "data" / "search_memory.db"
    before = {
        "color_evidence_in_5k": count_ev(pdb),
        "positives": positives(mem),
        "learning_events": learning_events(mem),
    }
    stats = backfill_color_evidence_db(str(pdb), limit=200, only_missing=True)
    after = {
        "color_evidence_in_5k": count_ev(pdb),
        "positives": positives(mem),
        "learning_events": learning_events(mem),
        "backfill": stats,
    }
    print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))
    assert after["positives"] == before["positives"]
    assert after["learning_events"] == before["learning_events"]
    assert after["color_evidence_in_5k"] >= before["color_evidence_in_5k"]


if __name__ == "__main__":
    main()
