"""Stage 7 live window: backfill + metrics (learning untouched)."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.color_evidence import backfill_color_evidence_db  # noqa: E402


def scan_evidence(db: Path, sample: int = 8000) -> dict:
    c = sqlite3.connect(str(db))
    out = {
        "named": 0,
        "with_ratio": 0,
        "user_locked": 0,
        "sampled": 0,
    }
    for (tm,) in c.execute(
        "SELECT texture_map FROM features WHERE dominant_colors IS NOT NULL "
        "AND dominant_colors NOT IN ('','[]') LIMIT ?",
        (sample,),
    ):
        out["sampled"] += 1
        try:
            d = json.loads(tm or "{}")
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        src = str(d.get("color_source") or "")
        if src in {"teach_me", "manual_user", "user"}:
            out["user_locked"] += 1
        ev = d.get("color_evidence") if isinstance(d.get("color_evidence"), dict) else {}
        if ev.get("detected_colors"):
            out["named"] += 1
        if ev.get("ratio_confidence") == "cluster_weight" and ev.get("color_ratios"):
            out["with_ratio"] += 1
    c.close()
    return out


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
    keys = ("Tiger", "Leopard", "Snake Skin", "Floral", "Rose", "dudak", "kalp")
    out = {}
    for r in rows:
        can = str(r["canonical"])
        if can in keys or any(k.lower() in can.lower() for k in keys):
            out[can] = int(r["n"])
    return out


def query_smoke() -> dict:
    from core.query_attribute_intel import extract_query_attributes

    out = {}
    for q in (
        "kaplan",
        "siyah kaplan",
        "siyah krem kaplan",
        "black tiger",
        "red flower",
        "mavi çiçek",
    ):
        a = extract_query_attributes(q)
        out[q] = {"motif": a.motif, "colors": list(a.colors), "style": a.style}
    return out


def main() -> None:
    pdb = ROOT / "data" / "patterns.db"
    mem = ROOT / "data" / "search_memory.db"
    before = {
        "evidence": scan_evidence(pdb),
        "positives": positives(mem),
    }
    stats = backfill_color_evidence_db(
        str(pdb),
        limit=200,
        only_missing=True,
        use_thumbnails=True,
        batch_size=100,
        upgrade_ratios=True,
    )
    after = {
        "evidence": scan_evidence(pdb),
        "positives": positives(mem),
        "backfill": stats,
        "query_smoke": query_smoke(),
    }
    assert after["positives"] == before["positives"]
    print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
