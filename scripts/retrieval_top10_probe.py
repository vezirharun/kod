"""TOP-10 retrieval + OCR-only Gucci check. No index rebuild, no ranking change.

  python scripts/retrieval_top10_probe.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = ("leopar", "kaplan", "gül", "gucci")


def _ocr_only_ids(db_path: str, term: str) -> list[dict]:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    like = f"%{term}%"
    rows = con.execute(
        """
        SELECT id, filename, path, substr(ocr_text,1,120) AS ocr
        FROM files
        WHERE lower(COALESCE(ocr_text,'')) LIKE lower(?)
          AND lower(filename) NOT LIKE lower(?)
          AND lower(path) NOT LIKE lower(?)
          AND status NOT IN ('missing','excluded_internal')
        """,
        (like, like, like),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _row_pack(r, q: str) -> dict:
    dbg = dict(r.debug or {})
    bd = dict(r.breakdown or {})
    fname = str(r.filename or "")
    path = str(getattr(r, "path", "") or dbg.get("path") or "")
    ocr = str(dbg.get("ocr_raw") or bd.get("ocr_text") or "")[:120]
    qn = q.lower()
    fn_hit = qn in fname.lower() or qn in path.lower()
    # gucci aliases
    if qn == "gucci":
        fn_hit = any(t in (fname + " " + path).lower() for t in ("gucci", "gg"))
    ocr_hit = qn in (ocr or "").lower() or (
        qn == "gucci" and "gucci" in (ocr or "").lower()
    )
    dna = dbg.get("visual_dna") or {}
    return {
        "file_id": int(getattr(r, "file_id", 0) or 0),
        "filename": fname[:90],
        "score": round(float(r.score or 0), 4),
        "tier": dbg.get("uvi_tier") or "",
        "reason": (dbg.get("uvi_reason") or "")[:80],
        "node": dna.get("fine_grained") or "",
        "family": dbg.get("pattern_family") or dna.get("pattern_family") or "",
        "filename_hit": bool(fn_hit),
        "ocr_hit": bool(ocr_hit),
        "filename_score": round(float(bd.get("filename_score") or 0), 3),
        "ocr_score": round(float(bd.get("ocr_score") or bd.get("ocr_text") or 0), 3),
        "semantic_score": round(float(bd.get("semantic_score") or bd.get("family_score") or 0), 3),
        "match": (getattr(r, "text_match_reason", "") or "")[:120],
        "ocr_snip": ocr.replace("\n", " ")[:80],
    }


def main() -> None:
    from core.search_engine import SearchEngine
    from core.search_models import SearchQuery
    from core.settings import AppSettings
    from core.universal_visual_intel import parse_universal_query

    settings = AppSettings.load()
    settings.search_scope = "all"
    settings.selected_source_ids = []
    settings.semantic_pattern_intel_enabled = True
    settings.pattern_intelligence_v2_enabled = True
    settings.universal_visual_intel_enabled = True

    ocr_only = _ocr_only_ids(settings.db_path, "gucci")
    engine = SearchEngine(settings, load_ai=False)
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ranking_changed": False,
        "ai_loaded": False,
        "ocr_only_gucci": ocr_only,
        "queries": {},
    }
    for q in QUERIES:
        uq = parse_universal_query(q)
        t0 = datetime.now()
        resp = engine.execute_search(SearchQuery(mode="text", text=q, threshold=0.35))
        all_rows = list(resp.all_results or resp.results or [])
        shown = list(resp.results or [])
        packed = [_row_pack(r, q) for r in shown[:10]]
        rec = {
            "parse": {
                "node": uq.node_id,
                "path": uq.path,
                "status": uq.status,
                "rivals": uq.rivals,
            },
            "n_all": len(all_rows),
            "n_shown": len(shown),
            "top10": packed,
            "ms": int((datetime.now() - t0).total_seconds() * 1000),
        }
        if q == "gucci":
            by_id = {int(getattr(r, "file_id", 0) or 0): i for i, r in enumerate(all_rows, 1)}
            rec["ocr_only_in_results"] = []
            for row in ocr_only:
                fid = int(row["id"])
                rank = by_id.get(fid)
                rec["ocr_only_in_results"].append(
                    {
                        "file_id": fid,
                        "filename": row["filename"],
                        "ocr": (row.get("ocr") or "").replace("\n", " ")[:80],
                        "rank": rank,
                        "in_top10": bool(rank and rank <= 10),
                        "in_results": rank is not None,
                    }
                )
            rec["channels"] = {
                "filename": sum(1 for x in packed if x["filename_hit"]),
                "ocr": sum(1 for x in packed if x["ocr_hit"] and not x["filename_hit"]),
                "both": sum(1 for x in packed if x["ocr_hit"] and x["filename_hit"]),
                "semantic": sum(1 for x in packed if x["semantic_score"] >= 0.4),
            }
        out["queries"][q] = rec
        print(f"\n=== {q} n={rec['n_shown']}/{rec['n_all']} {rec['ms']}ms node={uq.node_id}")
        for i, x in enumerate(packed, 1):
            print(
                f"  {i:2} {x['score']:.3f} fn={int(x['filename_hit'])} ocr={int(x['ocr_hit'])} "
                f"{x['filename'][:55]}"
            )
        if q == "gucci":
            for x in rec["ocr_only_in_results"]:
                print(
                    f"  OCR-only id={x['file_id']} rank={x['rank']} top10={x['in_top10']} {x['filename'][:50]}"
                )

    dest = ROOT / "data" / "reports" / "retrieval_top10.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
