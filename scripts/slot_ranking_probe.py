"""Does parsed slots change TOP ranking? Measure only. No ranking rewrite.

  python scripts/slot_ranking_probe.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = [
    ("color_rose", "gül", "kırmızı gül", ["color", "object"]),
    ("color_car", "araba", "mavi araba", ["color", "object"]),
    ("brand_suv", "Togg", "Togg SUV", ["brand", "type"]),
    ("suv_alone", "SUV", "Togg SUV", ["type", "brand"]),
    ("species", "leopar", "kaplan", ["object"]),
    ("orientation", "leopar", "sağa bakan leopar", ["orientation"]),
    ("count", "leopar", "iki leopar", ["count"]),
    ("composite", "leopar", "leopar çiçek", ["object2"]),
]


def _pack(r) -> dict:
    dbg = dict(r.debug or {})
    bd = dict(r.breakdown or {})
    return {
        "file_id": int(r.file_id),
        "filename": str(r.filename or "")[:80],
        "score": round(float(r.score or 0), 4),
        "tier": dbg.get("uvi_tier") or "",
        "family": r.pattern_family or dbg.get("pattern_family") or "",
        "match": (getattr(r, "text_match_reason", "") or "")[:140],
        "filename_score": round(float(bd.get("filename_score") or 0), 3),
        "ocr_score": round(float(bd.get("ocr_score") or 0), 3),
        "nl_score": round(float(bd.get("nl_score") or 0), 3),
        "nl_color": round(float(bd.get("nl_color") or 0), 3),
        "semantic_score": round(float(bd.get("semantic_score") or bd.get("family_score") or 0), 3),
        "residual_hint": "",
    }


def _ids(rows: list[dict], n: int = 10) -> list[int]:
    return [x["file_id"] for x in rows[:n]]


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return round(len(sa & sb) / max(len(sa | sb), 1), 3)


def main() -> None:
    from core.natural_language_query import parse_natural_query
    from core.pattern_intelligence_v2 import parse_pattern_query_v2
    from core.search_engine import SearchEngine
    from core.search_models import SearchQuery
    from core.settings import AppSettings
    from core.semantic_pattern_intel import parse_pattern_intent
    from core.universal_visual_intel import parse_universal_query

    settings = AppSettings.load()
    settings.search_scope = "all"
    settings.selected_source_ids = []
    settings.semantic_pattern_intel_enabled = True
    settings.pattern_intelligence_v2_enabled = True
    settings.universal_visual_intel_enabled = True
    engine = SearchEngine(settings, load_ai=False)
    cache: dict[str, dict] = {}

    def run(q: str) -> dict:
        if q in cache:
            return cache[q]
        uq = parse_universal_query(q)
        v2 = parse_pattern_query_v2(q)
        nlq = parse_natural_query(q)
        intent = parse_pattern_intent(q)
        resp = engine.execute_search(SearchQuery(mode="text", text=q, threshold=0.35))
        rows = [_pack(r) for r in list(resp.results or [])[:10]]
        rec = {
            "n": len(resp.results or []),
            "n_all": len(resp.all_results or []),
            "top10": rows,
            "parse": {
                "uvi_node": uq.node_id,
                "uvi_path": uq.path,
                "v2_required": v2.required,
                "v2_composition": v2.composition,
                "v2_color": v2.color,
                "nl_color": nlq.color,
                "nl_motif": nlq.motif,
                "nl_brand": nlq.brand,
                "nl_residual": nlq.residual,
                "intent_color": intent.color,
                "intent_motif": intent.motif,
            },
        }
        cache[q] = rec
        print(f"  {q!r:28} n={rec['n']:3} node={uq.node_id or '-':10} residual={nlq.residual!r}")
        return rec

    print("running queries…")
    comparisons = []
    for cid, base_q, slot_q, slots in PAIRS:
        base = run(base_q)
        slot = run(slot_q)
        b_ids = _ids(base["top10"])
        s_ids = _ids(slot["top10"])
        first_changed = (b_ids[:1] != s_ids[:1]) if b_ids or s_ids else False
        color_in_top = sum(1 for x in slot["top10"] if x["nl_color"] > 0 or "Renk:" in x["match"])
        residual = str(slot["parse"]["nl_residual"] or "")
        parsed_ok = True
        notes = []
        if "orientation" in slots and not any(
            t in residual for t in ()
        ):
            # orientation/count are parse-missing if they remain residual
            pass
        if "orientation" in slots:
            parsed_ok = False
            notes.append("orientation slot yok; residual=" + residual)
        if "count" in slots:
            parsed_ok = False
            notes.append("count slot yok; residual=" + residual)
        if "type" in slots and "suv" in residual.lower() and slot["parse"]["uvi_node"] != "suv":
            parsed_ok = False
            notes.append("SUV ontolojide yok")
        jac = _jaccard(b_ids, s_ids)
        ranking_changed = jac < 0.99 or first_changed or b_ids != s_ids
        if cid == "species":
            overlap = len(set(b_ids) & set(s_ids))
            ranking_changed = overlap == 0 and bool(s_ids)
            notes.append(f"top10 overlap={overlap}")
        applied = ranking_changed and (
            color_in_top > 0
            or "çiçek" in slot_q
            or cid in ("species", "composite", "brand_suv", "suv_alone", "color_rose", "color_car")
        )
        # Honest: change can be filename token add (çiçek, mavi, kırmızı) not visual slot
        filename_driven = all(
            x["filename_score"] >= 0.7 or "Dosya adı" in x["match"]
            for x in slot["top10"][:3]
        ) if slot["top10"] else False
        if cid in ("orientation", "count"):
            verdict = "PARSE_ONLY"
        elif ranking_changed and not parsed_ok:
            verdict = "PARSE_ONLY"
        elif ranking_changed and filename_driven and color_in_top == 0 and cid.startswith("color"):
            verdict = "PARTIAL"
        elif ranking_changed:
            verdict = "RANKING_CHANGED"
        elif not parsed_ok:
            verdict = "NOT_IN_ONTOLOGY"
        else:
            verdict = "NO_RANK_CHANGE"
        if cid in ("orientation", "count"):
            verdict = "NOT_IN_ONTOLOGY"
        if cid in ("brand_suv", "suv_alone") and "suv" in residual.lower():
            verdict = "NOT_IN_ONTOLOGY" if slot["parse"]["uvi_node"] in ("togg", "") else "PARTIAL"

        comparisons.append(
            {
                "id": cid,
                "base": base_q,
                "query": slot_q,
                "slots": slots,
                "verdict": verdict,
                "jaccard_top10": jac,
                "first_changed": first_changed,
                "color_hits_in_top": color_in_top,
                "notes": notes,
                "base_n": base["n"],
                "query_n": slot["n"],
                "base_top": [
                    {"id": x["file_id"], "file": x["filename"], "score": x["score"], "match": x["match"]}
                    for x in base["top10"][:5]
                ],
                "query_top": [
                    {
                        "id": x["file_id"],
                        "file": x["filename"],
                        "score": x["score"],
                        "match": x["match"],
                        "nl_color": x["nl_color"],
                        "nl_score": x["nl_score"],
                    }
                    for x in slot["top10"][:5]
                ],
                "parse": slot["parse"],
            }
        )
        print(
            f"{verdict:18} {base_q!r} -> {slot_q!r}  jaccard={jac} first_changed={first_changed}"
        )

    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ranking_formula_changed": False,
        "ai_loaded": False,
        "clip_map": "unavailable",
        "comparisons": comparisons,
        "queries": {k: {"n": v["n"], "parse": v["parse"], "top10": v["top10"]} for k, v in cache.items()},
    }
    dest = ROOT / "data" / "reports" / "slot_ranking_probe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
