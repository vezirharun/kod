"""Pattern Intelligence v2 archive benchmark. No reindex/FAISS rebuild."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [
    "fare",
    "kuş",
    "gül",
    "papatya",
    "leopar",
    "yılan",
    "zebra",
    "leopar çiçek",
    "leopar gül",
    "leopar yılan",
    "kuş çiçek",
    "fare çiçek",
    "kelebek çiçek",
]

TOK = {
    "mouse": ("fare", "mouse", "mice"),
    "bird": ("kus", "kuş", "bird"),
    "butterfly": ("kelebek", "butterfly"),
    "rose": ("gul", "gül", "rose"),
    "daisy": ("papatya", "daisy"),
    "leopard": ("leopar", "leopard"),
    "snake": ("yilan", "snake", "snakeskin", "python"),
    "zebra": ("zebra",),
    "floral": ("cicek", "floral", "flower", "rose", "papatya", "daisy"),
}


def _has(blob: str, key: str) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", blob) for t in TOK.get(key, (key,)))


def _eval_row(r, required: list[str], composite: bool) -> str:
    from core.textile_terms import normalize_turkish

    blob = normalize_turkish(r.filename)
    ev = r.debug.get("semantic_evidence") or {}
    v2 = r.debug.get("pattern_intel_v2") or {}
    animal = str(ev.get("animal_type") or "")
    ids = [h.get("id") for h in (v2.get("motifs") or []) + (v2.get("objects") or [])]
    flags = []
    for req in required:
        ok = False
        if req == animal or req in ids:
            ok = True
        if _has(blob, req):
            ok = True
        flags.append(ok)
    if composite:
        if all(flags):
            return "relevant"
        if any(flags) and not all(flags):
            return "wrong_composite"
        return "unknown"
    if required and flags[0]:
        rivals = {"leopard", "snake", "zebra", "tiger"} - set(required)
        if animal in rivals or any(_has(blob, x) for x in rivals):
            return "wrong_subtype"
        return "relevant"
    return "unknown"


def main() -> None:
    from core.pattern_intelligence_v2 import parse_pattern_query_v2
    from core.search_engine import SearchEngine
    from core.search_models import SearchQuery
    from core.settings import AppSettings

    settings = AppSettings.load()
    settings.semantic_pattern_intel_enabled = True
    settings.pattern_intelligence_v2_enabled = True
    engine = SearchEngine(settings, load_ai=True)
    engine.ensure_ai_loaded()
    out = {"created_at": datetime.now().isoformat(timespec="seconds"), "queries": {}}
    for q in QUERIES:
        pq = parse_pattern_query_v2(q)
        resp = engine.execute_search(SearchQuery(mode="text", text=q, threshold=0.40))
        rows = list(resp.results or [])[:20]
        meta = resp.meta or {}
        labels = [_eval_row(r, pq.required, pq.composition == "composite") for r in rows]

        def prec(k: int) -> dict:
            chunk = labels[:k]
            if not chunk:
                return {"precision": 0.0, "n": 0, "wrong_subtype": 0, "wrong_composite": 0, "unknown": 0}
            rel = chunk.count("relevant")
            return {
                "precision": round(rel / len(chunk), 3),
                "n": len(chunk),
                "wrong_subtype": chunk.count("wrong_subtype"),
                "wrong_composite": chunk.count("wrong_composite"),
                "unknown": chunk.count("unknown"),
            }

        p10, p20 = prec(10), prec(20)
        v2m = meta.get("pattern_intelligence_v2") or {}
        rec = {
            "required": pq.required,
            "composition": pq.composition,
            "n": len(rows),
            "precision@10": p10,
            "precision@20": p20,
            "object_evidence": v2m.get("object_evidence"),
            "clip_evidence": v2m.get("clip_evidence"),
            "dna_evidence": v2m.get("dna_evidence"),
            "composition_evidence": v2m.get("composition_evidence"),
            "dropped": v2m.get("dropped"),
            "clip_used": bool(meta.get("ai_faiss_used")),
            "top": [
                {
                    "file": r.filename[:70],
                    "label": labels[i],
                    "reason": r.debug.get("v2_reason"),
                    "bucket": r.debug.get("v2_bucket"),
                    "rep": (r.debug.get("pattern_intel_v2") or {}).get("representation"),
                    "comp": (r.debug.get("pattern_intel_v2") or {}).get("composition"),
                }
                for i, r in enumerate(rows[:8])
            ],
        }
        out["queries"][q] = rec
        print(
            f"{q:18} n={len(rows):3} p@10={p10['precision']:.3f} "
            f"ws={p10['wrong_subtype']} wc={p10['wrong_composite']} "
            f"obj={v2m.get('object_evidence')} clip={v2m.get('clip_evidence')} "
            f"comp={v2m.get('composition_evidence')} dropped={v2m.get('dropped')}"
        )
    dest = ROOT / "data" / "reports" / "pattern_intel_v2_bench.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
