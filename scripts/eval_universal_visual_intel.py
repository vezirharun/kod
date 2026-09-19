"""Shadow eval for Universal Visual Intelligence. No reindex / FAISS rebuild."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [
    "araba",
    "Togg",
    "kuş",
    "karga",
    "leylek",
    "fare",
    "gül",
    "leopar",
    "yılan",
    "leopar çiçek",
    "gerçek leopar fotoğrafı",
    "küçük kargalı desen",
]


def main() -> None:
    from core.search_engine import SearchEngine
    from core.search_models import SearchQuery
    from core.settings import AppSettings
    from core.universal_visual_intel import CAPABILITIES, ONTOLOGY, parse_universal_query

    settings = AppSettings.load()
    settings.semantic_pattern_intel_enabled = True
    settings.pattern_intelligence_v2_enabled = True
    settings.universal_visual_intel_enabled = True
    engine = SearchEngine(settings, load_ai=True)
    engine.ensure_ai_loaded()
    clip_ok = bool(
        engine.extractor.ai_available
        and getattr(engine.extractor, "_clip_model", None) is not None
        and engine.faiss.available
        and engine.faiss.clip_count > 0
    )
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "production_index_touched": False,
        "clip_active": clip_ok,
        "capabilities": CAPABILITIES,
        "ontology_size": len(ONTOLOGY),
        "queries": {},
    }
    for q in QUERIES:
        uq = parse_universal_query(q)
        resp = engine.execute_search(SearchQuery(mode="text", text=q, threshold=0.40))
        rows = list(resp.results or [])[:20]
        meta = resp.meta or {}
        uvi = meta.get("universal_visual_intel") or {}
        labels = []
        for r in rows:
            dna = r.debug.get("visual_dna") or {}
            labels.append(
                {
                    "file": (r.filename or "")[:70],
                    "tier": r.debug.get("uvi_tier"),
                    "reason": r.debug.get("uvi_reason"),
                    "node": dna.get("fine_grained"),
                    "rep": dna.get("representation"),
                    "comp": dna.get("composition"),
                }
            )
        n = len(rows)
        exact_n = sum(1 for x in labels if x.get("tier") == "EXACT")

        def prec(k: int) -> dict:
            chunk = labels[:k]
            if not chunk:
                return {"p": 0.0, "n": 0, "exact": 0, "note": "filename_proxy_not_ground_truth"}
            return {
                "p": round(exact_n / max(len(chunk), 1), 3) if k >= n else round(
                    sum(1 for x in chunk if x.get("tier") == "EXACT") / len(chunk), 3
                ),
                "n": len(chunk),
                "exact": sum(1 for x in chunk if x.get("tier") == "EXACT"),
                "same_family": sum(1 for x in chunk if x.get("tier") == "SAME_FAMILY"),
                "note": "auto-P uses UVI tier, not human labels; circular for CLIP queries",
            }

        rec = {
            "node": uq.node_id,
            "path": uq.path,
            "status": uq.status,
            "leaf": uq.leaf,
            "n": n,
            "precision@10": prec(10),
            "precision@20": prec(20),
            "dropped": uvi.get("dropped_unspecific"),
            "tiers": uvi.get("tiers"),
            "clip_used": bool(meta.get("ai_faiss_used")) or clip_ok,
            "top": labels[:8],
        }
        out["queries"][q] = rec
        p10 = rec["precision@10"]
        print(
            f"{q:28} node={uq.node_id or '-':10} n={n:3} "
            f"p@10~{p10['p']:.3f} dropped={uvi.get('dropped_unspecific')}"
        )

    dest = ROOT / "data" / "reports" / "universal_visual_intel_v1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)
    print("production_index_touched=False")


if __name__ == "__main__":
    main()
