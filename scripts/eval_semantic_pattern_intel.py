"""Semantic Pattern Intelligence — before/after benchmark (no index/FAISS rebuild)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [
    "çiçek",
    "papatya",
    "gül",
    "lale",
    "puantiye",
    "geometrik",
    "çizgi",
    "ekose",
    "leopar",
    "zebra",
    "yılan derisi",
    "paisley",
    "monogram",
    "Louis Vuitton",
    "LV",
    "Gucci",
    "GG",
]


def _summarize(results: list) -> dict:
    n = len(results)
    clip_n = sum(1 for r in results if float((r.breakdown or {}).get("clip") or r.debug.get("clip_score") or 0) >= 0.20)
    fam_n = sum(1 for r in results if float((r.breakdown or {}).get("family_score") or 0) >= 0.60)
    meta_n = sum(
        1
        for r in results
        if float((r.breakdown or {}).get("filename_score") or 0) >= 0.70
        or float((r.breakdown or {}).get("ocr_score") or 0) >= 0.70
    )
    dna_n = sum(
        1
        for r in results
        if (r.debug or {}).get("semantic_evidence", {}).get("subtype_hit")
        or (r.debug or {}).get("semantic_evidence", {}).get("dna_hit")
    )
    tiers: dict[str, int] = {}
    for r in results:
        t = str((r.debug or {}).get("semantic_tier") or r.category or "")
        tiers[t] = tiers.get(t, 0) + 1
    top = []
    for r in results[:8]:
        top.append(
            {
                "file_id": r.file_id,
                "filename": r.filename,
                "score": round(float(r.score), 3),
                "tier": (r.debug or {}).get("semantic_tier") or "",
                "clip": round(float((r.debug or {}).get("clip_score") or 0), 3),
                "family": r.pattern_family,
            }
        )
    conf = round(sum(float(r.score) for r in results) / n, 3) if n else 0.0
    return {
        "candidates": n,
        "top_k": min(8, n),
        "clip_contrib": clip_n,
        "metadata_contrib": meta_n,
        "dna_contrib": dna_n,
        "family_contrib": fam_n,
        "confidence_mean": conf,
        "tiers": tiers,
        "top": top,
    }


def main() -> None:
    from core.search_engine import SearchEngine
    from core.settings import AppSettings

    settings = AppSettings.load()
    settings.semantic_pattern_intel_enabled = False
    settings.search_text_visual = False
    engine = SearchEngine(settings, load_ai=False)
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "clip_model_loaded": bool(getattr(engine.extractor, "_clip_model", None)),
        "faiss_clip": engine.faiss.clip_count,
        "queries": {},
    }
    for q in QUERIES:
        settings.semantic_pattern_intel_enabled = False
        old = engine.search_by_text(q, limit=400, threshold=0.60)
        settings.semantic_pattern_intel_enabled = True
        new = engine.search_by_text(q, limit=400, threshold=0.60)
        out["queries"][q] = {
            "before": _summarize(old),
            "after": _summarize(new),
            "meta": dict(getattr(engine, "_last_text_intel_meta", {}) or {}),
        }
        print(
            f"{q:20} before={len(old):4} after={len(new):4} "
            f"clip={out['queries'][q]['after']['clip_contrib']}"
        )
    dest = ROOT / "data" / "reports" / "semantic_pattern_intel_bench.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
