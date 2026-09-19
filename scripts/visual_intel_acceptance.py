"""Visual Search Intelligence acceptance — real retrieval, no index rebuild.

python scripts/visual_intel_acceptance.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = (
    "yılan",
    "leopar",
    "kaplan",
    "gül",
    "kuş",
    "araba",
    "Togg",
    "kırmızı gül",
    "leopar çiçek",
    "mavi Togg SUV",
)


def _pack(r, q: str) -> dict:
    dbg = dict(r.debug or {})
    bd = dict(r.breakdown or {})
    qev = dbg.get("query_evidence_report") or bd.get("query_evidence_report") or {}
    concepts = qev.get("concepts") or {}
    channels = set()
    visual_ch = 0
    fname_ocr = 0
    for ev in concepts.values():
        ch = ev.get("channels") or {}
        channels.update(ch.keys())
        if float(ch.get("visual") or 0) >= 0.20:
            visual_ch += 1
        if float(ch.get("filename") or 0) > 0 or float(ch.get("ocr") or 0) > 0:
            fname_ocr += 1
    for k, v in bd.items():
        if k.endswith("_score") and float(v or 0) > 0:
            channels.add(k.replace("_score", ""))
    fname = str(r.filename or "")
    return {
        "file_id": getattr(r, "file_id", None),
        "filename": fname,
        "score": round(float(r.score or 0), 4),
        "tier": dbg.get("semantic_tier") or r.category,
        "clip_only": bool(dbg.get("clip_only")),
        "visual_grade": dbg.get("visual_grade") or qev.get("visual_grade") or "",
        "visual_verdict": dbg.get("visual_verdict") or qev.get("visual_verdict") or "",
        "visual_object": round(float(qev.get("visual_object") or 0), 4),
        "visual_similarity": round(float(qev.get("visual_similarity") or 0), 4),
        "visual_conflict": bool(qev.get("visual_conflict")),
        "conflict": bool(qev.get("conflict")),
        "filename_ocr_concepts": fname_ocr,
        "visual_concepts": visual_ch,
        "ocr_score": round(float(bd.get("ocr_score") or 0), 4),
        "filename_score": round(float(bd.get("filename_score") or 0), 4),
        "query_evidence": qev,
        "channels": sorted(channels),
        "composite": qev.get("composite") or "",
        "match": (r.text_match_reason or "")[:180],
    }


def main() -> None:
    from core.search_engine import SearchEngine
    from core.settings import AppSettings

    settings = AppSettings.load()
    engine = SearchEngine(settings, load_ai=True)
    faiss = getattr(engine, "faiss", None)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clip_map_unavailable": bool(getattr(faiss, "clip_map_unavailable", True)),
        "clip_count": int(getattr(faiss, "clip_count", 0) or 0),
        "queries": {},
    }
    print(
        f"CLIP map unavailable={out['clip_map_unavailable']} "
        f"clip_count={out['clip_count']}"
    )
    for q in QUERIES:
        t0 = time.perf_counter()
        rows = engine.search_by_text(q, limit=40, threshold=0.55)
        total_s = time.perf_counter() - t0
        meta = dict(getattr(engine, "_last_text_intel_meta", {}) or {})
        packed = [_pack(r, q) for r in rows]
        visual_n = sum(1 for p in packed if p.get("visual_object", 0) >= 0.20)
        fname_ocr_n = sum(1 for p in packed if p.get("filename_ocr_concepts"))
        conflict_n = sum(1 for p in packed if p.get("conflict"))
        visual_conflict_n = sum(1 for p in packed if p.get("visual_conflict"))
        first_s = float(meta.get("first_clip_ms") or 0) / 1000.0
        out["queries"][q] = {
            "n": len(rows),
            "ttfr_s": round(first_s or total_s, 3),
            "total_s": round(total_s, 3),
            "clip_k": meta.get("clip_k"),
            "clip_hits": meta.get("clip_hits"),
            "adaptive_clip": meta.get("adaptive_clip") or [],
            "visual_evidence_n": visual_n,
            "filename_ocr_n": fname_ocr_n,
            "conflict_n": conflict_n,
            "visual_conflict_n": visual_conflict_n,
            "composite_pass": sum(1 for p in packed if p.get("composite") == "PASS"),
            "top10": packed[:10],
        }
        print(
            f"{q!r:18} n={len(rows):3} first={first_s or total_s:.2f}s "
            f"final={total_s:.2f}s k={meta.get('clip_k')} "
            f"vis={visual_n} fn/ocr={fname_ocr_n} "
            f"conf={conflict_n} vconf={visual_conflict_n}"
        )
        for i, p in enumerate(packed[:10], 1):
            print(
                f"  {i:2}. {p['score']:.3f} {p['visual_grade'] or '—':16} "
                f"{p['filename'][:48]}"
            )
    dest = ROOT / "data" / "reports" / "visual_intel_acceptance.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
