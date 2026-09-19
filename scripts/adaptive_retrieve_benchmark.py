"""Adaptive retrieve vs sabit Top-800 — ranking formülü aynı, sadece havuz.

  python scripts/adaptive_retrieve_benchmark.py --queries 5
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.adaptive_retrieve import is_relevant_hit
from core.search_engine import SearchEngine
from core.search_latency import classify_budget
from core.search_models import SearchQuery
from core.settings import AppSettings

OUT = Path("data") / "benchmarks"


def _ids(results, n: int) -> list[int]:
    return [int(r.file_id) for r in (results or [])[:n]]


def _recall(ref: list[int], got: list[int]) -> float:
    if not ref:
        return 1.0
    s = set(got)
    return round(sum(1 for i in ref if i in s) / len(ref), 4)


def _run(engine: SearchEngine, path: str) -> dict:
    t0 = time.perf_counter()
    first = {}

    def cb(scored, n, tot):
        if "ttfr" not in first and scored:
            first["ttfr"] = (time.perf_counter() - t0) * 1000.0
            first["k"] = tot

    engine.start_latency_profile()
    resp = engine.execute_search(
        SearchQuery(mode="image", image_path=path),
        result_callback=cb,
    )
    wall = (time.perf_counter() - t0) * 1000.0
    lat = dict(resp.meta.get("latency") or {})
    floor = float(engine.settings.similarity_threshold or 0.40)
    relevant = [r for r in (resp.all_results or []) if is_relevant_hit(r, floor)]
    strong = [r for r in relevant if float(r.score) >= 0.70]
    return {
        "wall_ms": round(wall, 1),
        "ttfr_ms": round(float(first.get("ttfr") or 0), 1),
        "budget": classify_budget(wall),
        "retrieve_k": int(resp.meta.get("retrieve_k") or 0),
        "candidates": len(resp.all_results or []),
        "relevant": len(relevant),
        "strong": len(strong),
        "extra_strong": int(resp.meta.get("extra_strong") or 0),
        "stop_reason": resp.meta.get("stop_reason") or "",
        "batches": resp.meta.get("batches") or [],
        "top20": _ids(resp.all_results, 20),
        "top100": _ids(resp.all_results, 100),
        "spans": (lat.get("spans_ms") or {}),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=5)
    args = ap.parse_args()
    qpath = OUT / "visual_search_queries.json"
    queries = json.loads(qpath.read_text(encoding="utf-8"))[: max(3, args.queries)]
    settings = AppSettings.load()
    engine = SearchEngine(settings, load_ai=False)
    rows = []
    extra_total = 0
    for i, q in enumerate(queries):
        settings.adaptive_retrieve_enabled = False
        old = _run(engine, q["path"])
        settings.adaptive_retrieve_enabled = True
        new = _run(engine, q["path"])
        extra = max(0, int(new["strong"]) - int(old["strong"]))
        extra_total += extra
        rec = {
            "query": q.get("filename") or q["path"],
            "bucket": q.get("bucket"),
            "old": {k: old[k] for k in ("wall_ms", "ttfr_ms", "candidates", "relevant", "strong", "retrieve_k")},
            "new": {k: new[k] for k in ("wall_ms", "ttfr_ms", "candidates", "relevant", "strong", "retrieve_k", "extra_strong", "stop_reason")},
            "recall@10": _recall(old["top20"][:10], new["top20"][:10]),
            "recall@20": _recall(old["top20"], new["top20"]),
            "recall@50": _recall(old["top100"][:50], new["top100"][:50]),
            "recall@100": _recall(old["top100"], new["top100"]),
            "new_strong_beyond_800": extra,
            "batches": new.get("batches"),
        }
        rows.append(rec)
        print(
            f"[{i+1}] {q.get('bucket')} {str(q.get('filename'))[:36]} "
            f"old_k=800 new_k={new['retrieve_k']} rel {old['relevant']}→{new['relevant']} "
            f"strong+{extra} R@20={rec['recall@20']} ttfr={new['ttfr_ms']:.0f}ms "
            f"stop={new['stop_reason']}",
            flush=True,
        )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "new_strong_beyond_800_total": extra_total,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "adaptive_retrieve_latest.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("NEW_STRONG_BEYOND_800", extra_total)
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
