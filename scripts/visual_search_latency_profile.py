"""Görsel arama latency kabul testi — yalnız ölçüm, ranking değişmez.

Kullanım (production DB/FAISS'e yazmaz):
  python scripts/visual_search_latency_profile.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import Database
from core.search_engine import SearchEngine
from core.search_latency import (
    classify_budget,
    classify_storage,
    percentile,
    summarize_ms,
)
from core.search_models import SearchQuery
from core.settings import AppSettings


OUT_DIR = Path("data") / "benchmarks"
QUERY_LIST = OUT_DIR / "visual_search_queries.json"
K_SWEEP = (20, 50, 100, 250, 500, 1000)


def _parse_tm(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str) and raw.strip():
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}
    return {}


def _pick_queries(db: Database, n: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with db.connect() as conn:
        cur = conn.execute(
            """
            SELECT f.id, f.path, f.filename, f.pattern_family, f.pattern_type,
                   f.pattern_subtype, f.partial_hash, f.source_id
            FROM files f
            JOIN sources s ON s.id = f.source_id
            JOIN features fe ON fe.file_id = f.id
            WHERE s.is_active = 1
              AND fe.dino_embedding IS NOT NULL
              AND length(fe.dino_embedding) > 32
            ORDER BY f.id
            LIMIT 4000
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
    existing = [r for r in rows if r.get("path") and os.path.isfile(str(r["path"]))]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_hash: dict[str, int] = defaultdict(int)
    for r in existing:
        ph = str(r.get("partial_hash") or "")
        if ph:
            seen_hash[ph] += 1
    for r in existing:
        fam = str(r.get("pattern_family") or "unknown")
        animal = str(r.get("pattern_type") or "")
        subtype = str(r.get("pattern_subtype") or "")
        r["family"] = fam
        r["animal"] = animal
        r["subtype"] = subtype
        r["storage"] = classify_storage(str(r["path"]))
        ph = str(r.get("partial_hash") or "")
        if ph and seen_hash.get(ph, 0) >= 2:
            buckets["exact_near"].append(r)
        elif "leopard" in fam.lower() or "animal" in fam.lower() or "animal" in animal.lower():
            buckets["very_similar"].append(r)
        elif fam not in ("", "unknown", "plain"):
            buckets["same_family"].append(r)
        else:
            buckets["different"].append(r)
        if "composite" in subtype.lower() or "karış" in subtype.lower() or "mix" in subtype.lower():
            buckets["composite"].append(r)

    plan = [
        ("exact_near", 2),
        ("very_similar", 2),
        ("same_family", 2),
        ("different", 2),
        ("composite", 2),
    ]
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for bucket, need in plan:
        for rec in buckets.get(bucket, []):
            fid = int(rec["id"])
            if fid in used:
                continue
            used.add(fid)
            picked.append(
                {
                    "bucket": bucket,
                    "file_id": fid,
                    "path": rec["path"],
                    "filename": rec.get("filename") or Path(str(rec["path"])).name,
                    "family": rec.get("family") or "",
                    "storage": rec.get("storage") or "unknown",
                    "source_id": int(rec.get("source_id") or 0),
                }
            )
            if sum(1 for p in picked if p["bucket"] == bucket) >= need:
                break
    if len(picked) < n:
        for rec in existing:
            fid = int(rec["id"])
            if fid in used:
                continue
            used.add(fid)
            picked.append(
                {
                    "bucket": "fallback",
                    "file_id": fid,
                    "path": rec["path"],
                    "filename": rec.get("filename") or Path(str(rec["path"])).name,
                    "family": rec.get("family") or "",
                    "storage": classify_storage(str(rec["path"])),
                    "source_id": int(rec.get("source_id") or 0),
                }
            )
            if len(picked) >= n:
                break
    return picked[:n]


def _cpu_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "threads": threading.active_count(),
        "cpu_percent": None,
        "process_cpu_percent": None,
    }
    try:
        import psutil

        proc = psutil.Process()
        info["cpu_percent"] = psutil.cpu_percent(interval=None)
        info["process_cpu_percent"] = proc.cpu_percent(interval=None)
        info["num_threads"] = proc.num_threads()
        info["cpu_count"] = psutil.cpu_count()
    except Exception:
        pass
    return info


def _run_one(engine: SearchEngine, path: str, *, k: int | None = None) -> dict[str, Any]:
    settings = engine.settings
    prev_k = int(getattr(settings, "prefilter_faiss_top_k", 800) or 800)
    if k is not None:
        settings.prefilter_faiss_top_k = int(k)
    cpu0 = _cpu_snapshot()
    t0 = time.perf_counter()
    first_cb: dict[str, float] = {}

    def _cb(results, done, total):
        if "ttfr_cb_ms" not in first_cb and results:
            first_cb["ttfr_cb_ms"] = (time.perf_counter() - t0) * 1000.0

    engine.start_latency_profile()
    try:
        resp = engine.execute_search(
            SearchQuery(mode="image", image_path=path),
            result_callback=_cb,
        )
    finally:
        if k is not None:
            settings.prefilter_faiss_top_k = prev_k
    wall_ms = (time.perf_counter() - t0) * 1000.0
    lat = dict(resp.meta.get("latency") or {})
    spans = dict(lat.get("spans_ms") or {})
    counts = dict(lat.get("counts") or {})
    pre = dict(resp.meta.get("prefilter") or {})
    cpu1 = _cpu_snapshot()
    shown = list(resp.results or [])
    family_n = int(counts.get("family_count") or 0)
    ranking_ms = (
        float(spans.get("scoring") or 0)
        + float(spans.get("sort") or 0)
        + float(spans.get("textile_rerank") or 0)
    )
    family_ms = (
        float(spans.get("rank.family") or 0)
        + float(spans.get("rank.dna") or 0)
        + float(spans.get("family_overrides") or 0)
        + float(spans.get("family_display") or 0)
    )
    return {
        "wall_ms": round(wall_ms, 3),
        "budget": classify_budget(wall_ms),
        "ttfr_ms": counts.get("ttfr_ms") or first_cb.get("ttfr_cb_ms"),
        "ttfr_cb_ms": first_cb.get("ttfr_cb_ms"),
        "query_embedding_source": counts.get("query_embedding_source"),
        "query_storage": counts.get("query_storage") or counts.get("query_storage"),
        "query_nas_reread": bool(counts.get("query_nas_reread")),
        "clip_map_unavailable": bool(counts.get("clip_map_unavailable")),
        "uvi_in_image_path": bool(counts.get("uvi_in_image_path")),
        "candidates_scored": int(counts.get("candidates_scored") or pre.get("final_candidate_count") or 0),
        "prefilter": {
            "total_index": int(pre.get("total_index") or 0),
            "final": int(pre.get("final_candidate_count") or 0),
            "faiss_dino": int(pre.get("faiss_dino_candidates") or 0),
            "faiss_clip": int(pre.get("faiss_clip_candidates") or 0),
            "layers": list(pre.get("prefilter_layers") or []),
        },
        "family_count": family_n,
        "results_shown": int(counts.get("results_shown") or len(shown)),
        "results_total": int(counts.get("results_total") or len(resp.all_results or [])),
        "family_ms": round(family_ms, 3),
        "ranking_ms": round(ranking_ms, 3),
        "spans_ms": spans,
        "top_bottlenecks": list(lat.get("top_bottlenecks") or []),
        "cpu_before": cpu0,
        "cpu_after": cpu1,
        "top1": (
            {
                "file_id": getattr(shown[0], "file_id", None),
                "score": round(float(getattr(shown[0], "score", 0) or 0), 4),
                "filename": getattr(shown[0], "filename", ""),
            }
            if shown
            else None
        ),
        "error": resp.meta.get("error"),
        "faiss_dino_count": resp.meta.get("faiss_dino_count"),
        "faiss_clip_count": resp.meta.get("faiss_clip_count"),
        "indexed_pool_size": resp.meta.get("indexed_pool_size"),
        "top3_spans": list(lat.get("top_bottlenecks") or [])[:5],
    }


def _why_slow(report: dict[str, Any]) -> str:
    warm = [q["warm"]["wall_ms"] for q in report["queries"] if q.get("warm")]
    avg = sum(warm) / len(warm) if warm else 0.0
    spans_sum: dict[str, float] = defaultdict(float)
    nas_hits = 0
    cache_hits = 0
    extracts = 0
    cands = []
    for q in report["queries"]:
        w = q.get("warm") or q.get("cold") or {}
        for k, v in (w.get("spans_ms") or {}).items():
            spans_sum[k] += float(v)
        if w.get("query_nas_reread"):
            nas_hits += 1
        src = w.get("query_embedding_source")
        if src == "db_cache":
            cache_hits += 1
        elif src:
            extracts += 1
        cands.append(int(w.get("candidates_scored") or 0))
    top = sorted(spans_sum.items(), key=lambda kv: kv[1], reverse=True)[:5]
    bottleneck = top[0][0] if top else "unknown"
    avg_c = sum(cands) / len(cands) if cands else 0
    parts = [
        f"Warm ortalama {avg:.0f} ms ({classify_budget(avg)}).",
        f"En pahalı aşama: {bottleneck}.",
        f"Ortalama skorlanan aday: {avg_c:.0f}.",
        f"Sorgu embedding kaynağı: {cache_hits} DB cache, {extracts} extract.",
        f"NAS'tan sorgu görseli yeniden okuma: {nas_hits}/{len(report['queries'])}.",
        "Görsel arama yolunda UVI yok (UVI metin aramasında).",
        "Aday skorlama kaynak dosyayı değil SQLite blob/thumbnail cache kullanır.",
        "CLIP map unavailable ise CLIP FAISS katmanı düşer; DINO devam eder.",
    ]
    if bottleneck.startswith("rank.") or bottleneck == "scoring":
        parts.append(
            "Kök neden: Python'da aday × özellik döngüsü (hash/doku/DNA/DINO/CLIP/family)."
        )
    elif bottleneck in ("prefilter", "faiss_boost"):
        parts.append("Kök neden: FAISS/prefilter aday genişletme.")
    elif bottleneck in ("hash_verify", "query_thumbnail", "query_embed_extract"):
        parts.append("Kök neden: disk/görüntü decode (NAS veya local cache).")
    elif bottleneck == "indexed_pool":
        parts.append("Kök neden: index pool yükleme (SQLite).")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--skip-k-sweep", action="store_true")
    args = ap.parse_args()
    settings = AppSettings.load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path, read_only=True)
    queries = _pick_queries(db, max(3, int(args.queries)))
    QUERY_LIST.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    engine = SearchEngine(settings, load_ai=False)
    engine.start_latency_profile()
    engine.stop_latency_profile()

    results: list[dict[str, Any]] = []
    latest = OUT_DIR / "visual_search_latency_latest.json"
    for i, q in enumerate(queries):
        cold = _run_one(engine, q["path"])
        print(
            f"[{i+1:02d}C] {q['bucket']:14} {q['filename'][:36]:36} "
            f"{cold['wall_ms']:.0f}ms ttfr={cold.get('ttfr_ms')} "
            f"top={cold.get('top3_spans', [])[:3]}",
            flush=True,
        )
        warm = _run_one(engine, q["path"])
        row = {"query": q, "cold": cold, "warm": warm, "index": i}
        results.append(row)
        latest.write_text(
            json.dumps({"partial": True, "queries": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{i+1:02d}W] {q['bucket']:14} {q['filename'][:36]:36} "
            f"{warm['wall_ms']:.0f}ms ttfr={warm.get('ttfr_ms')} "
            f"top={warm.get('top3_spans', [])[:3]}",
            flush=True,
        )

    k_rows = []
    if queries and not args.skip_k_sweep:
        probe = queries[0]["path"]
        for k in K_SWEEP:
            row = _run_one(engine, probe, k=k)
            k_rows.append({"k": k, **row})
            print(
                f"[K={k:4d}] wall={row['wall_ms']:.0f}ms ranking={row['ranking_ms']:.0f}ms "
                f"cand={row['candidates_scored']}",
                flush=True,
            )

    cold_total = [r["cold"]["wall_ms"] for r in results]
    warm_total = [r["warm"]["wall_ms"] for r in results]
    warm_ttfr = [float(r["warm"]["ttfr_ms"]) for r in results if r["warm"].get("ttfr_ms") is not None]
    warm_family = [r["warm"]["family_ms"] for r in results]
    warm_rank = [r["warm"]["ranking_ms"] for r in results]
    warm_cand = [r["warm"]["candidates_scored"] for r in results]

    span_totals: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for k, v in (r["warm"].get("spans_ms") or {}).items():
            span_totals[k].append(float(v))
    global_bottlenecks = sorted(
        ((k, sum(vs) / len(vs)) for k, vs in span_totals.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )[:10]

    k_scale = None
    if len(k_rows) >= 2:
        a, b = k_rows[0], k_rows[-1]
        c0 = max(1, int(a["candidates_scored"]))
        c1 = max(1, int(b["candidates_scored"]))
        r0 = max(0.001, float(a["ranking_ms"]))
        r1 = max(0.001, float(b["ranking_ms"]))
        k_scale = {
            "k_from": a["k"],
            "k_to": b["k"],
            "candidate_ratio": round(c1 / c0, 3),
            "ranking_ratio": round(r1 / r0, 3),
            "note": "prefilter max_candidates = max(200, prefilter_faiss_top_k); K<200 hâlâ ~200 aday skorlar.",
        }

    n_index = int(results[0]["warm"].get("indexed_pool_size") or 0) if results else 0
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_files": n_index,
        "faiss_dino": results[0]["warm"].get("faiss_dino_count") if results else None,
        "faiss_clip": results[0]["warm"].get("faiss_clip_count") if results else None,
        "query_list_path": str(QUERY_LIST),
        "ui_measured": False,
        "queries": results,
        "k_sweep": k_rows,
        "k_scale": k_scale,
        "stats": {
            "total_cold": summarize_ms(cold_total),
            "total_warm": summarize_ms(warm_total),
            "ttfr_warm": summarize_ms(warm_ttfr),
            "family_warm": summarize_ms(warm_family),
            "ranking_warm": summarize_ms(warm_rank),
            "candidates_warm": {
                "avg": round(sum(warm_cand) / len(warm_cand), 1) if warm_cand else 0,
                "min": min(warm_cand) if warm_cand else 0,
                "max": max(warm_cand) if warm_cand else 0,
            },
        },
        "top_bottlenecks_warm_avg_ms": [
            {"stage": k, "ms": round(v, 3)} for k, v in global_bottlenecks
        ],
        "scale_116k": {
            "current_n": n_index,
            "scoring_is_ocandidates_not_oindex": True,
            "likely_bottleneck_at_116k": (
                "FAISS retrieval + prefilter over 116k, then scoring of capped candidates "
                f"(~{max(warm_cand) if warm_cand else 800}). "
                "If cap stays ~800, ranking grows little; FAISS ntotal grows."
            ),
        },
    }
    report["root_cause"] = _why_slow(report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"visual_search_latency_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = OUT_DIR / "visual_search_latency_latest.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print("WARM TOTAL", report["stats"]["total_warm"])
    print("TTFR", report["stats"]["ttfr_warm"])
    print("FAMILY", report["stats"]["family_warm"])
    print("RANKING", report["stats"]["ranking_warm"])
    print("CANDIDATES", report["stats"]["candidates_warm"])
    print("TOP", report["top_bottlenecks_warm_avg_ms"][:5])
    print("WHY:", report["root_cause"])
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
