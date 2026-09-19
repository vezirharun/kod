"""Ranking Trace Analyzer — Search Engine'e dokunmadan skor izi.

Kullanim:
  python scripts/ranking_trace_analyzer.py
  python scripts/ranking_trace_analyzer.py --expect-id 130165
  python scripts/ranking_trace_analyzer.py --query "C:\\Users\\HARUN\\Desktop\\Screenshot_7.jpg"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import Database
from core.faiss_store import FaissStore
from core.feature_extractor import FeatureExtractor
from core.search_engine import SearchEngine
from core.search_models import SearchQuery
from core.settings import AppSettings


SCORE_KEYS = (
    "phash",
    "dhash",
    "whash",
    "color",
    "texture",
    "patch",
    "dna",
    "semantic",
    "dino",
    "clip",
    "faiss_boost",
    "filename_text",
    "ocr_text",
)


def _round(x: Any, n: int = 4) -> float:
    try:
        return round(float(x or 0), n)
    except (TypeError, ValueError):
        return 0.0


def pack_result(r: Any, rank: int) -> dict[str, Any]:
    b = dict(getattr(r, "breakdown", None) or {})
    d = dict(getattr(r, "debug", None) or {})
    scores = {k: _round(b.get(k)) for k in SCORE_KEYS}
    scores["knowledge"] = _round(d.get("knowledge_score"))
    scores["exact"] = _round(d.get("exact_score", d.get("exact_search_score")))
    scores["family_variant"] = _round(
        d.get("family_variant_score", d.get("pattern_family_score"))
    )
    scores["hierarchy"] = _round(getattr(r, "hierarchy_score", 0) or d.get("hierarchy_score"))
    scores["visual"] = _round(d.get("visual_score"))
    scores["final"] = _round(getattr(r, "score", 0))
    return {
        "rank": rank,
        "file_id": int(r.file_id),
        "filename": r.filename,
        "path": r.path,
        "score": _round(r.score),
        "score_percent": _round(r.score_percent, 1),
        "pattern_family": getattr(r, "pattern_family", "") or "",
        "cluster_group": getattr(r, "cluster_group", "") or "",
        "category": getattr(r, "category", "") or "",
        "reason": getattr(r, "reason", "") or d.get("matched_reason", ""),
        "is_self_match": bool(getattr(r, "is_self_match", False) or d.get("is_self_match")),
        "result_layer": d.get("result_layer"),
        "protected_exact": bool(d.get("protected_exact")),
        "same_family": bool(d.get("same_family")),
        "sort_key_hint": {
            "layer": d.get("result_layer"),
            "exact": _round(d.get("exact_score", d.get("exact_search_score"))),
            "family": _round(d.get("family_variant_score", d.get("pattern_family_score"))),
            "final": _round(r.score),
        },
        "scores": scores,
        "breakdown_raw": {k: _round(v) for k, v in b.items() if isinstance(v, (int, float))},
        "debug_selected": {
            k: d.get(k)
            for k in (
                "result_layer",
                "protected_exact",
                "tier_a_hash",
                "tier_b",
                "exact_score",
                "family_variant_score",
                "pattern_family_score",
                "knowledge_score",
                "knowledge_weight",
                "phash_score",
                "texture_score",
                "dino_score",
                "clip_score",
                "dna_score",
                "hierarchy_score",
                "visual_score",
                "final_score",
                "reject_reason",
                "family_gate",
                "similarity_tier",
            )
            if k in d
        },
        "family_breakdown": d.get("family_breakdown") or {},
        "contribution_scores": d.get("contribution_scores") or {},
    }


def faiss_trace(
    settings: AppSettings,
    db: Database,
    query_path: str,
    expect_id: int,
    top_k: int = 800,
) -> dict[str, Any]:
    """Aday üretiminde FAISS sırasını gerçek embedding ile ölç."""
    out: dict[str, Any] = {
        "expect_in_dino_map": False,
        "expect_in_clip_map": False,
        "dino_rank": None,
        "dino_score": None,
        "clip_rank": None,
        "clip_score": None,
        "dino_top5": [],
        "clip_top5": [],
    }
    feat_q = None
    # Query features from DB if indexed, else extract
    qrec = db.get_file_by_path(str(query_path))
    if qrec:
        feat_q = db.get_features(int(qrec["id"]))
    dino_q = (feat_q or {}).get("dino_embedding") or b""
    clip_q = (feat_q or {}).get("clip_embedding") or b""
    if not dino_q or not clip_q:
        # lightweight: use Feature/FE only if needed
        try:
            from core.thumbnailer import Thumbnailer
            import numpy as np

            img = Thumbnailer.load_image(str(query_path))
            if img is not None:
                fx = FeatureExtractor(settings)
                # Prefer existing extract API if present
                if hasattr(fx, "extract_embeddings"):
                    emb = fx.extract_embeddings(img)
                    dino_q = emb.get("dino") or dino_q
                    clip_q = emb.get("clip") or clip_q
        except Exception as exc:
            out["extract_error"] = str(exc)

    fs = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    out["expect_in_dino_map"] = expect_id in fs.dino_id_map
    out["expect_in_clip_map"] = expect_id in fs.clip_id_map
    out["dino_ntotal"] = int(getattr(fs.dino_index, "ntotal", 0) or 0)
    out["clip_ntotal"] = int(getattr(fs.clip_index, "ntotal", 0) or 0)

    import numpy as np

    if dino_q and fs.dino_index is not None:
        vec = np.frombuffer(dino_q, dtype=np.float32).reshape(1, -1)
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
        k = min(top_k, fs.dino_index.ntotal)
        scores, idxs = fs.dino_index.search(vec, k)
        pairs = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(fs.dino_id_map):
                continue
            fid = int(fs.dino_id_map[idx])
            pairs.append((fid, float(score)))
            if fid == expect_id and out["dino_rank"] is None:
                out["dino_rank"] = len(pairs)
                out["dino_score"] = round(float(score), 4)
        out["dino_top5"] = [
            {"file_id": fid, "score": round(sc, 4)} for fid, sc in pairs[:5]
        ]

    if clip_q and fs.clip_index is not None:
        vec = np.frombuffer(clip_q, dtype=np.float32).reshape(1, -1)
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
        k = min(top_k, fs.clip_index.ntotal)
        scores, idxs = fs.clip_index.search(vec, k)
        pairs = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(fs.clip_id_map):
                continue
            fid = int(fs.clip_id_map[idx])
            pairs.append((fid, float(score)))
            if fid == expect_id and out["clip_rank"] is None:
                out["clip_rank"] = len(pairs)
                out["clip_score"] = round(float(score), 4)
        out["clip_top5"] = [
            {"file_id": fid, "score": round(sc, 4)} for fid, sc in pairs[:5]
        ]

    return out


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    sa = a.get("scores") or {}
    sb = b.get("scores") or {}
    deltas = {}
    for k in sorted(set(sa) | set(sb)):
        deltas[k] = round(float(sa.get(k) or 0) - float(sb.get(k) or 0), 4)
    # biggest absolute gaps where expected is behind (negative delta)
    behind = sorted(
        ((k, v) for k, v in deltas.items() if v < 0),
        key=lambda x: x[1],
    )
    ahead = sorted(
        ((k, v) for k, v in deltas.items() if v > 0),
        key=lambda x: -x[1],
    )
    return {
        "deltas_expect_minus_ref": deltas,
        "largest_deficits": [{"metric": k, "delta": v} for k, v in behind[:8]],
        "largest_advantages": [{"metric": k, "delta": v} for k, v in ahead[:8]],
        "layer_expect": a.get("result_layer"),
        "layer_ref": b.get("result_layer"),
        "rank_expect": a.get("rank"),
        "rank_ref": b.get("rank"),
        "final_expect": a.get("score"),
        "final_ref": b.get("score"),
        "final_delta": round(float(a.get("score") or 0) - float(b.get("score") or 0), 4),
    }


def diagnose(
    expect: dict[str, Any] | None,
    top1: dict[str, Any],
    top_nonself: dict[str, Any] | None,
    faiss: dict[str, Any],
    prefilter: dict[str, Any] | None,
) -> dict[str, Any]:
    if not expect:
        return {
            "root_cause": "Beklenen dosya sonuç listesinde yok (aday üretimi veya eşik).",
            "largest_score_gap": None,
            "algorithm_to_fix": "candidate_generation_or_threshold",
            "expected_improvement": "Önce aday havuzuna girmesini sağla.",
        }

    # Primary comparison: top1 if not self, else first non-self
    ref = top1
    ref_label = "top1"
    if top1.get("is_self_match") or top1.get("protected_exact"):
        if top_nonself:
            ref = top_nonself
            ref_label = "top_nonself"
        else:
            ref_label = "top1_self"

    cmp = compare(expect, ref)
    deficits = cmp["largest_deficits"]
    biggest = deficits[0] if deficits else None

    # Layer effect: how many same_files / same_pattern_family ahead
    layer_penalty = False
    if expect.get("result_layer") in ("other_results", "similar_patterns", None):
        if ref.get("result_layer") in ("same_files", "same_pattern_family"):
            layer_penalty = True

    # Self-match occupies #1
    self_at_top = bool(top1.get("is_self_match") or top1.get("protected_exact"))

    root_parts: list[str] = []
    if self_at_top:
        root_parts.append(
            f"Top-1 = sorgu dosyasının kendisi (protected_exact/self), skor={top1.get('score_percent')}%"
        )
    if layer_penalty:
        root_parts.append(
            f"Beklenen result_layer='{expect.get('result_layer')}', "
            f"referans result_layer='{ref.get('result_layer')}' "
            f"(layer sıralaması final skorundan önce geliyor)"
        )
    if biggest:
        root_parts.append(
            f"Referans ({ref_label} id={ref.get('file_id')}) karşısında en büyük açık: "
            f"{biggest['metric']} delta={biggest['delta']}"
        )
    root_parts.append(
        f"Final skor: expect={expect.get('score')} vs {ref_label}={ref.get('score')} "
        f"(delta={cmp['final_delta']})"
    )
    if faiss.get("dino_rank") is not None:
        root_parts.append(
            f"FAISS DINO sırası={faiss.get('dino_rank')} skor={faiss.get('dino_score')}; "
            f"CLIP sırası={faiss.get('clip_rank')} skor={faiss.get('clip_score')}"
        )

    # Decide single algorithm
    algo = "unknown"
    if layer_penalty and (
        not biggest
        or abs(float(biggest["delta"])) < abs(float(cmp["final_delta"])) * 0.5
        or expect.get("result_layer") == "other_results"
        and ref.get("result_layer") == "same_files"
    ):
        # If both are same_files, layer not the issue
        if expect.get("result_layer") != ref.get("result_layer"):
            algo = "result_layer_sort / cluster_gate"
    if biggest:
        metric = biggest["metric"]
        mapping = {
            "patch": "patch_similarity / patch_pattern_score",
            "texture": "texture_similarity",
            "dino": "DINO embedding similarity (ranking weight)",
            "clip": "OpenCLIP embedding similarity (ranking weight)",
            "faiss_boost": "FAISS boost aggregation",
            "dna": "pattern_dna_similarity",
            "semantic": "semantic_tag_similarity",
            "color": "color_similarity",
            "phash": "perceptual hash (exact engine)",
            "dhash": "dhash",
            "exact": "exact_search_score / protected_exact path",
            "family_variant": "pattern_family_score",
            "knowledge": "knowledge_graph reason_knowledge",
            "ocr_text": "OCR text score",
            "final": "final score fusion",
        }
        if algo == "unknown" or (
            biggest and abs(float(biggest["delta"])) >= 0.05
        ):
            # Prefer concrete metric if deficit is meaningful
            if biggest and abs(float(biggest["delta"])) >= 0.05:
                algo = mapping.get(metric, metric)

    # Expected improvement heuristic based on measured gap only
    improvement = (
        f"Referans sırasına yaklaşmak için {algo} düzeltilmeli; "
        f"ölçülen final fark {cmp['final_delta']} "
        f"(expect %{expect.get('score_percent')} vs {ref_label} %{ref.get('score_percent')})."
    )
    if self_at_top:
        improvement += (
            " Not: Gerçek Top-1 self-match; pratik hedef ilk non-self sırasıdır "
            f"(şu an #{top_nonself.get('rank') if top_nonself else '?'})."
        )

    return {
        "root_cause": " | ".join(root_parts),
        "comparison_reference": ref_label,
        "reference_file_id": ref.get("file_id"),
        "expect_rank": expect.get("rank"),
        "expect_score": expect.get("score"),
        "largest_score_gap": biggest,
        "all_deficits": deficits,
        "layer_effect": {
            "expect_layer": expect.get("result_layer"),
            "ref_layer": ref.get("result_layer"),
            "layer_pushed_down": layer_penalty,
        },
        "self_match_occupies_top1": self_at_top,
        "algorithm_to_fix": algo,
        "expected_improvement": improvement,
        "prefilter_final_candidates": (prefilter or {}).get("final_candidate_count"),
        "candidate_generation": {
            "faiss_dino_candidates": (prefilter or {}).get("faiss_dino_candidates"),
            "faiss_clip_candidates": (prefilter or {}).get("faiss_clip_candidates"),
            "hash_bucket_candidates": (prefilter or {}).get("hash_bucket_candidates"),
            "protected_exact_candidates": (prefilter or {}).get(
                "protected_exact_candidates"
            ),
            "layers": (prefilter or {}).get("prefilter_layers"),
        },
        "compare": cmp,
    }


def print_report(report: dict[str, Any]) -> None:
    w = 72
    print("=" * w)
    print("  RANKING TRACE ANALYZER")
    print("=" * w)
    print(f"Query: {report.get('query')}")
    print(f"Elapsed: {report.get('elapsed_ms')} ms | results: {report.get('n_results')}")
    print("-" * w)
    cg = report.get("candidate_generation") or {}
    print("CANDIDATE GENERATION")
    for k, v in cg.items():
        print(f"  {k}: {v}")
    ft = report.get("faiss_trace") or {}
    print(
        f"  FAISS expect DINO rank/score: {ft.get('dino_rank')} / {ft.get('dino_score')}"
    )
    print(
        f"  FAISS expect CLIP rank/score: {ft.get('clip_rank')} / {ft.get('clip_score')}"
    )
    print("-" * w)

    def block(title: str, row: dict[str, Any] | None) -> None:
        print(title)
        if not row:
            print("  (yok)")
            return
        print(
            f"  #{row['rank']} id={row['file_id']}  {row['filename'][:55]}"
        )
        print(
            f"  final={row['score_percent']}%  family={row['pattern_family']}  "
            f"layer={row['result_layer']}  cluster={row['cluster_group']}"
        )
        sc = row.get("scores") or {}
        print(
            "  "
            + " | ".join(
                f"{k}={sc.get(k)}"
                for k in (
                    "faiss_boost",
                    "dino",
                    "clip",
                    "texture",
                    "dna",
                    "color",
                    "patch",
                    "semantic",
                    "ocr_text",
                    "knowledge",
                    "final",
                )
            )
        )

    block("TOP-1", report.get("top1"))
    block("TOP NON-SELF", report.get("top_nonself"))
    block("EXPECTED", report.get("expected"))
    print("-" * w)
    d = report.get("diagnosis") or {}
    print("ROOT CAUSE")
    print(f"  {d.get('root_cause')}")
    print("EN BUYUK SKOR FARKI")
    print(f"  {d.get('largest_score_gap')}")
    print("DUZELTILMESI GEREKEN TEK ALGORITMA")
    print(f"  {d.get('algorithm_to_fix')}")
    print("BEKLENEN IYILESME")
    print(f"  {d.get('expected_improvement')}")
    print("=" * w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default=r"C:\Users\HARUN\Desktop\Screenshot_7.jpg",
    )
    parser.add_argument("--expect-id", type=int, default=130165)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    settings = AppSettings.load()
    db = Database(settings.db_path)
    engine = SearchEngine(settings, load_ai=True)

    t0 = time.perf_counter()
    resp = engine.execute_search(
        SearchQuery(
            mode="image",
            image_path=str(args.query),
            threshold=float(args.threshold),
            fast_only=False,
        )
    )
    elapsed = (time.perf_counter() - t0) * 1000
    results = list(resp.results or resp.all_results or [])
    packed = [pack_result(r, i + 1) for i, r in enumerate(results)]

    top1 = packed[0] if packed else None
    top_nonself = next(
        (
            r
            for r in packed
            if not r.get("is_self_match") and not r.get("protected_exact")
        ),
        None,
    )
    # If all protected, fall back to first non-self path match
    if top_nonself is None:
        top_nonself = next((r for r in packed if not r.get("is_self_match")), None)

    expected = next((r for r in packed if r["file_id"] == int(args.expect_id)), None)

    # How many ahead in same_files layer
    ahead_same_files = 0
    ahead_higher_score = 0
    if expected:
        for r in packed:
            if r["file_id"] == expected["file_id"]:
                break
            if r.get("result_layer") == "same_files":
                ahead_same_files += 1
            if float(r.get("score") or 0) > float(expected.get("score") or 0):
                ahead_higher_score += 1

    prefilter = (resp.meta or {}).get("prefilter") or {}
    faiss = faiss_trace(settings, db, str(args.query), int(args.expect_id))

    diagnosis = diagnose(expected, top1 or {}, top_nonself, faiss, prefilter)
    diagnosis["files_ahead_in_same_files_layer"] = ahead_same_files
    diagnosis["files_ahead_with_higher_final_score"] = ahead_higher_score

    # Side-by-side table metrics
    side_by_side = None
    if expected and top1:
        ref = top_nonself if (
            top1.get("is_self_match") or top1.get("protected_exact")
        ) and top_nonself else top1
        keys = [
            "faiss_boost",
            "dino",
            "clip",
            "texture",
            "dna",
            "color",
            "patch",
            "semantic",
            "ocr_text",
            "knowledge",
            "exact",
            "family_variant",
            "final",
        ]
        side_by_side = {
            "columns": ["metric", "expected", "reference", "delta"],
            "reference_label": (
                "top_nonself" if ref is top_nonself else "top1"
            ),
            "rows": [
                {
                    "metric": k,
                    "expected": (expected.get("scores") or {}).get(k),
                    "reference": (ref.get("scores") or {}).get(k),
                    "delta": round(
                        float((expected.get("scores") or {}).get(k) or 0)
                        - float((ref.get("scores") or {}).get(k) or 0),
                        4,
                    ),
                }
                for k in keys
            ],
        }

    report = {
        "query": str(args.query),
        "expect_id": int(args.expect_id),
        "elapsed_ms": round(elapsed, 1),
        "n_results": len(packed),
        "meta": {
            "query_pattern_family": (resp.meta or {}).get("query_pattern_family"),
            "ai_used": (resp.meta or {}).get("ai_used"),
            "faiss_dino_count": (resp.meta or {}).get("faiss_dino_count"),
            "faiss_clip_count": (resp.meta or {}).get("faiss_clip_count"),
            "prefilter": prefilter,
        },
        "candidate_generation": diagnosis.get("candidate_generation"),
        "faiss_trace": faiss,
        "top1": top1,
        "top_nonself": top_nonself,
        "expected": expected,
        "side_by_side": side_by_side,
        "top10": packed[:10],
        "diagnosis": diagnosis,
    }

    print_report(report)
    if args.save:
        out = ROOT / "data" / "reports" / "ranking_trace_130165.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved:", out)


if __name__ == "__main__":
    main()
