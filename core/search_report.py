"""Arama debug raporları."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.search_models import SearchResponse
from core.settings import DEFAULT_DATA_DIR, SUPPORTED_EXTENSIONS


def save_search_report(
    response: SearchResponse, extra: dict[str, Any] | None = None
) -> Path:
    reports_dir = DEFAULT_DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"search_debug_{ts}.json"

    all_sorted = sorted(response.all_results, key=lambda r: r.score, reverse=True)
    threshold = float(response.meta.get("threshold", 0.0))
    above = [r for r in all_sorted if r.score >= threshold]
    below = sorted(
        [r for r in all_sorted if r.score < threshold],
        key=lambda r: r.score,
        reverse=True,
    )[:50]

    fmt_dist: dict[str, int] = {}
    reject_reasons: Counter[str] = Counter()
    feature_missing = 0
    thumbnail_missing = 0
    unsupported = 0

    for r in all_sorted:
        ext = Path(r.path).suffix.lower() or "?"
        fmt_dist[ext] = fmt_dist.get(ext, 0) + 1
        if ext not in SUPPORTED_EXTENSIONS and ext != "?":
            unsupported += 1
        if not r.debug.get("has_features"):
            feature_missing += 1
        if not r.debug.get("has_thumbnail"):
            thumbnail_missing += 1
        reason = r.debug.get("reject_reason", "")
        if reason:
            for part in str(reason).split(";"):
                if part:
                    reject_reasons[part.split(":")[0]] += 1

    stats = response.stats
    top_100 = [_result_detail(r) for r in above[:100]]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_path": stats.query_path,
        "query_text": stats.text_query,
        "query_family": response.meta.get("query_pattern_family", ""),
        "query_pattern_family": response.meta.get("query_pattern_family", ""),
        "query_animal_print_type": response.meta.get("query_animal_print_type", ""),
        "query_color_family": response.meta.get("query_color_family", ""),
        "query_texture_summary": response.meta.get("query_texture_summary", {}),
        "used_crop": stats.used_crop,
        "text_query": stats.text_query,
        "selected_sources": response.meta.get("selected_sources", []),
        "active_sources": response.meta.get("selected_sources", []),
        "search_scope": response.meta.get("search_scope", ""),
        "search_mode": response.meta.get("search_mode", ""),
        "color_weight_mode": response.meta.get("color_weight_mode", ""),
        "threshold": threshold,
        "stats": {
            "total_indexed": stats.total_indexed,
            "selected_source_files": stats.selected_source_files,
            "supported_image_files": stats.supported_image_files,
            "thumbnail_files": stats.thumbnail_files,
            "feature_files": stats.feature_files,
            "candidates_scored": stats.candidates_evaluated,
            "prefilter_candidates": getattr(stats, "prefilter_candidates", 0),
            "prefilter_used": getattr(stats, "prefilter_used", False),
            "above_threshold": stats.above_threshold,
            "displayed": stats.displayed,
            "near_below_threshold": stats.near_below_threshold,
            "sources_searched": stats.sources_searched,
            "search_ms": stats.search_ms,
            "indexed_pool_size": response.meta.get("indexed_pool_size", 0),
        },
        "prefilter": response.meta.get("prefilter", {}),
        "ai_faiss_used": bool(
            response.meta.get("ai_embedding_enabled")
            and (
                response.meta.get("faiss_dino_count", 0)
                or response.meta.get("faiss_clip_count", 0)
            )
        ),
        "faiss_dino_count": response.meta.get("faiss_dino_count", 0),
        "faiss_clip_count": response.meta.get("faiss_clip_count", 0),
        "user_feedback_applied": response.meta.get("user_feedback_applied", False),
        "top_100_breakdown": top_100,
        "top_100_score_breakdown": top_100,
        "settings_snapshot": {
            "search_mode": response.meta.get("search_mode"),
            "color_weight_mode": response.meta.get("color_weight_mode"),
            "threshold": threshold,
            "fast_hash_only": response.meta.get("fast_hash_only", False),
        },
        "engine_usage": {
            "fast_hash_only": response.meta.get("fast_hash_only", False),
            "patch_used": response.meta.get("patch_used", False),
            "texture_map_used": response.meta.get("texture_map_used", False),
            "ai_used": response.meta.get("ai_used", False),
        },
        "top_results": [_result_detail(r) for r in above[:20]],
        "top_50_below_threshold": [_result_detail(r) for r in below],
        "format_distribution": fmt_dist,
        "feature_missing_count": feature_missing,
        "thumbnail_missing_count": thumbnail_missing,
        "unsupported_format_count": unsupported,
        "reject_reason_counts": dict(reject_reasons),
        "protected_exact_candidates_count": response.meta.get(
            "protected_exact_candidates_count", 0
        ),
        "hash_candidates_count": (response.meta.get("prefilter") or {}).get(
            "hash_bucket_candidates", 0
        ),
        "patch_candidates_count": (response.meta.get("prefilter") or {}).get(
            "patch_candidates", 0
        ),
        "texture_candidates_count": (response.meta.get("prefilter") or {}).get(
            "texture_family_candidates", 0
        ),
        "text_candidates_count": (response.meta.get("prefilter") or {}).get(
            "text_candidates", 0
        ),
        "family_candidates_count": (response.meta.get("prefilter") or {}).get(
            "texture_family_candidates", 0
        ),
        "final_candidates_count": stats.candidates_evaluated,
        "shown_count": stats.displayed,
        "skipped_by_scope": max(0, stats.total_indexed - stats.selected_source_files),
        "skipped_by_threshold": max(
            0, stats.candidates_evaluated - stats.above_threshold
        ),
        "skipped_by_family": sum(
            1
            for r in all_sorted
            if (getattr(r, "cluster_group", "") or r.category) == "unrelated"
        ),
        "query_confidence": response.meta.get("query_confidence", 0),
        "query_subtype": response.meta.get("query_pattern_subtype", ""),
        "text_family_conflict": response.meta.get("text_family_conflict", False),
        "exact_candidate_found_but_not_shown": _detect_exact_not_shown(
            all_sorted, threshold
        ),
        "per_candidate_reject_sample": [
            {
                "file_id": r.file_id,
                "filename": r.filename,
                "score_percent": r.score_percent,
                "reject_reason": r.debug.get("reject_reason", ""),
            }
            for r in below[:50]
        ],
        "reader_used": "pillow",
        "meta": response.meta,
        "extra": extra or {},
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if payload.get("exact_candidate_found_but_not_shown"):
            crit = reports_dir / f"CRITICAL_exact_lost_{ts}.txt"
            crit.write_text(
                "CRITICAL: exact candidate was found but not displayed\n"
                f"query={payload.get('query_path')}\n",
                encoding="utf-8",
            )
    except OSError:
        reports_dir = Path(tempfile.gettempdir()) / "vezir_pattern_search_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"search_debug_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    save_quality_html_report(payload, reports_dir, ts)
    return path


def save_quality_html_report(
    payload: dict[str, Any], reports_dir: Path, ts: str
) -> Path:
    html_path = reports_dir / f"search_quality_report_{ts}.html"
    stats = payload.get("stats", {})
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats.items())
    top = payload.get("top_results", [])[:10]
    top_rows = "".join(
        f"<tr><td>{r.get('filename', '')}</td><td>{r.get('score_percent', '')}</td>"
        f"<td>{r.get('pattern_family', '')}</td></tr>"
        for r in top
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Search Quality {ts}</title>
<style>body{{font-family:sans-serif;background:#1e1e2e;color:#e2e8f0;padding:20px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #444;padding:6px}}</style>
</head><body>
<h1>Arama Kalite Raporu</h1>
<p>Query: {payload.get("query_path", "")} | Text: {payload.get("text_query", "")}</p>
<h2>İstatistikler</h2><table>{rows}</table>
<h2>Top 10</h2><table><tr><th>Dosya</th><th>Skor</th><th>Family</th></tr>{top_rows}</table>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _detect_exact_not_shown(results: list, threshold: float) -> bool:
    """Korumalı exact aday eşik altında kaldıysa kritik."""
    for r in results:
        if r.debug.get("protected_exact") and r.score < threshold:
            return True
        if r.debug.get("protected_exact") and r.score_percent < 85:
            if not r.debug.get("threshold_passed", True):
                return True
    return False


def _result_detail(r) -> dict[str, Any]:
    return {
        "file_id": r.file_id,
        "filename": r.filename,
        "path": r.path,
        "score_percent": r.score_percent,
        "category": r.category,
        "category_label": r.category_label,
        "cluster_group": getattr(r, "cluster_group", ""),
        "pattern_family": getattr(r, "pattern_family", ""),
        "animal_print_type": getattr(r, "animal_print_type", ""),
        "color_family": getattr(r, "color_family", ""),
        "palette_similarity": getattr(r, "palette_similarity", 0),
        "texture_family_score": getattr(r, "texture_family_score", 0),
        "hierarchy_score": getattr(r, "hierarchy_score", 0),
        "visual_score": r.debug.get("visual_score"),
        "patch_score": r.debug.get("patch_score"),
        "reason": getattr(r, "cluster_reason", "") or r.debug.get("matched_reason", ""),
        "breakdown": r.breakdown,
        "matched_reason": r.debug.get("matched_reason", ""),
        "reject_reason": r.debug.get("reject_reason", ""),
        "debug": {
            k: r.debug.get(k)
            for k in (
                "phash_score",
                "dhash_score",
                "patch_score",
                "texture_score",
                "color_score",
                "filename_score",
                "has_features",
                "has_thumbnail",
                "threshold_passed",
                "cluster_group",
                "pattern_family",
                "animal_print_type",
                "uvi_tier",
                "uvi_reason",
                "object_evidence",
                "uvi_explain",
            )
            if k in r.debug
        },
    }
