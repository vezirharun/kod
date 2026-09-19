"""Multi query-image search helpers. Does not change DINO/CLIP/FAISS/OWL."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from core.search_models import SearchQuery, SearchResponse, SearchStats
from core.settings import SUPPORTED_EXTENSIONS

_QUERY_SUFFIXES = frozenset(
    {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".psd", ".ai"}
) | {str(x).lower() for x in SUPPORTED_EXTENSIONS}


def collect_query_images(paths: Iterable[str]) -> list[str]:
    """Keep existing image files; skip missing, duplicates, and non-images."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths or ():
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text)
        if not path.is_file():
            continue
        if path.suffix.lower() not in _QUERY_SUFFIXES:
            continue
        try:
            key = str(path.resolve()).casefold()
        except OSError:
            key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(str(path))
    return out


def result_identity(result: Any) -> str:
    try:
        fid = int(getattr(result, "file_id", 0) or 0)
    except (TypeError, ValueError):
        fid = 0
    if fid > 0:
        return f"id:{fid}"
    raw = str(getattr(result, "path", "") or "")
    try:
        return "path:" + str(Path(raw).resolve()).casefold()
    except OSError:
        return "path:" + raw.casefold()


def stamp_query_group(result: Any, *, index: int, total: int, query_path: str) -> None:
    name = Path(query_path).name
    dbg = dict(getattr(result, "debug", {}) or {})
    prev = list(dbg.get("query_sources") or [])
    if name not in prev:
        prev.append(name)
    dbg["query_sources"] = prev
    dbg["query_group_index"] = min(int(dbg.get("query_group_index") or index), index)
    dbg["query_group_total"] = max(int(dbg.get("query_group_total") or 0), total)
    dbg["query_source"] = str(prev[0] if prev else name)
    result.debug = dbg
    tag = f"Sorgu {index}/{total}: {name}"
    reasons = list(getattr(result, "match_explanations", None) or [])
    if tag not in reasons:
        reasons.insert(0, tag)
    result.match_explanations = reasons[:6]


def single_search_sort_key(result: Any) -> tuple:
    """Same order as SearchEngine._engine_sort_key. Does not rescore."""
    dbg = getattr(result, "debug", None) or {}
    layer = str(dbg.get("result_layer") or "")
    score = float(getattr(result, "score", 0) or 0)
    fid = int(getattr(result, "file_id", 0) or 0)
    hier = float(getattr(result, "hierarchy_score", 0) or 0)
    exact = float(dbg.get("exact_score", dbg.get("exact_search_score", 0)) or 0)
    family = float(dbg.get("family_variant_score", dbg.get("pattern_family_score", 0)) or 0)
    if getattr(result, "is_self_match", False):
        return (0, -1.0, -1.0, -score, fid)
    if dbg.get("face_match"):
        face_score = float(dbg.get("face_similarity", 0.0) or 0.0)
        return (1, -face_score, -score, -hier, fid)
    if dbg.get("learned_concept_exact") or dbg.get("user_taught_positive"):
        return (2, -2.0, -score, -hier, fid)
    if dbg.get("learned_concept"):
        return (2, -1.5, -score, -hier, fid)
    if dbg.get("protected_exact") or layer == "same_files":
        return (2, -exact, -score, -hier, fid)
    if layer == "same_pattern_family":
        return (3, -family, -score, -hier, fid)
    if layer == "similar_patterns":
        return (4, -family, -score, -hier, fid)
    return (5, -score, -family, -hier, fid)


def merge_query_batches(
    batches: list[tuple[str, list[Any]]],
    *,
    total: int,
) -> list[Any]:
    """Dedupe by file; keep better single-search rank. No new ranking."""
    merged: dict[str, Any] = {}
    for index, (query_path, rows) in enumerate(batches, 1):
        for row in rows or []:
            stamp_query_group(row, index=index, total=total, query_path=query_path)
            key = result_identity(row)
            prev = merged.get(key)
            if prev is None:
                merged[key] = row
                continue
            stamp_query_group(prev, index=index, total=total, query_path=query_path)
            if single_search_sort_key(row) < single_search_sort_key(prev):
                sources = list((prev.debug or {}).get("query_sources") or [])
                row.debug = dict(getattr(row, "debug", {}) or {})
                row.debug["query_sources"] = sources
                stamp_query_group(row, index=index, total=total, query_path=query_path)
                merged[key] = row
    out = list(merged.values())
    out.sort(key=single_search_sort_key)
    return out


def summarize_multi_search(*, processed: int, matched: int, missed: int) -> str:
    return (
        f"{processed} görsel işlendi — {matched} eşleşti, {missed} bulunamadı"
    )


def run_multi_image_queries(
    engine: Any,
    paths: list[str],
    *,
    base_query: SearchQuery | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    stopped: Callable[[], bool] | None = None,
    on_partial: Callable[[SearchResponse], None] | None = None,
) -> SearchResponse:
    """One execute_search per query image. Failures do not stop the rest."""
    queries = collect_query_images(paths)
    batches: list[tuple[str, list[Any]]] = []
    missed: list[str] = []
    matched = 0
    n = len(queries)
    threshold = float(getattr(base_query, "threshold", 0.5) or 0.5)
    customer = str(getattr(base_query, "customer", "") or "")
    for i, path in enumerate(queries, 1):
        if stopped and stopped():
            break
        if progress:
            progress(i, n, path)
        query = SearchQuery(
            mode="image",
            image_path=path,
            threshold=threshold,
            customer=customer,
            fast_only=bool(getattr(base_query, "fast_only", False)),
            category_path_filter=str(
                getattr(base_query, "category_path_filter", "") or ""
            ),
        )
        try:
            resp = engine.execute_search(query)
            rows = list(
                getattr(resp, "all_results", None)
                or getattr(resp, "results", None)
                or []
            )
        except Exception:
            missed.append(path)
            continue
        if not rows:
            missed.append(path)
            continue
        matched += 1
        batches.append((path, rows))
        if on_partial:
            merged = merge_query_batches(batches, total=n)
            on_partial(
                _response(
                    merged,
                    processed=i,
                    matched=matched,
                    missed=len(missed),
                    total=n,
                    threshold=threshold,
                )
            )
    processed = matched + len(missed)
    merged = merge_query_batches(batches, total=n)
    return _response(
        merged,
        processed=processed,
        matched=matched,
        missed=len(missed),
        total=n,
        threshold=threshold,
        missed_paths=missed,
    )


def _response(
    rows: list[Any],
    *,
    processed: int,
    matched: int,
    missed: int,
    total: int,
    threshold: float,
    missed_paths: list[str] | None = None,
) -> SearchResponse:
    return SearchResponse(
        results=list(rows),
        all_results=list(rows),
        stats=SearchStats(
            above_threshold=len(rows),
            displayed=len(rows),
            candidates_evaluated=len(rows),
            query_path="",
        ),
        meta={
            "mode": "image",
            "multi_image": True,
            "multi_processed": processed,
            "multi_matched": matched,
            "multi_missed": missed,
            "multi_total": total,
            "multi_summary": summarize_multi_search(
                processed=processed, matched=matched, missed=missed
            ),
            "missed_paths": list(missed_paths or []),
            "threshold": threshold,
        },
    )
