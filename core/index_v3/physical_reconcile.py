"""Fiziksel cache SSOT — path/bayrak ile disk gerçekliğini hizala."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.index_analysis_guards import local_artifact_exists
from core.settings import DEFAULT_CACHE_DIR
from core.utils import file_id_from_path


def _first_existing(*candidates: str) -> str:
    for raw in candidates:
        path = str(raw or "").strip()
        if path and local_artifact_exists(path):
            return path
    return ""


def _cache_dirs(cache_dir: str | None) -> list[str]:
    """Ayar cache'i + proje cache'i. İki kopya varsa ikisini de tara."""
    out: list[str] = []
    for raw in (cache_dir, str(DEFAULT_CACHE_DIR)):
        p = str(raw or "").strip()
        if p and p not in out:
            out.append(p)
    return out


def _cache_candidates(source_path: str, cache_dir: str | None) -> tuple[list[str], list[str]]:
    if not source_path:
        return [], []
    fid = file_id_from_path(source_path)
    thumbs: list[str] = []
    previews: list[str] = []
    for cache in _cache_dirs(cache_dir):
        root = Path(cache)
        thumbs.extend(
            [
                str(root / "thumbnails" / f"{fid}.webp"),
                str(root / "thumbnails" / f"{fid}.jpg"),
            ]
        )
        previews.extend(
            [
                str(root / "feature_previews" / f"{fid}_fp.webp"),
                str(root / "feature_previews" / f"{fid}_fp.jpg"),
            ]
        )
    return thumbs, previews


def reconcile_stale_physical_flags(
    db: Any,
    source_ids: list[int] | None,
    *,
    limit: int | None = None,
    cache_dir: str | None = None,
) -> int:
    """physical_* bayraklarını disk/cache gerçekliğiyle hizala.

    source_ids=[] → no-op.
    source_ids=None → tüm dosyalar (yalnız bilinçli/diagnostic).
    cache_dir verilirse boş path'ler mevcut thumbnail/preview cache'ine bağlanır.
    Returns: güncellenen dosya sayısı.
    """
    if source_ids is not None and not source_ids:
        return 0

    where = "f.status NOT IN ('excluded_internal','missing')"
    params: list[Any] = []
    if source_ids is not None:
        ph = ",".join("?" * len(source_ids))
        where += f" AND f.source_id IN ({ph})"
        params.extend(int(x) for x in source_ids)

    where += (
        " AND ((ifnull(f.thumbnail_path,'')!='' AND ifnull(f.physical_thumbnail_ready,0)!=1)"
        " OR (ifnull(f.feature_preview_path,'')!='' AND ifnull(f.physical_preview_ready,0)!=1)"
        " OR ifnull(f.physical_thumbnail_ready,0)=1"
        " OR ifnull(f.physical_preview_ready,0)=1"
        " OR ifnull(f.thumbnail_path,'')=''"
        " OR ifnull(f.feature_preview_path,'')='')"
    )
    sql = f"""
        SELECT f.id, f.path, f.thumbnail_path, f.feature_preview_path,
               ifnull(f.physical_thumbnail_ready,0) AS pt,
               ifnull(f.physical_preview_ready,0) AS pp
        FROM files f
        WHERE {where}
        ORDER BY f.id
    """
    if limit is not None and int(limit) > 0:
        sql += f" LIMIT {int(limit)}"

    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    changed = 0
    for row in rows:
        source_path = str(row.get("path") or "").strip()
        cache_thumbs, cache_previews = _cache_candidates(source_path, cache_dir)
        thumb = _first_existing(str(row.get("thumbnail_path") or ""), *cache_thumbs)
        prev = _first_existing(str(row.get("feature_preview_path") or ""), *cache_previews)
        thumb_ok = bool(thumb)
        prev_ok = bool(prev)
        was_t = int(row.get("pt") or 0) == 1
        was_p = int(row.get("pp") or 0) == 1
        old_thumb = str(row.get("thumbnail_path") or "").strip()
        old_prev = str(row.get("feature_preview_path") or "").strip()
        path_changed = (thumb and thumb != old_thumb) or (prev and prev != old_prev)
        if was_t == thumb_ok and was_p == prev_ok and not path_changed:
            continue

        if path_changed:
            payload: dict[str, Any] = {"path": source_path}
            if thumb and thumb != old_thumb:
                payload["thumbnail_path"] = thumb
            if prev and prev != old_prev:
                payload["feature_preview_path"] = prev
            if len(payload) > 1:
                db.upsert_file(payload)

        # Bayrak hizala; light/heavy_status'u toplu pending yapma.
        # Eksik artifact'ler planner tarafından sparse kuyruğa alınır.
        db.update_physical_readiness(
            int(row["id"]),
            thumbnail_ready=thumb_ok,
            preview_ready=prev_ok,
            requeue_missing=False,
        )
        changed += 1
    return changed
