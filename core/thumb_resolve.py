"""Thumbnail path çözümleme — kart, inspector ve scheduler ortak kaynak."""

from __future__ import annotations

import os
from pathlib import Path


def lookup_valid_preview(source_path: str, cache_dir: str) -> str:
    """Return valid Preview Pool path for source, or "". Does not open source/NAS."""
    src = str(source_path or "").strip()
    cache = str(cache_dir or "").strip()
    if not src or not cache:
        return ""
    try:
        from core.preview_cache import FeaturePreviewCache

        existing = FeaturePreviewCache(cache).get_existing(src)
        if existing.success and existing.preview_path:
            return str(existing.preview_path)
    except Exception:
        return ""
    return ""


def resolve_thumb_path(
    thumb_path: str,
    *,
    cache_dir: str = "",
    feature_preview_path: str = "",
    project_root: str = "",
    source_path: str = "",
) -> tuple[str, str]:
    """Return (absolute_path_or_empty, status).

    status: hit | miss | empty

    Prefer existing thumbnail; fall back to a valid Preview Pool artifact.
    Does not open the NAS source file.
    """
    raw = str(thumb_path or "").strip()
    fp = str(feature_preview_path or "").strip()
    if not fp and source_path and cache_dir:
        fp = lookup_valid_preview(source_path, cache_dir)
    if not raw and not fp:
        return "", "empty"

    cache = Path(cache_dir) if cache_dir else None
    root = Path(project_root) if project_root else (cache.parent if cache else None)
    name = Path(raw).name if raw else ""
    stem = Path(raw).stem if raw else ""

    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            if root is not None:
                candidates.append(root / raw)
            if cache is not None:
                candidates.append(cache / "thumbnails" / name)
                candidates.append(cache / raw)
    if fp:
        fpp = Path(fp)
        if fpp.is_absolute():
            candidates.append(fpp)
        elif root is not None:
            candidates.append(root / fp)
        if cache is not None and fp:
            candidates.append(cache / "feature_previews" / Path(fp).name)
    if cache is not None and stem:
        candidates.append(cache / "thumbnails" / f"{stem}.webp")
        candidates.append(cache / "thumbnails" / f"{stem}.jpg")
        candidates.append(cache / "feature_previews" / f"{stem}_fp.webp")

    for cand in candidates:
        try:
            if cand.is_file() and cand.stat().st_size > 0:
                return str(cand.resolve()), "hit"
        except OSError:
            continue
    return "", "miss"


def source_file_exists(path: str) -> bool:
    p = str(path or "").strip()
    if not p:
        return False
    try:
        return os.path.isfile(p) and os.path.getsize(p) > 0
    except OSError:
        return False
