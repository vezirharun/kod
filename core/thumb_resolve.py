"""Thumbnail path çözümleme — kart, inspector ve scheduler ortak kaynak."""

from __future__ import annotations

import os
from pathlib import Path

# Tiny white EPS/AI stubs are typically ~100–200 bytes; real thumbs are larger.
_MIN_EPS_THUMB_BYTES = 512
_EPS_AI_EXTS = {".eps", ".ai"}


def _is_eps_ai(path: str | Path) -> bool:
    return Path(str(path or "")).suffix.lower() in _EPS_AI_EXTS


def _thumb_usable_for_eps(path: str | Path) -> bool:
    """Reject missing / empty / tiny white stubs; prefer blank/EPS gate when available."""
    p = Path(path)
    try:
        if not p.is_file():
            return False
        size = int(p.stat().st_size)
        if size <= 0 or size < _MIN_EPS_THUMB_BYTES:
            return False
    except OSError:
        return False
    try:
        from core.preview_renderer import _eps_ai_preview_gate

        ok, _reason = _eps_ai_preview_gate(p)
        return bool(ok)
    except Exception:
        try:
            from explorer_preview.host.blank import is_valid_preview_image

            ok, _reason = is_valid_preview_image(p)
            return bool(ok)
        except Exception:
            # Gate unavailable (tests / minimal env): size threshold is enough.
            return True


def is_eps_ai_thumb_usable(path: str | Path) -> bool:
    """Public helper: whether a list/detail thumb artifact is safe for EPS/AI."""
    return _thumb_usable_for_eps(path)


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


def _collect_thumb_candidates(
    raw: str,
    *,
    cache: Path | None,
    root: Path | None,
    stem: str,
    name: str,
) -> list[Path]:
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
    if cache is not None and stem:
        candidates.append(cache / "thumbnails" / f"{stem}.webp")
        candidates.append(cache / "thumbnails" / f"{stem}.jpg")
    return candidates


def _collect_fp_candidates(
    fp: str,
    *,
    cache: Path | None,
    root: Path | None,
    stem: str,
) -> list[Path]:
    candidates: list[Path] = []
    if fp:
        fpp = Path(fp)
        if fpp.is_absolute():
            candidates.append(fpp)
        elif root is not None:
            candidates.append(root / fp)
        if cache is not None:
            candidates.append(cache / "feature_previews" / Path(fp).name)
    if cache is not None and stem:
        candidates.append(cache / "feature_previews" / f"{stem}_fp.webp")
    return candidates


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
    For .eps/.ai: prefer valid feature preview; never serve tiny/white thumbs.
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

    # EPS context: source ext, or FP path that itself looks like an EPS/AI source.
    is_eps = _is_eps_ai(source_path) or _is_eps_ai(fp)

    thumb_cands = _collect_thumb_candidates(
        raw, cache=cache, root=root, stem=stem, name=name
    )
    fp_cands = _collect_fp_candidates(fp, cache=cache, root=root, stem=stem)

    if is_eps:
        # 1) Prefer valid FP (list can show FP when thumb is white stub).
        for cand in fp_cands:
            try:
                if cand.is_file() and cand.stat().st_size > 0 and _thumb_usable_for_eps(cand):
                    return str(cand.resolve()), "hit"
            except OSError:
                continue
        # 2) Only return thumbnail if EPS-usable (not tiny/white).
        for cand in thumb_cands:
            try:
                if _thumb_usable_for_eps(cand):
                    return str(cand.resolve()), "hit"
            except OSError:
                continue
        # 3) Thumb invalid / missing and no usable FP → miss (caller may render).
        return "", "miss"

    # Non-EPS/AI: keep current candidate order (thumb first, then FP).
    candidates: list[Path] = []
    candidates.extend(thumb_cands)
    candidates.extend(fp_cands)
    # Preserve stem FP after stem thumbs (already in lists); original also had
    # stem thumbs then stem fp interleaved via single list — thumb_cands then
    # fp_cands matches: raw thumbs, stem thumbs, fp, stem fp.

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
