"""Two-phase FAST queue: normal first, jumbo (fast_tif_defer_mb) deferred."""

from __future__ import annotations

from typing import Any

# Claim filter uses available_at <= now. Far-future keeps jumbo pending but unclaimable.
JUMBO_AVAILABLE_AT_FAR = 4_102_444_800  # 2100-01-01 UTC
JUMBO_DEFER_MARKER = "jumbo_deferred"

_JUMBO_EXTS = frozenset({".tif", ".tiff"})


def is_jumbo_for_queue(
    path: str,
    file_size: int,
    settings: Any | None = None,
    *,
    defer_mb: int | None = None,
) -> bool:
    """Jumbo = large TIFF using settings.fast_tif_defer_mb (not is_risky_path)."""
    # UNC paths like \\server\share\file.tif must not rely on Path.suffix
    # (Windows may treat the share segment as the final part).
    name = str(path or "").replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in name:
        return False
    ext = "." + name.rsplit(".", 1)[-1].lower()
    if ext not in _JUMBO_EXTS:
        return False
    mb = defer_mb
    if mb is None and settings is not None:
        mb = getattr(settings, "fast_tif_defer_mb", None)
    if mb is None:
        mb = 64
    try:
        threshold = max(0, int(mb)) * 1024 * 1024
    except (TypeError, ValueError):
        threshold = 64 * 1024 * 1024
    try:
        size = int(file_size or 0)
    except (TypeError, ValueError):
        size = 0
    return size >= threshold


def light_artifact_names() -> frozenset[str]:
    return frozenset({"preview", "thumbnail"})
