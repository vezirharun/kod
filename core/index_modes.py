"""Index çalışma modları — Hızlı / Gece / Eksikleri tamamla."""

from __future__ import annotations

from enum import Enum
from typing import Any

from core.settings import AppSettings


class IndexMode(str, Enum):
    FAST_ARCHIVE = "fast_archive"  # Hızlı Index
    NIGHT_COMPLETE = "night_complete"  # Derin AI Analiz
    BACKFILL = "backfill"  # Eksikleri Tamamla
    COMPLETE = "complete"  # Hızlı Index + Otomatik Ağır Analiz
    STANDARD = "standard"


def apply_index_mode(settings: AppSettings, mode: IndexMode | str) -> dict[str, Any]:
    """Moda göre geçici index ayarları — orijinali geri yüklemek için snapshot döner."""
    m = IndexMode(mode) if not isinstance(mode, IndexMode) else mode
    snapshot = {
        "index_skip_ai": settings.index_skip_ai,
        "ocr_enabled": settings.ocr_enabled,
        "index_light_first": settings.index_light_first,
        "index_defer_preview": settings.index_defer_preview,
        "index_heavy_after_light": settings.index_heavy_after_light,
        "index_skip_heavy_formats": getattr(settings, "index_skip_heavy_formats", True),
        "fast_hash_only": settings.fast_hash_only,
        "index_mode": getattr(settings, "index_mode", "standard"),
    }
    settings.index_mode = m.value
    if m == IndexMode.FAST_ARCHIVE:
        # Hızlı: thumb + medium preview + metadata — AI/OCR/hash yok
        settings.index_skip_ai = True
        settings.ocr_enabled = False
        settings.index_light_first = True
        settings.index_defer_preview = False
        settings.index_heavy_after_light = False
        settings.index_skip_heavy_formats = True
        settings.fast_hash_only = True
    elif m == IndexMode.NIGHT_COMPLETE:
        # Derin AI: DNA + semantic + OCR + texture + patch
        settings.index_skip_ai = False
        settings.index_light_first = False
        settings.index_defer_preview = False
        settings.index_heavy_after_light = False
        settings.index_skip_heavy_formats = False
        settings.fast_hash_only = False
    elif m == IndexMode.BACKFILL:
        settings.index_skip_ai = not settings.ai_embedding_enabled
        settings.index_light_first = True
        settings.index_defer_preview = True
        settings.index_heavy_after_light = False
        settings.index_skip_heavy_formats = False
    elif m == IndexMode.COMPLETE:
        # Ana index: önizleme → embedding → AI → semantic → DNA → OCR → texture
        settings.index_skip_ai = not settings.ai_embedding_enabled
        settings.index_light_first = True
        settings.index_defer_preview = False
        settings.index_heavy_after_light = False
        settings.index_skip_heavy_formats = False
        settings.fast_hash_only = False
    return snapshot


def restore_index_settings(settings: AppSettings, snapshot: dict[str, Any]) -> None:
    for key, val in snapshot.items():
        if hasattr(settings, key):
            setattr(settings, key, val)
