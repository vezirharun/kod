"""Thumbnail üzerinden perceptual hash doğrulama — bozuk DB kayıtlarını yakala."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from core.logger import setup_logger
from core.utils import phash_similarity

logger = setup_logger(__name__)

_STALE_SIM_THRESHOLD = 0.82


@dataclass(frozen=True)
class VerifiedHashes:
    phash: str
    dhash: str
    whash: str
    source: str  # stored | thumbnail | feature_preview
    stale: bool = False


def _compute_from_image_path(path: str) -> tuple[str, str, str]:
    try:
        import imagehash
        from PIL import Image

        from core.thumbnailer import Thumbnailer

        with Image.open(path) as img:
            img = Thumbnailer.normalize_pillow_image(img)
            return (
                str(imagehash.phash(img)),
                str(imagehash.dhash(img)),
                str(imagehash.whash(img)),
            )
    except Exception as exc:
        logger.debug("Hash doğrulama başarısız %s: %s", path, exc)
        return "", "", ""


def verified_hashes_for_record(rec: dict[str, Any]) -> VerifiedHashes:
    """DB'deki hash ile önizleme uyumsuzsa önizlemeden yeniden hesapla."""
    stored_p = (rec.get("phash") or "").strip()
    stored_d = (rec.get("dhash") or "").strip()
    stored_w = (rec.get("whash") or "").strip()

    from core.index_freeze import search_write_protection_active

    # B: search sırasında mevcut hash'i oku (hesap/persist yok).
    # A (INDEX_FROZEN) yalnız başına hash doğrulamayı sonsuza kapatmaz.
    if search_write_protection_active():
        return VerifiedHashes(
            phash=stored_p,
            dhash=stored_d,
            whash=stored_w,
            source="stored",
            stale=False,
        )

    preview = (rec.get("feature_preview_path") or "").strip()
    thumb = (rec.get("thumbnail_path") or "").strip()
    # Thumbnail önce — feature_preview bazen yanlış/bozuk cache'ten gelir.
    for label, path in (("thumbnail", thumb), ("feature_preview", preview)):
        if not path or not os.path.isfile(path):
            continue
        live_p, live_d, live_w = _compute_from_image_path(path)
        if not live_p:
            continue
        stale = False
        if stored_p and phash_similarity(stored_p, live_p) < _STALE_SIM_THRESHOLD:
            stale = True
        if stale:
            logger.info(
                "Bozuk hash düzeltildi file_id=%s (%s): %s → %s",
                rec.get("id"),
                rec.get("filename", ""),
                stored_p[:8],
                live_p[:8],
            )
            return VerifiedHashes(
                phash=live_p,
                dhash=live_d or stored_d,
                whash=live_w or stored_w,
                source=label,
                stale=True,
            )
        return VerifiedHashes(
            phash=stored_p or live_p,
            dhash=stored_d or live_d,
            whash=stored_w or live_w,
            source="stored",
            stale=False,
        )

    return VerifiedHashes(
        phash=stored_p,
        dhash=stored_d,
        whash=stored_w,
        source="stored",
        stale=False,
    )


def apply_verified_hashes(rec: dict[str, Any]) -> dict[str, Any]:
    """Aday kaydına doğrulanmış hash alanlarını uygula (kopya)."""
    v = verified_hashes_for_record(rec)
    out = dict(rec)
    out["phash"] = v.phash
    out["dhash"] = v.dhash
    out["whash"] = v.whash
    out["_hash_verified_from"] = v.source
    out["_hash_stale"] = v.stale
    return out
