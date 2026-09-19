"""Öğretim sonrası görsel embedding yenileme — tek dosya, hızlı."""

from __future__ import annotations

import os
from typing import Iterable

from core.db import Database
from core.indexer import Indexer
from core.logger import setup_logger
from core.manual_label_guard import merge_texture_map_preserve_manual, parse_texture_map
from core.settings import AppSettings

logger = setup_logger(__name__)


def _feature_source_path(rec: dict) -> str:
    fp = rec.get("feature_preview_path", "") or ""
    if fp and os.path.isfile(fp):
        return fp
    thumb = rec.get("thumbnail_path", "") or ""
    if thumb and os.path.isfile(thumb):
        return thumb
    path = rec.get("path", "") or ""
    return path if path and os.path.isfile(path) else ""


def refresh_visual_after_teach(
    settings: AppSettings,
    file_ids: Iterable[int],
) -> dict[str, int]:
    """
    Öğretilen dosyalar için DINO/CLIP + patch yenile.
    Manuel kategori etiketleri korunur; FAISS güncellenir.
    """
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not ids:
        return {"processed": 0, "errors": 0, "embeddings": 0}

    db = Database(settings.db_path)
    indexer = Indexer(settings)
    prev_skip = settings.index_skip_ai
    prev_ai = settings.ai_embedding_enabled
    stats = {"processed": 0, "errors": 0, "embeddings": 0}

    try:
        settings.index_skip_ai = False
        if settings.ai_embedding_enabled:
            indexer._ensure_extractor(use_ai=True, fast_hash_only=False)
        else:
            indexer._ensure_extractor(use_ai=False, fast_hash_only=False)

        for fid in ids:
            rec = db.get_file_by_id(fid) or {}
            src = _feature_source_path(rec)
            if not src:
                stats["errors"] += 1
                continue
            path = rec.get("path", "") or src
            filename = rec.get("filename", "") or os.path.basename(path)
            try:
                existing_feat = db.get_features(fid) or {}
                old_tm = parse_texture_map(existing_feat.get("texture_map"))
                features = indexer.extractor.extract_from_path(src, source_path=path)
                merged_tm = merge_texture_map_preserve_manual(
                    old_tm, features.texture_map
                )

                db.upsert_features(
                    fid,
                    {
                        "phash": features.phash,
                        "dhash": features.dhash,
                        "whash": features.whash,
                        "color_hist": features.color_hist,
                        "dominant_colors": features.dominant_colors,
                        "texture_features": features.texture_features,
                        "dino_embedding": features.dino_embedding,
                        "clip_embedding": features.clip_embedding,
                        "patch_embeddings_meta": features.patch_embeddings_meta,
                        "texture_map": merged_tm,
                    },
                )

                if settings.ai_embedding_enabled:
                    if features.dino_embedding:
                        indexer.faiss.add_dino(fid, features.dino_embedding)
                        stats["embeddings"] += 1
                    if features.clip_embedding:
                        indexer.faiss.add_clip(fid, features.clip_embedding)

                indexer._update_text_index(
                    fid, filename, path, rec.get("ocr_text", "") or "", merged_tm
                )
                stats["processed"] += 1
            except Exception as exc:
                logger.warning("Teach reindex hatası file_id=%s: %s", fid, exc)
                stats["errors"] += 1

        if stats["processed"]:
            indexer.faiss.save()
    finally:
        settings.index_skip_ai = prev_skip
        settings.ai_embedding_enabled = prev_ai
        indexer._ensure_extractor(
            use_ai=settings.index_uses_ai(),
            fast_hash_only=settings.fast_hash_only,
        )

    return stats
