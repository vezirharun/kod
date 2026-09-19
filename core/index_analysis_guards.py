"""Index tekrar analiz ve karantina atlama kuralları."""

from __future__ import annotations

import os
from typing import Any, Protocol

from core.manual_label_guard import parse_texture_map
from core.network_index_throttle import is_network_path, track_local_cache_read
from core.quarantine import is_hard_quarantine, is_soft_quarantine


class FeaturePreviewLookup(Protocol):
    def get_existing(self, source_path: str) -> Any: ...


def texture_map_has_pattern_dna(texture_map: dict[str, Any] | None) -> bool:
    dna = dict((texture_map or {}).get("pattern_dna") or {})
    if not dna:
        return False
    return bool(
        dna.get("motif_family")
        or dna.get("pattern_family")
        or dna.get("family")
        or float(dna.get("confidence") or dna.get("pattern_dna_confidence") or 0) > 0.05
    )


def texture_map_has_semantic_tags(texture_map: dict[str, Any] | None) -> bool:
    sem = dict((texture_map or {}).get("semantic_tags") or {})
    if not sem:
        return False
    for key in ("motifs", "styles", "colors", "materials", "brand_references"):
        if sem.get(key):
            return True
    return float(sem.get("confidence") or 0) > 0.05


def local_artifact_exists(path: str, *, stats: dict | None = None) -> bool:
    """Yerel önbellek yolu — ağ orijinallerine asla dokunmaz.

    DB'de göreli `cache\\thumbnails\\...` yolları cwd'ye bağlı kalmasın diye
    cache_dir köküne göre de denenir.
    """
    path = str(path or "").strip()
    if not path:
        return False
    lowered = path.lower().replace("/", "\\")
    is_cache_artifact = any(
        token in lowered
        for token in (
            "\\thumbnails\\",
            "/thumbnails/",
            "\\previews\\",
            "feature_preview",
            ".v3_cache",
        )
    )
    # Önbellek NAS'ta olsa da disk kontrolü yapılır; aksi halde her thumb
    # thumb_verify_failed → 2 deneme → kalıcı hata (indeks ilerlemez).
    if is_network_path(path) and not is_cache_artifact:
        return False
    candidates = [path]
    remapped = ""
    try:
        from core.path_safety import remap_legacy_storage_path

        remapped = remap_legacy_storage_path(path)
        if remapped and remapped != path:
            candidates.append(remapped)
    except Exception:
        remapped = ""
    if not os.path.isabs(path):
        try:
            from core.settings import AppSettings

            cache_dir = str(AppSettings().cache_dir or "").strip()
            if cache_dir:
                root = os.path.dirname(os.path.abspath(cache_dir))
                candidates.append(os.path.join(root, path))
                # cache\thumbnails\x.webp → {cache_dir}\thumbnails\x.webp
                norm = path.replace("/", os.sep)
                prefix = "cache" + os.sep
                if norm.lower().startswith(prefix):
                    candidates.append(
                        os.path.join(cache_dir, norm[len(prefix) :])
                    )
                candidates.append(os.path.join(cache_dir, os.path.basename(path)))
        except Exception:
            pass
    for cand in candidates:
        if cand and os.path.isfile(cand):
            track_local_cache_read(stats)
            return True
    return False


def has_thumbnail_artifact(record: dict[str, Any] | None, *, stats: dict | None = None) -> bool:
    if not record:
        return False
    thumb = str(record.get("thumbnail_path") or "").strip()
    return local_artifact_exists(thumb, stats=stats)


def resolve_medium_preview_path(
    record: dict[str, Any] | None,
    *,
    source_path: str = "",
    feature_preview_cache: FeaturePreviewLookup | None = None,
    stats: dict | None = None,
) -> str:
    if not record:
        record = {}
    fp = str(record.get("feature_preview_path") or "").strip()
    if local_artifact_exists(fp, stats=stats):
        return fp
    if feature_preview_cache and source_path:
        disk = feature_preview_cache.get_existing(source_path)
        if getattr(disk, "success", False):
            preview_path = str(getattr(disk, "preview_path", "") or "")
            if local_artifact_exists(preview_path, stats=stats):
                return preview_path
    return ""


def has_medium_preview(
    record: dict[str, Any] | None,
    *,
    source_path: str = "",
    feature_preview_cache: FeaturePreviewLookup | None = None,
    stats: dict | None = None,
) -> bool:
    return bool(
        resolve_medium_preview_path(
            record,
            source_path=source_path,
            feature_preview_cache=feature_preview_cache,
            stats=stats,
        )
    )


def needs_medium_preview(
    record: dict[str, Any] | None,
    *,
    source_path: str = "",
    feature_preview_cache: FeaturePreviewLookup | None = None,
    stats: dict | None = None,
) -> bool:
    if not has_thumbnail_artifact(record, stats=stats):
        return False
    return not has_medium_preview(
        record,
        source_path=source_path,
        feature_preview_cache=feature_preview_cache,
        stats=stats,
    )


def has_preview_artifacts(record: dict[str, Any] | None, *, stats: dict | None = None) -> bool:
    """Light pass tamamlandı mı — en az thumbnail veya medium preview."""
    if has_thumbnail_artifact(record, stats=stats):
        return True
    fp = str((record or {}).get("feature_preview_path") or "").strip()
    return local_artifact_exists(fp, stats=stats)


def has_preview_artifacts_for_deep(
    record: dict[str, Any] | None,
    *,
    source_path: str = "",
    feature_preview_cache: FeaturePreviewLookup | None = None,
    stats: dict | None = None,
) -> bool:
    """Deep AI — medium preview zorunlu."""
    return has_medium_preview(
        record,
        source_path=source_path,
        feature_preview_cache=feature_preview_cache,
        stats=stats,
    )


def is_quarantined_file(record: dict[str, Any] | None) -> bool:
    """Sert karantina — index tekrar denemez. Yumuşak karantina aranabilir kalır."""
    if not record:
        return False
    reason = str(record.get("quarantine_reason") or "").strip()
    if is_soft_quarantine(reason) and has_preview_artifacts(record):
        return False
    if is_hard_quarantine(reason):
        return True
    if str(record.get("status") or "") != "error":
        return False
    if has_preview_artifacts(record):
        return False
    if int(record.get("needs_review") or 0):
        return True
    stage = str(record.get("index_stage") or "")
    return stage in ("needs_review", "failed")


def features_have_deep_analysis(feat: dict[str, Any] | None, *, ocr_text: str = "") -> bool:
    tm = parse_texture_map((feat or {}).get("texture_map"))
    has_dna = texture_map_has_pattern_dna(tm)
    has_sem = texture_map_has_semantic_tags(tm)
    has_ocr = bool((ocr_text or "").strip())
    return has_dna and has_sem and has_ocr
