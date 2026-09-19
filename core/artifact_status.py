"""RC2 Artifact-First SSOT — gerçek artifact durumu, sparse plan, reconcile.

light_done / heavy_done tek başına tamamlanma kanıtı değildir.
Tamamlanma = zorunlu artifact'ların DB (ve Light için fiziksel) varlığı.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from core.index_analysis_guards import (
    texture_map_has_pattern_dna,
    texture_map_has_semantic_tags,
)
from core.manual_label_guard import parse_texture_map

LIGHT_ARTIFACTS = ("thumbnail", "preview", "hash", "metadata")
HEAVY_ARTIFACTS = ("dino", "clip", "texture", "semantic", "dna", "patch")
VISION_ARTIFACTS = frozenset({"dino", "clip", "texture", "patch", "hash"})


@dataclass
class ArtifactReport:
    file_id: int
    light: dict[str, bool] = field(default_factory=dict)
    heavy: dict[str, bool] = field(default_factory=dict)
    preview_ready: bool = False
    light_complete: bool = False
    ai_final: bool = False
    heavy_status: str = ""
    light_status: str = ""

    @property
    def missing_light(self) -> list[str]:
        return [k for k in LIGHT_ARTIFACTS if not self.light.get(k)]

    @property
    def missing_heavy(self) -> list[str]:
        return [k for k in HEAVY_ARTIFACTS if not self.heavy.get(k)]


def _blob_ok(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, memoryview)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _json_list_ok(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        s = value.strip()
        return s not in ("", "[]", "null", "None")
    return False


def assess_row(
    file_row: dict[str, Any] | None,
    feat_row: dict[str, Any] | None,
    *,
    require_physical: bool = False,
) -> ArtifactReport:
    """Tek dosya artifact gerçeği (DB satırlarından)."""
    f = file_row or {}
    fe = feat_row or {}
    fid = int(f.get("id") or 0)
    tm = parse_texture_map(fe.get("texture_map"))

    thumb_path = str(f.get("thumbnail_path") or "").strip()
    preview_path = str(f.get("feature_preview_path") or "").strip()
    thumb_ok = bool(thumb_path)
    preview_ok = bool(preview_path)
    if require_physical:
        phys_t = f.get("physical_thumbnail_ready")
        phys_p = f.get("physical_preview_ready")
        if phys_t is not None:
            thumb_ok = thumb_ok and int(phys_t or 0) == 1
        if phys_p is not None:
            preview_ok = preview_ok and int(phys_p or 0) == 1

    hash_ok = bool(str(fe.get("phash") or "").strip())
    meta_ok = bool(str(f.get("format_metadata") or "").strip()) or bool(
        str(f.get("width") or "")
    )

    light = {
        "thumbnail": thumb_ok,
        "preview": preview_ok,
        "hash": hash_ok,
        "metadata": meta_ok,
    }
    heavy = {
        "dino": _blob_ok(fe.get("dino_embedding")),
        "clip": _blob_ok(fe.get("clip_embedding")),
        "texture": _json_list_ok(fe.get("texture_features")) and hash_ok,
        "semantic": texture_map_has_semantic_tags(tm),
        "dna": texture_map_has_pattern_dna(tm),
        "patch": _json_list_ok(
            fe.get("patch_embeddings_meta")
            if fe.get("patch_embeddings_meta") is not None
            else fe.get("patch_embeddings")
        ),
    }
    # Preview AI'nin tek giriş kapısıdır; Thumbnail UI artifactidir.
    preview_ready = preview_ok
    # Hızlı index tamamlanması bütün light prosedürleri gerektirir.
    light_complete = all(light.values())
    ai_final = preview_ready and all(heavy.values())
    return ArtifactReport(
        file_id=fid,
        light=light,
        heavy=heavy,
        preview_ready=preview_ready,
        light_complete=light_complete,
        ai_final=ai_final,
        heavy_status=str(f.get("heavy_status") or ""),
        light_status=str(f.get("light_status") or ""),
    )


def assess_file(
    db: Any, file_id: int, *, require_physical: bool = False
) -> ArtifactReport:
    row = db.get_file_by_id(int(file_id)) or {}
    feat = db.get_features(int(file_id)) or {}
    return assess_row(row, feat, require_physical=require_physical)


def sparse_heavy_plan(report: ArtifactReport) -> list[str]:
    """Yalnız eksik Heavy artifact adları — boşsa iş yok."""
    if not report.preview_ready:
        return []
    return list(report.missing_heavy)


def vision_compute_set(missing: Iterable[str]) -> set[str]:
    """extract_from_array'e verilecek compute kümesi."""
    miss = set(missing)
    out: set[str] = set()
    if "dino" in miss:
        out.add("dino")
    if "clip" in miss:
        out.add("clip")
    if "texture" in miss or "hash" in miss:
        out.update({"texture", "hash", "color"})
    if "patch" in miss:
        out.add("patch")
    return out


def reconcile_source_artifacts(
    db: Any,
    source_id: int = 0,
    *,
    limit: int = 50000,
) -> dict[str, int]:
    """DONE status artifact kaybını gizleyemez — demote + needs_* güncelle.

    - Preview eksik → needs_medium_preview=1, heavy pending
    - Heavy done ama AI artifact eksik → mark_heavy_incomplete
    """
    stats = {
        "scanned": 0,
        "preview_requeued": 0,
        "heavy_incomplete": 0,
        "already_ok": 0,
    }
    sql = """
        SELECT id FROM files
        WHERE status NOT IN ('excluded_internal','missing')
          AND light_status='done'
    """
    params: list[Any] = []
    if int(source_id or 0) > 0:
        sql += " AND source_id=?"
        params.append(int(source_id))
    sql += " ORDER BY id LIMIT ?"
    params.append(int(limit))

    with db.connect() as conn:
        ids = [int(r["id"]) for r in conn.execute(sql, params).fetchall()]

    for fid in ids:
        stats["scanned"] += 1
        report = assess_file(db, fid, require_physical=True)
        if not report.preview_ready:
            db.update_physical_readiness(
                fid,
                thumbnail_ready=bool(report.light.get("thumbnail")),
                preview_ready=bool(report.light.get("preview")),
                requeue_missing=True,
            )
            stats["preview_requeued"] += 1
            continue
        if report.ai_final:
            stats["already_ok"] += 1
            continue
        if report.missing_heavy:
            db.mark_heavy_incomplete(fid)
            stats["heavy_incomplete"] += 1
    return stats


def features_from_existing(feat: dict[str, Any]) -> dict[str, Any]:
    """Mevcut features satırını upsert_features dict'ine çevir."""
    import json

    tm = parse_texture_map(feat.get("texture_map"))
    meta = feat.get("patch_embeddings_meta")
    if meta is None:
        raw = feat.get("patch_embeddings")
        if isinstance(raw, list):
            meta = raw
        elif isinstance(raw, str) and raw.strip() not in ("", "[]"):
            try:
                meta = json.loads(raw)
            except Exception:
                meta = []
        else:
            meta = []
    return {
        "phash": feat.get("phash") or "",
        "dhash": feat.get("dhash") or "",
        "whash": feat.get("whash") or "",
        "color_hist": feat.get("color_hist"),
        "dominant_colors": feat.get("dominant_colors") or [],
        "texture_features": feat.get("texture_features") or [],
        "dino_embedding": feat.get("dino_embedding"),
        "clip_embedding": feat.get("clip_embedding"),
        "patch_embeddings_meta": meta or [],
        "texture_map": tm,
    }
