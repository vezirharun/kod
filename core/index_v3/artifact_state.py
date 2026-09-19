"""Index Engine V3 — artifact reality SSOT (no light_done/heavy_done truth)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from core.index_analysis_guards import (
    texture_map_has_pattern_dna,
    texture_map_has_semantic_tags,
)
from core.index_v3.types import (
    HEAVY_ARTIFACTS,
    LIGHT_ARTIFACTS,
    OBJECT_CONCEPT_ARTIFACTS,
    OWLV2_ARTIFACTS,
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
)
from core.manual_label_guard import parse_texture_map


def object_index_path_for_patterns_db(db: Any, settings: Any | None = None) -> str:
    """object_index.db: patterns.db ile aynı klasör (test tmp / production data)."""
    settings_obj = str(getattr(settings, "object_db_path", "") or "").strip()
    dbp = str(getattr(db, "db_path", "") or "").strip()
    if dbp and dbp not in {":memory:", ""}:
        sibling = str(Path(dbp).parent / "object_index.db")
        if settings_obj:
            try:
                if Path(settings_obj).parent.resolve() == Path(dbp).parent.resolve():
                    return settings_obj
            except Exception:
                pass
        return sibling
    return settings_obj


def _owlv2_artifact_ready(db: Any, row: dict[str, Any]) -> bool:
    """READY = OCR ocr_processed / Object Index has_scan: Preview havuzunda bir OWL geçişi var."""
    fid = int(row.get("id") or 0)
    if fid <= 0:
        return False
    preview = str(row.get("feature_preview_path") or "").strip()
    if not preview or not os.path.isfile(preview):
        return False
    obj_path = object_index_path_for_patterns_db(db)
    if not obj_path:
        return False
    try:
        from core.object_index import ObjectIndexStore

        store = ObjectIndexStore(obj_path, readonly=True)
        return store.ovd_has_done_scan(fid)
    except Exception:
        return False


def _blob_ok(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, memoryview)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _list_ok(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        s = value.strip()
        if s in ("", "[]", "null", "None"):
            return False
        try:
            parsed = json.loads(s)
            return isinstance(parsed, list) and len(parsed) > 0
        except Exception:
            return False
    return False


def _visual_concept_dna_ok(tm: dict[str, Any]) -> bool:
    blob = tm.get("visual_concept_dna") if isinstance(tm, dict) else None
    if isinstance(blob, dict) and blob:
        return True
    goi = tm.get("global_object_intelligence") if isinstance(tm, dict) else None
    if isinstance(goi, dict):
        inner = goi.get("visual_concept_dna")
        if isinstance(inner, dict) and inner:
            return True
    return False


def assess_rows(
    file_row: dict[str, Any] | None,
    feat_row: dict[str, Any] | None,
    *,
    require_disk: bool = False,
    disk_exists: Callable[[str], bool] | None = None,
) -> FileArtifactReport:
    """Compute READY/MISSING/INVALID from DB rows (+ optional disk)."""
    f = file_row or {}
    fe = feat_row or {}
    fid = int(f.get("id") or 0)
    sid = int(f.get("source_id") or 0)
    path = str(f.get("path") or "")
    tm = parse_texture_map(fe.get("texture_map"))

    thumb = str(f.get("thumbnail_path") or "").strip()
    preview = str(f.get("feature_preview_path") or "").strip()
    thumb_phys = int(f.get("physical_thumbnail_ready") or 0) == 1
    prev_phys = int(f.get("physical_preview_ready") or 0) == 1
    check = disk_exists or (lambda p: bool(p))

    def path_status(p: str, *, physical_ready: bool) -> ArtifactStatus:
        # V3 SSOT: path yetmez — physical_*_ready=1 (UI Tamamlanan ile aynı)
        if not p or not physical_ready:
            return ArtifactStatus.MISSING
        if require_disk and not check(p):
            return ArtifactStatus.INVALID
        return ArtifactStatus.READY

    status: dict[Artifact, ArtifactStatus] = {
        Artifact.THUMBNAIL: path_status(thumb, physical_ready=thumb_phys),
        Artifact.PREVIEW: path_status(preview, physical_ready=prev_phys),
        Artifact.HASH: (
            ArtifactStatus.READY
            if str(fe.get("phash") or "").strip()
            else ArtifactStatus.MISSING
        ),
        Artifact.METADATA: (
            ArtifactStatus.READY
            if (
                str(f.get("format_metadata") or "").strip()
                or int(f.get("width") or 0) > 0
            )
            else ArtifactStatus.MISSING
        ),
        Artifact.DINO: (
            ArtifactStatus.READY
            if _blob_ok(fe.get("dino_embedding"))
            else ArtifactStatus.MISSING
        ),
        Artifact.CLIP: (
            ArtifactStatus.READY
            if _blob_ok(fe.get("clip_embedding"))
            else ArtifactStatus.MISSING
        ),
        Artifact.TEXTURE: (
            ArtifactStatus.READY
            if (
                _list_ok(fe.get("texture_features"))
                and str(fe.get("phash") or "").strip()
            )
            else ArtifactStatus.MISSING
        ),
        Artifact.SEMANTIC: (
            ArtifactStatus.READY
            if texture_map_has_semantic_tags(tm)
            else ArtifactStatus.MISSING
        ),
        Artifact.DNA: (
            ArtifactStatus.READY
            if texture_map_has_pattern_dna(tm)
            else ArtifactStatus.MISSING
        ),
        Artifact.OBJECT_CONCEPT: (
            ArtifactStatus.READY
            if _visual_concept_dna_ok(tm)
            else ArtifactStatus.MISSING
        ),
        Artifact.PATCH: (
            ArtifactStatus.INVALID
            if str(f.get("patch_error") or "").strip()
            else ArtifactStatus.READY
            if _list_ok(
                fe.get("patch_embeddings_meta")
                if fe.get("patch_embeddings_meta") is not None
                else fe.get("patch_embeddings")
            )
            else ArtifactStatus.MISSING
        ),
        # OCR READY = işlendi (ocr_processed=1); text boş olabilir (empty ≠ failed).
        # OCR/PATCH fail INVALID — HASH…DNA / ai_final'e karışmaz.
        Artifact.OCR: (
            ArtifactStatus.READY
            if int(f.get("ocr_processed") or 0) == 1
            else ArtifactStatus.INVALID
            if str(f.get("ocr_error") or "").strip()
            else ArtifactStatus.MISSING
        ),
    }
    return FileArtifactReport(
        file_id=fid,
        source_id=sid,
        path=path,
        status=status,
    )


def assess_file(
    db: Any,
    file_id: int,
    *,
    require_disk: bool = False,
) -> FileArtifactReport:
    from core.index_analysis_guards import local_artifact_exists

    row = db.get_file_by_id(int(file_id)) or {}
    feat = db.get_features(int(file_id)) or {}

    def _exists(p: str) -> bool:
        return local_artifact_exists(p)

    report = assess_rows(
        row,
        feat,
        require_disk=require_disk,
        disk_exists=_exists if require_disk else None,
    )
    report.status[Artifact.OWLV2] = (
        ArtifactStatus.READY
        if _owlv2_artifact_ready(db, row)
        else ArtifactStatus.MISSING
    )
    return report


def plan_missing(
    report: FileArtifactReport,
    *,
    light: bool = True,
    heavy: bool = True,
) -> list[Artifact]:
    """Return only missing artifacts the planner should enqueue."""
    out: list[Artifact] = []
    if light and not report.light_complete:
        # Preview/Thumbnail are Fast; Hash/Metadata belong to General AI
        out.extend(report.missing(LIGHT_ARTIFACTS))
    if heavy and report.preview_ready:
        out.extend(report.missing(HEAVY_ARTIFACTS))
        out.extend(report.missing(OBJECT_CONCEPT_ARTIFACTS))
        out.extend(report.missing(OWLV2_ARTIFACTS))
    # de-dupe preserve order
    seen: set[Artifact] = set()
    ordered: list[Artifact] = []
    for a in out:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered
