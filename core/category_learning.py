"""Manuel kategori öğrenme — exact duplicate yayılımı, text index güncelleme."""

from __future__ import annotations

import json

from core.category_tree import (
    aliases_for_path,
    category_path,
    metadata_for_path,
    pattern_fields_for_path,
    resolve_category_query,
)
from core.db import Database
from core.indexer import Indexer
from core.settings import AppSettings
from core.teach_reindex import refresh_visual_after_teach
from core.text_index import build_text_search_blob

EXACT_RELATIONS = frozenset(
    {
        "exact",
        "format_variant",
        "resolution_variant",
        "crop_variant",
    }
)


def build_full_category_path(
    parent: str,
    child: str = "",
    custom_tag: str = "",
) -> str:
    base = category_path(parent, child) if parent else ""
    tag = (custom_tag or "").strip()
    if tag and base:
        return f"{base}/{tag}"
    if tag:
        resolved = resolve_category_query(tag)
        if resolved.category_path:
            return resolved.category_path
        return tag
    return base


def collect_exact_duplicate_targets(db: Database, file_id: int) -> set[int]:
    """Yalnızca exact kopyalar — benzer görünümlere kör yayılım yok."""
    targets = {int(file_id)}
    rec = db.get_file_by_id(file_id) or {}
    for duplicate in db.get_identity_candidates(
        partial_hash=rec.get("partial_hash", "") or "",
        full_hash=rec.get("full_hash", "") or "",
    ):
        targets.add(int(duplicate["id"]))
    group = db.get_pattern_group_for_file(file_id)
    if group:
        for member in db.list_pattern_group_members(int(group["id"])):
            relation = (member.get("relation") or "").strip()
            if relation in EXACT_RELATIONS:
                targets.add(int(member["file_id"]))
    return targets


def apply_manual_category(
    settings: AppSettings,
    file_id: int,
    category_path_value: str,
    *,
    custom_tag: str = "",
    propagate_exact: bool = True,
    refresh_visual: bool = True,
) -> dict[str, int]:
    """
    Kullanıcı öğretimi: category_path + pattern alanları + text blob.
    propagate_exact=True ise yalnızca exact duplicate'lara yayılır.
    """
    path = (category_path_value or "").strip().strip("/")
    if custom_tag and path:
        path = f"{path}/{custom_tag.strip()}"
    if not path:
        # Freeform custom_tag is a label, never a category path.
        return {"updated": 0, "propagated": 0}

    try:
        from core.index_freeze import INDEX_FROZEN, in_search_session

        if INDEX_FROZEN or in_search_session():
            from core.search_memory import record_category_overlay

            record_category_overlay(
                settings.db_path,
                int(file_id),
                path,
                extra={"custom_tag": custom_tag},
            )
            if path.lower().startswith("marka/"):
                try:
                    from core.search_memory import teach_file_brand

                    teach_file_brand(
                        settings.db_path, int(file_id), path.split("/", 1)[1]
                    )
                except Exception:
                    pass
            return {
                "updated": 1,
                "propagated": 0,
                "memory_only": True,
                "category_path": path,
            }
    except Exception:
        pass

    meta = metadata_for_path(path, manual=True)
    fields = pattern_fields_for_path(path)
    pattern_family = fields.get("pattern_family", "")
    pattern_subtype = fields.get("pattern_subtype", "")
    animal_print_type = fields.get("animal_print_type", "")
    brand_name = fields.get("brand_name", "")
    aliases = list(meta.get("category_aliases") or aliases_for_path(path))
    if custom_tag and custom_tag not in aliases:
        aliases.append(custom_tag)

    db = Database(settings.db_path)
    target_ids = (
        collect_exact_duplicate_targets(db, file_id)
        if propagate_exact
        else {int(file_id)}
    )

    indexer = Indexer(settings)
    updated = 0
    for fid in target_ids:
        rec = db.get_file_by_id(fid) or {}
        detailed = db.get_indexed_files_by_ids(
            [fid],
            include_processing_ready=True,
            include_inactive_sources=True,
        )
        tm = dict((detailed[0].get("texture_map") if detailed else {}) or {})

        if pattern_family:
            tm["pattern_family"] = pattern_family
            tm["classification_confidence"] = 1.0
            tm["user_labeled"] = True
            tm["user_label_source"] = "manual_user"
        if pattern_subtype:
            tm["pattern_subtype"] = pattern_subtype
        if animal_print_type:
            tm["animal_print_type"] = animal_print_type
        elif pattern_family and pattern_family != "animal_print":
            tm["animal_print_type"] = ""
        if brand_name:
            tm["brand_name"] = brand_name
        tm["category_path"] = path
        tm["manual_category_path"] = path
        tm["category_confidence"] = 1.0
        tm["category_source"] = "manual_user"
        tm["category_aliases"] = aliases
        if custom_tag:
            tm["user_custom_tag"] = custom_tag
        from core.semantic_tags import build_semantic_tags
        from core.pattern_dna import apply_dna_to_texture_map, build_pattern_dna
        from core.category_predictions import category_predictions_from_metadata

        tm["semantic_tags"] = build_semantic_tags(
            tm,
            category_path=path,
            ocr_text=rec.get("ocr_text", "") or "",
            filename=rec.get("filename", ""),
        )
        dna = build_pattern_dna(
            tm,
            semantic_tags=tm.get("semantic_tags"),
            category_path=path,
            source="manual_user",
        )
        tm = apply_dna_to_texture_map(tm, dna)
        tm["training_pool"] = "manual_category"
        tm["training_pool_source_file_id"] = int(file_id)
        tm["ai_category_predictions"] = category_predictions_from_metadata(
            pattern_family,
            animal_print_type or pattern_subtype,
            tm,
        )

        db.upsert_texture_map(fid, tm)
        blob = build_text_search_blob(
            filename=rec.get("filename", ""),
            path=rec.get("path", ""),
            customer=rec.get("customer", ""),
            ocr_text=rec.get("ocr_text", "") or "",
            texture_map=tm,
            pattern_family=pattern_family,
            pattern_subtype=pattern_subtype,
            pattern_type=animal_print_type or pattern_subtype,
            category_path=path,
            category_aliases=aliases,
            feedback_labels=aliases,
            semantic_enabled=bool(
                getattr(settings, "semantic_text_search_enabled", False)
            ),
        )
        db.update_file_category(
            fid,
            category_path=path,
            manual_category_path=path,
            category_confidence=1.0,
            category_source="manual_user",
            category_aliases=aliases,
            pattern_family=pattern_family,
            pattern_subtype=pattern_subtype,
            pattern_type=animal_print_type or pattern_subtype,
            text_search_blob=blob,
        )
        indexer._update_text_index(
            fid,
            rec.get("filename", ""),
            rec.get("path", ""),
            rec.get("ocr_text", "") or "",
            tm,
        )
        updated += 1

    propagated = max(0, len(target_ids) - 1)
    visual_stats = (
        refresh_visual_after_teach(settings, target_ids)
        if refresh_visual
        else {"processed": 0, "embeddings": 0}
    )
    return {
        "updated": updated,
        "propagated": propagated,
        "category_path": path,
        "visual_refresh": visual_stats,
    }
