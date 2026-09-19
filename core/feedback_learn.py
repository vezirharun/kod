"""Kullanıcı düzeltmelerini kalıcı öğrenme — DB + kategori ağacı."""

from __future__ import annotations

from core.category_learning import (
    apply_manual_category,
    collect_exact_duplicate_targets,
)
from core.category_tree import (
    category_path,
    pattern_fields_for_path,
)
from core.db import Database
from core.indexer import Indexer
from core.settings import AppSettings
from core.similarity_tiers import tier_label
from core.user_feedback import (
    ACTION_CUSTOM_TAG,
    ACTION_LABEL_FAMILY,
    ACTION_LABEL_NOT_FAMILY,
    ACTION_WRONG,
    UserFeedbackStore,
)


def _refresh_pattern_dna_on_texture_map(
    texture_map: dict,
    *,
    source: str = "user_feedback",
) -> dict:
    from core.classification_pipeline import (
        apply_identity_to_texture_map,
        resolve_pattern_identity,
    )
    from core.pattern_dna import apply_dna_to_texture_map, build_pattern_dna
    from core.semantic_tags import build_semantic_tags

    tm = dict(texture_map)
    if "semantic_tags" not in tm:
        tm["semantic_tags"] = build_semantic_tags(tm)
    identity = resolve_pattern_identity(
        texture_map=tm,
        manual_category_path=str(tm.get("manual_category_path") or ""),
        category_path=str(tm.get("category_path") or ""),
    )
    tm = apply_identity_to_texture_map(tm, identity)
    dna = build_pattern_dna(tm, source=source)
    dna["source"] = source
    dna["confidence"] = max(float(dna.get("confidence") or 0), 0.95)
    return apply_dna_to_texture_map(tm, dna)


def _persist_tier_fields(
    settings: AppSettings,
    file_ids: set[int],
    *,
    pattern_family: str = "",
    tag: str = "",
    animal_print_type: str = "",
    pattern_subtype: str = "",
    similarity_tier: str = "",
    tier_label_text: str = "",
) -> None:
    db = Database(settings.db_path)
    indexer = Indexer(settings)
    try:
        from core.index_freeze import in_search_session

        if in_search_session():
            from core.search_memory import record_feedback_overlay

            for fid in file_ids:
                record_feedback_overlay(
                    settings.db_path,
                    tag or pattern_family or "",
                    int(fid),
                    "teach",
                    label=tag or pattern_family,
                    extra={
                        "pattern_family": pattern_family,
                        "animal_print_type": animal_print_type,
                        "pattern_subtype": pattern_subtype,
                    },
                )
            return
    except Exception:
        pass
    for fid in file_ids:
        detailed = db.get_indexed_files_by_ids(
            [fid],
            include_processing_ready=True,
            include_inactive_sources=True,
        )
        rec = db.get_file_by_id(fid) or {}
        tm = dict((detailed[0].get("texture_map") if detailed else {}) or {})
        if pattern_family:
            tm["pattern_family"] = pattern_family
            tm["classification_confidence"] = 1.0
            tm["user_labeled"] = True
            tm["user_label_source"] = "feedback"
            if pattern_family != "animal_print" and not animal_print_type:
                tm["animal_print_type"] = ""
        if animal_print_type:
            tm["animal_print_type"] = animal_print_type
        if pattern_subtype:
            tm["pattern_subtype"] = pattern_subtype
        if similarity_tier:
            tm["user_similarity_tier"] = similarity_tier
        if tier_label_text:
            tm["user_tier_label"] = tier_label_text
        if tag:
            tm["user_custom_tag"] = tag
        if tm:
            db.upsert_texture_map(fid, tm)
            tm = _refresh_pattern_dna_on_texture_map(tm, source="user_feedback")
            db.upsert_texture_map(fid, tm)
        pf = pattern_family or tm.get("pattern_family", "")
        pt = pattern_subtype or tm.get("pattern_subtype", "")
        if pf or pt:
            db.update_file_text_index(
                fid,
                pattern_family=pf,
                pattern_type=animal_print_type or pt,
                texture_family=pf,
                pattern_subtype=pt,
                pattern_confidence=(
                    1.0
                    if pattern_family
                    else float(rec.get("pattern_confidence", 0) or 0)
                ),
            )
        indexer._update_text_index(
            fid,
            rec.get("filename", ""),
            rec.get("path", ""),
            rec.get("ocr_text", "") or "",
            tm,
        )


def apply_wrong_match_correction(
    settings: AppSettings,
    *,
    query_path: str,
    result_file_id: int,
    pattern_family: str = "",
    tag: str = "",
    reject_query_family: str = "",
    animal_print_type: str = "",
    pattern_subtype: str = "",
    similarity_tier: str = "",
    tier_label_text: str = "",
    cluster_group: str = "",
    category_path: str = "",
    parent_category: str = "",
    child_category: str = "",
    propagate_exact: bool = True,
) -> dict[str, int | str]:
    """Yanlış eşleşme + doğru kategori; yalnızca exact kopyalara yayılır."""
    db = Database(settings.db_path)
    store = UserFeedbackStore(db)

    # Yanlış eşleşme, yalnızca seçilen kartı değil, aynı görselin exact
    # kopyalarını da aynı sorgu bağlamında dışarı almalıdır. Böylece
    # Leopard aramasında bir çiçeği "bu çiçek" diye düzelttiğimizde,
    # aynı çiçeğin başka format/kopyaları da Leopard listesini tekrar
    # kirletmez. Bu kayıt GLOBAL bir kategori değişimi değildir; yalnızca
    # bu sorguya karşı negatif kanıttır.
    wrong_targets = (
        collect_exact_duplicate_targets(db, result_file_id)
        if propagate_exact
        else {int(result_file_id)}
    )
    wrong_targets.add(int(result_file_id))
    for target_id in wrong_targets:
        store.record(query_path, target_id, ACTION_WRONG)
        if reject_query_family:
            store.record(
                query_path,
                target_id,
                ACTION_LABEL_NOT_FAMILY,
                label=reject_query_family,
            )

    path = (category_path or "").strip()
    if not path and parent_category:
        path = category_path(parent_category, child_category)
    # Freeform tag is a custom label, not a category name.
    if not path and tag:
        try:
            from core.concept_registry import learn as learn_concept

            learn_concept(
                settings.db_path,
                tag,
                file_id=int(result_file_id),
                concept_type="custom_tag",
            )
        except Exception:
            pass

    fields = pattern_fields_for_path(path) if path else {}
    pattern_family = pattern_family or fields.get("pattern_family", "")
    pattern_subtype = pattern_subtype or fields.get("pattern_subtype", "")
    animal_print_type = animal_print_type or fields.get("animal_print_type", "")

    if pattern_family:
        store.record(
            query_path, result_file_id, ACTION_LABEL_FAMILY, label=pattern_family
        )
    if tag:
        store.record(query_path, result_file_id, ACTION_CUSTOM_TAG, label=tag)
    try:
        rec = db.get_file_by_id(int(result_file_id)) or {}
        from core.motif_gt_store import ingest_from_wrong_match

        ingest_from_wrong_match(
            file_id=int(result_file_id),
            path=str(rec.get("path") or ""),
            filename=str(rec.get("filename") or ""),
            query_path=query_path,
            category_path=path,
            pattern_family=pattern_family,
            animal_print_type=animal_print_type,
            reject_query_family=reject_query_family,
            tag=tag,
            db_path=settings.db_path,
        )
    except Exception:
        pass
    if similarity_tier or tier_label_text:
        store.record_similarity_tier(
            query_path,
            result_file_id,
            similarity_tier,
            tier_label=tier_label_text or tier_label(similarity_tier),
        )

    if path:
        cat_stats = apply_manual_category(
            settings,
            result_file_id,
            path,
            custom_tag=tag,
            propagate_exact=propagate_exact,
            refresh_visual=False,
        )
        label_text = tier_label_text or tier_label(similarity_tier)
        if similarity_tier or label_text:
            targets = (
                collect_exact_duplicate_targets(db, result_file_id)
                if propagate_exact
                else {int(result_file_id)}
            )
            _persist_tier_fields(
                settings,
                targets,
                similarity_tier=similarity_tier or cluster_group,
                tier_label_text=label_text,
            )
        return {
            "targets": int(cat_stats.get("updated", 1)),
            "propagated_extra": int(cat_stats.get("propagated", 0)),
            "category_path": str(cat_stats.get("category_path", path)),
            "visual_refresh": cat_stats.get("visual_refresh") or {},
        }

    targets = wrong_targets
    _persist_tier_fields(
        settings,
        targets,
        pattern_family=pattern_family,
        tag=tag,
        animal_print_type=animal_print_type,
        pattern_subtype=pattern_subtype,
        similarity_tier=similarity_tier or cluster_group,
        tier_label_text=tier_label_text or tier_label(similarity_tier),
    )
    return {
        "targets": len(targets),
        "propagated_extra": max(0, len(targets) - 1),
        "category_path": "",
    }


def persist_learned_family(
    settings: AppSettings,
    file_id: int,
    *,
    pattern_family: str = "",
    tag: str = "",
    animal_print_type: str = "",
    pattern_subtype: str = "",
) -> None:
    """Geriye uyumluluk — label_family butonu."""
    _persist_tier_fields(
        settings,
        {int(file_id)},
        pattern_family=pattern_family,
        tag=tag,
        animal_print_type=animal_print_type,
        pattern_subtype=pattern_subtype,
    )
    path = str(tag or "").strip()
    if path.lower().startswith("marka/"):
        try:
            from core.search_memory import teach_file_brand

            teach_file_brand(settings.db_path, int(file_id), path.split("/", 1)[1])
        except Exception:
            pass


def clear_learned_family(settings: AppSettings, file_id: int) -> bool:
    from core.indexer import Indexer

    db = Database(settings.db_path)
    rec = db.get_file_by_id(file_id) or {}
    indexer = Indexer(settings)
    ok = indexer._process_texture_only(
        file_id,
        rec.get("thumbnail_path", "") or rec.get("feature_preview_path", ""),
        rec.get("filename", ""),
        rec.get("path", ""),
    )
    if not ok:
        return False
    features = db.get_features(file_id) or {}
    tm = dict(features.get("texture_map") or {})
    for key in (
        "user_similarity_tier",
        "user_tier_label",
        "user_custom_tag",
        "user_labeled",
        "user_label_source",
        "category_path",
        "manual_category_path",
        "category_aliases",
    ):
        tm.pop(key, None)
    if tm:
        db.upsert_texture_map(file_id, tm)
    db.update_file_category(
        file_id,
        category_path="",
        manual_category_path="",
        category_confidence=0.0,
        category_source="",
        category_aliases=[],
    )
    indexer._update_text_index(
        file_id,
        rec.get("filename", ""),
        rec.get("path", ""),
        rec.get("ocr_text", "") or "",
        tm,
    )
    return True
