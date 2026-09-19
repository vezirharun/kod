"""Kullanıcı geri bildirimi — skor öğrenmesi."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.db import Database
from core.logger import setup_logger

logger = setup_logger(__name__)

ACTION_SAME_PATTERN = "same_pattern"
ACTION_SIMILAR_TEXTURE = "similar_texture"
ACTION_WRONG = "wrong"
ACTION_LABEL_FAMILY = "label_family"
ACTION_LABEL_NOT_FAMILY = "label_not_family"
ACTION_CLEAR_FAMILY_LABELS = "clear_family_labels"
ACTION_PROMOTE = "promote"
ACTION_DEMOTE = "demote"
ACTION_CUSTOM_TAG = "custom_tag"
ACTION_SIMILARITY_TIER = "similarity_tier"
ACTION_NOT_SIMILAR = "not_similar"
ACTION_RELATED_FAMILY = "related_family"
ACTION_METADATA_EDIT = "metadata_edit"


def _looks_like_path(value: str) -> bool:
    s = str(value or "")
    if not s.strip():
        return False
    if "\\" in s or s.startswith("/") or (len(s) >= 3 and s[1] == ":"):
        return True
    low = s.lower()
    return any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".psd", ".ai"))


def teach_query_keys(query_text: str, overlay: dict[str, Any] | None = None) -> list[str]:
    """Ranking keys from the search box only. Category/lemmas stay in metadata."""
    from core.search_memory import _norm_query

    del overlay  # stored on the overlay row; never becomes a query_key
    if not query_text or _looks_like_path(query_text):
        return []
    q = _norm_query(query_text)
    if not q or len(q) < 2:
        return []
    return [q]

BOOST_ACTIONS = {ACTION_SAME_PATTERN, ACTION_PROMOTE, ACTION_SIMILAR_TEXTURE}
PENALTY_ACTIONS = {ACTION_WRONG, ACTION_DEMOTE}


class UserFeedbackStore:
    def __init__(self, db: Database):
        self.db = db

    def record(
        self,
        query_path: str,
        result_file_id: int,
        action: str,
        *,
        query_file_id: int = 0,
        label: str = "",
        note: str = "",
    ) -> int:
        from core.index_freeze import in_search_session
        from core.search_memory import record_feedback_overlay

        overlay_id = 0
        try:
            overlay_id = record_feedback_overlay(
                getattr(self.db, "db_path", ""),
                query_path,
                result_file_id,
                action,
                label=label,
                extra={"note": note},
            )
        except Exception:
            overlay_id = 0
        try:
            from core.motif_gt_store import ingest_from_store_record

            ingest_from_store_record(
                self.db, action, query_path, result_file_id, label
            )
        except Exception:
            pass
        if in_search_session():
            return overlay_id
        return self.db.insert_user_feedback(
            {
                "query_file_id": query_file_id,
                "query_path": query_path,
                "result_file_id": result_file_id,
                "action": action,
                "label": label,
                "note": note,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def score_adjustments(
        self,
        query_path: str,
        candidate_ids: list[int] | None = None,
    ) -> dict[int, float]:
        """file_id -> delta (-0.3 .. +0.15)"""
        rows = self.db.get_feedback_for_query(query_path, candidate_ids)
        deltas: dict[int, float] = {}
        for row in rows:
            action = row.get("action", "")
            if action == ACTION_METADATA_EDIT:
                continue
            fid = int(row["result_file_id"])
            delta = 0.0
            if action == ACTION_SAME_PATTERN:
                delta = 0.15
            elif action == ACTION_SIMILAR_TEXTURE:
                delta = 0.08
            elif action == ACTION_PROMOTE:
                delta = 0.10
            elif action == ACTION_WRONG:
                delta = -0.55
            elif action == ACTION_DEMOTE:
                delta = -0.15
            elif action == ACTION_NOT_SIMILAR:
                delta = -0.35
            else:
                continue
            deltas[fid] = deltas.get(fid, 0.0) + delta
        try:
            from core.search_memory import overlay_adjustments

            for fid, delta in overlay_adjustments(
                getattr(self.db, "db_path", ""), query_path
            ).items():
                deltas[fid] = deltas.get(fid, 0.0) + float(delta)
        except Exception:
            pass
        return {k: max(-0.70, min(0.20, v)) for k, v in deltas.items()}

    def wrong_result_ids(self, query_path: str) -> set[int]:
        """Bu sorgu için kullanıcının 'yanlış' dediği sonuçlar."""
        rows = self.db.get_feedback_for_query(query_path)
        blocked: set[int] = set()
        for row in rows:
            if (row.get("action") or "") == ACTION_WRONG:
                blocked.add(int(row["result_file_id"]))
        try:
            from core.search_memory import overlay_wrong_ids

            blocked |= overlay_wrong_ids(getattr(self.db, "db_path", ""), query_path)
        except Exception:
            pass
        return blocked

    def labels_for_file(self, file_id: int) -> list[str]:
        return self.db.get_feedback_labels_for_file(file_id)

    def family_overrides_for_file(self, file_id: int) -> dict[str, object]:
        """
        Returns:
          {
            "positive": <str|''>,
            "negative": <set[str]>
          }
        """
        return self.db.get_feedback_family_overrides_for_file(file_id)

    def custom_tags_for_file(self, file_id: int) -> list[str]:
        tags = list(self.db.get_feedback_custom_tags_for_file(file_id) or [])
        try:
            from core.search_memory import overlay_custom_tags

            for t in overlay_custom_tags(getattr(self.db, "db_path", ""), file_id):
                if t not in tags:
                    tags.append(t)
        except Exception:
            pass
        return tags

    def learned_tier_for_file(self, file_id: int) -> dict[str, str]:
        return self.db.get_learned_tier_for_file(file_id)

    def record_similarity_tier(
        self,
        query_path: str,
        result_file_id: int,
        tier_id: str,
        *,
        tier_label: str = "",
    ) -> int:
        label = (tier_label or tier_id or "").strip()
        if not label:
            return 0
        note = tier_id if tier_id and tier_id != label else ""
        return self.record(
            query_path,
            result_file_id,
            ACTION_SIMILARITY_TIER,
            label=label,
            note=note,
        )

    def record_custom_tag(self, query_path: str, result_file_id: int, tag: str) -> int:
        tag = (tag or "").strip()
        if not tag:
            return 0
        return self.record(query_path, result_file_id, ACTION_CUSTOM_TAG, label=tag)

    def _supersede_file_feedback(self, result_file_id: int, actions: tuple[str, ...]) -> None:
        """Eski yanlış/metadata satırlarını bu dosya için düşür; başka file_id dokunulmaz."""
        fid = int(result_file_id)
        if fid <= 0:
            return
        wanted = tuple(str(a).strip().lower() for a in actions if str(a).strip())
        if not wanted:
            return
        try:
            from core.search_memory import clear_overlay_actions

            clear_overlay_actions(
                getattr(self.db, "db_path", ""),
                fid,
                wanted,
            )
        except Exception:
            pass
        try:
            ph = ",".join("?" * len(wanted))
            with self.db.connect() as conn:
                conn.execute(
                    f"""
                    DELETE FROM user_feedback
                    WHERE result_file_id=? AND lower(action) IN ({ph})
                    """,
                    (fid, *wanted),
                )
        except Exception:
            pass

    def save_metadata_overlay(
        self,
        result_file_id: int,
        overlay: dict[str, Any],
        *,
        query_path: str = "",
    ) -> int:
        """Per-image user correction. Does not write Pattern Index / FAISS."""
        import json

        fid = int(result_file_id)
        payload = {k: v for k, v in (overlay or {}).items() if v not in (None, "")}
        if not payload:
            return 0
        rid = self.record(
            query_path,
            fid,
            ACTION_METADATA_EDIT,
            label=json.dumps(payload, ensure_ascii=False),
        )
        try:
            from core.search_memory import record_feedback_overlay

            keys = teach_query_keys(query_path, payload)
            dbp = getattr(self.db, "db_path", "")
            for key in keys:
                record_feedback_overlay(
                    dbp,
                    key,
                    fid,
                    "correct",
                    label=str(payload.get("category_path") or payload.get("child") or key),
                    extra={"source": "metadata_edit", **payload},
                )
        except Exception:
            pass
        try:
            from core.motif_gt_store import ingest_from_metadata_overlay

            ingest_from_metadata_overlay(
                self.db, result_file_id, payload, query_path=query_path
            )
        except Exception:
            pass
        return rid

    def metadata_overlay_for_file(self, file_id: int) -> dict[str, Any]:
        import json

        try:
            with self.db.connect() as conn:
                row = conn.execute(
                    """
                    SELECT label FROM user_feedback
                    WHERE result_file_id=? AND action=?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (int(file_id), ACTION_METADATA_EDIT),
                ).fetchone()
        except Exception:
            return {}
        if not row:
            try:
                from core.search_memory import overlay_metadata

                return overlay_metadata(getattr(self.db, "db_path", ""), file_id)
            except Exception:
                return {}
        raw = row["label"] if not isinstance(row, dict) else row.get("label")
        try:
            data = json.loads(raw or "{}")
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}


def apply_metadata_overlay_to_result(result: Any, overlay: dict[str, Any]) -> None:
    """UI-only overlay. Does not change search ranking or index rows."""
    if not result or not overlay:
        return
    parent = str(overlay.get("parent") or "").strip()
    child = str(overlay.get("child") or "").strip()
    path = str(overlay.get("category_path") or "").strip()
    if not path and parent:
        path = f"{parent}/{child}".rstrip("/") if child else parent
    color = str(overlay.get("color_family") or "").strip()
    family = str(overlay.get("pattern_family") or "").strip()
    brand = str(overlay.get("brand") or "").strip()
    tags = overlay.get("tags") or []
    animal = str(overlay.get("animal_print_type") or "").strip()
    # Taksonomi yolu varsa makine alanlarını kullan (stale "Leopard / Animal" metni düşsün).
    if path:
        try:
            from core.category_tree import pattern_fields_for_path

            pf = pattern_fields_for_path(path) or {}
            if str(pf.get("pattern_family") or "").strip():
                family = str(pf.get("pattern_family") or "").strip()
            if str(pf.get("animal_print_type") or "").strip():
                animal = str(pf.get("animal_print_type") or "").strip()
        except Exception:
            pass
    if color:
        result.color_family = color
    if family:
        result.pattern_family = family
    if animal:
        result.animal_print_type = animal
    dbg = result.debug if isinstance(getattr(result, "debug", None), dict) else {}
    if path:
        dbg["manual_category_path"] = path
        dbg["category_path"] = path
        dbg["user_labeled"] = True
        dbg["category_source"] = "manual_user"
        tm = dict(dbg.get("texture_map") or {})
        tm["user_labeled"] = True
        tm["manual_category_path"] = path
        tm["category_path"] = path
        tm["category_source"] = "manual_user"
        if family:
            tm["pattern_family"] = family
        if animal:
            tm["animal_print_type"] = animal
        dbg["texture_map"] = tm
        # Stale search enrichment (eski Leopard learned_canonical) aktif
        # sınıflandırma ile çelişiyorsa güncelle; yoksa learned_* uydurma
        # (admin_label prediction yolunu bozmasın).
        active = child or (path.rsplit("/", 1)[-1].strip() if path else "")
        old_learned = str(
            dbg.get("learned_canonical") or dbg.get("learned_concept_label") or ""
        ).strip()
        if (
            active
            and old_learned
            and old_learned.casefold() != active.casefold()
        ):
            dbg["learned_canonical"] = active
            dbg["learned_concept_label"] = active
            dbg["learned_concept"] = True
            dbg["learned_concept_exact"] = True
            dbg["user_taught_positive"] = True
        elif active and (dbg.get("learned_concept") or dbg.get("learned_concept_exact")):
            # Enrichment bayrakları varken isim boşsa path child yaz.
            if not old_learned:
                dbg["learned_canonical"] = active
                dbg["learned_concept_label"] = active
    if brand:
        dbg["brand_name"] = brand
        tm = dict(dbg.get("texture_map") or {})
        tm["brand_name"] = brand
        dbg["texture_map"] = tm
    if "tags" in overlay:
        dbg["user_tags"] = [str(t) for t in (tags or []) if str(t).strip()]
    elif tags:
        dbg["user_tags"] = [str(t) for t in tags if str(t).strip()]
    if family:
        dbg["result_family"] = family
        dbg["result_confidence"] = 1.0
    result.debug = dbg
