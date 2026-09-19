"""Reviewed category corrections: admin gold data and user learning pool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.category_learning import apply_manual_category
from core.db import Database


def submit_category_correction(
    settings,
    *,
    file_id: int,
    category_path: str,
    old_category: str = "",
    old_predictions: list[dict[str, Any]] | None = None,
    propagation_scope: str = "single",
    apply_category: bool = True,
) -> dict[str, Any]:
    path = (category_path or "").strip().strip("/")
    if not path:
        return {"saved": False, "status": "invalid"}
    is_admin = str(getattr(settings, "user_role", "user")).lower() == "admin"
    parent, _, child = path.partition("/")
    db = Database(settings.db_path)
    status = "gold_dataset" if is_admin else "pending_review"
    correction_id = db.insert_category_correction(
        {
            "file_id": file_id,
            "old_predictions": old_predictions or [],
            "old_category": old_category,
            "correct_category": parent,
            "correct_subcategory": child,
            "actor": getattr(settings, "user_name", "local_user"),
            "is_admin": is_admin,
            "propagation_scope": propagation_scope,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    applied = {"updated": 0, "propagated": 0}
    if is_admin and apply_category:
        applied = apply_manual_category(
            settings,
            file_id,
            path,
            propagate_exact=propagation_scope in ("exact", "family", "future"),
            refresh_visual=False,
        )
    return {
        "saved": True,
        "correction_id": correction_id,
        "status": status,
        "applied": applied,
        "category_path": path,
    }
