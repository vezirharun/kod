"""Thin hooks into Öğret inbox — no new feedback UI."""

from __future__ import annotations

from typing import Any

from core.archive_intelligence.store import (
    ArchiveIntelligenceStore,
    default_store_path,
)


def _store_for(db_path: str, settings: Any = None) -> ArchiveIntelligenceStore:
    custom = ""
    if settings is not None:
        custom = str(getattr(settings, "archive_intelligence_db_path", "") or "")
    path = custom or default_store_path(db_path)
    return ArchiveIntelligenceStore(path)


def resolve_candidates_for_files(
    db_path: str,
    file_ids: list[int],
    *,
    status: str = "resolved",
    settings: Any = None,
) -> int:
    try:
        store = _store_for(db_path, settings)
        return store.resolve_files(list(file_ids), status=status)
    except Exception:
        return 0


def pending_archive_cards(db, db_path: str, *, limit: int = 40) -> dict[str, list]:
    """Build TeachMeCard-compatible cards from AI candidate store."""
    from core.teach_me import TeachMeCard
    from core.manual_label_guard import parse_texture_map

    pools: dict[str, list] = {"suspicious": [], "outlier": []}
    try:
        store = _store_for(db_path)
        pending = store.list_pending(limit=int(limit))
    except Exception:
        return {"suspicious": []}
    if not pending:
        return {"suspicious": []}

    ids = [int(r["file_id"]) for r in pending if int(r.get("file_id") or 0) > 0]
    recs: dict[int, dict[str, Any]] = {}
    if ids:
        ph = ",".join("?" * len(ids))
        try:
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT f.id, f.filename, f.path, f.thumbnail_path, "
                    "f.feature_preview_path, f.pattern_family, f.category_path, "
                    "f.pattern_confidence, fe.texture_map "
                    "FROM files f LEFT JOIN features fe ON fe.file_id=f.id "
                    f"WHERE f.id IN ({ph})",
                    ids,
                ).fetchall()
            recs = {int(r["id"]): dict(r) for r in rows}
        except Exception:
            recs = {}

    out_suspicious: list = []
    for row in pending:
        fid = int(row.get("file_id") or 0)
        rec = recs.get(fid) or {"id": fid}
        tm = parse_texture_map(rec.get("texture_map"))
        kind = str(row.get("kind") or "suspicious")
        # Teach inbox only has suspicious/undefined/undecided/new_concept —
        # map outlier → suspicious with stronger reason prefix.
        pool = "suspicious"
        reason = str(row.get("reason") or "Arşiv tutarsızlığı")
        if kind == "outlier":
            reason = f"[Ayık] {reason}"
        guess = str(row.get("guess") or rec.get("pattern_family") or tm.get("pattern_family") or "Belirsiz")
        try:
            conf = float(row.get("consistency") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        thumb = str(rec.get("thumbnail_path") or "")
        feat = str(rec.get("feature_preview_path") or "")
        card = TeachMeCard(
            file_id=fid,
            filename=str(rec.get("filename") or ""),
            path=str(rec.get("path") or ""),
            preview_path=thumb or feat,
            pool=pool,
            reason=reason,
            guess=guess,
            confidence=conf,
            category=str(rec.get("category_path") or tm.get("category_path") or ""),
            feature_preview_path=feat,
            color_family=str(tm.get("color_family") or ""),
            pattern_family=str(rec.get("pattern_family") or tm.get("pattern_family") or ""),
            brand=str(tm.get("brand_name") or ""),
            tags=[],
            suggested=guess,
        )
        out_suspicious.append(card)
        if len(out_suspicious) >= int(limit):
            break
    return {"suspicious": out_suspicious}


def merge_archive_candidates_into_pools(
    pools: dict[str, list],
    db,
    db_path: str,
    *,
    skip_ids: set[int] | None = None,
    limit: int = 40,
    cache_dir: str = "",
) -> dict[str, list]:
    """Merge AI candidates into existing Öğret pools (suggest-only)."""
    skip = set(skip_ids or set())
    try:
        extra = pending_archive_cards(db, db_path, limit=limit)
    except Exception:
        return pools
    existing = {
        int(c.file_id)
        for cards in pools.values()
        for c in (cards or [])
        if int(getattr(c, "file_id", 0) or 0) > 0
    }
    try:
        from core.teach_me import enrich_teach_card_paths
    except Exception:
        enrich_teach_card_paths = None  # type: ignore

    for kind, cards in extra.items():
        dest = pools.setdefault(kind if kind in pools else "suspicious", pools.setdefault("suspicious", []))
        for card in cards or []:
            fid = int(card.file_id)
            if fid <= 0 or fid in skip or fid in existing:
                continue
            if len(dest) >= int(limit):
                break
            if enrich_teach_card_paths is not None:
                try:
                    card = enrich_teach_card_paths(card, db=db, cache_dir=cache_dir)
                except Exception:
                    pass
            dest.append(card)
            existing.add(fid)
    return pools
