"""Format denetimi — arşivdeki format dağılımı."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.db import sql_processing_ready_clause
from core.settings import SUPPORTED_EXTENSIONS

# Display merge: physical extensions collapse for UI stats.
_EXT_ALIASES: dict[str, str] = {
    "jpeg": "jpg",
    "tiff": "tif",
}


def _canon_ext(ext: str) -> str:
    e = str(ext or "").lower().lstrip(".")
    return _EXT_ALIASES.get(e, e)


def audit_format_counts(db_path: str) -> list[dict]:
    """DB'deki dosyaları format bazında say.

    ``searchable`` = search engine processing-ready predicate.
    JPG+JPEG / TIF+TIFF birleştirilir. ``missing`` ayrı kolon.
    """
    ready = sql_processing_ready_clause()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT
            lower(substr(f.filename, instr(f.filename, '.') + 1)) AS ext,
            COUNT(*) AS found,
            SUM(CASE WHEN f.status='indexed' THEN 1 ELSE 0 END) AS indexed,
            SUM(CASE WHEN {ready} THEN 1 ELSE 0 END) AS searchable,
            SUM(CASE WHEN f.status='missing' THEN 1 ELSE 0 END) AS missing,
            SUM(CASE WHEN f.preview_status='preview_ok' THEN 1 ELSE 0 END) AS preview_ok,
            SUM(CASE WHEN f.thumbnail_status='thumb_ok' OR f.thumbnail_path!='' THEN 1 ELSE 0 END) AS thumbnail_ok,
            SUM(CASE WHEN f.metadata_status='metadata_ok' THEN 1 ELSE 0 END) AS metadata_ok,
            SUM(CASE WHEN f.semantic_status='semantic_ok' OR f.semantic_status='ok' THEN 1 ELSE 0 END) AS ai_analyzed,
            SUM(CASE WHEN fe.dino_embedding IS NOT NULL AND length(fe.dino_embedding)>0 THEN 1 ELSE 0 END) AS embedding_ok,
            SUM(CASE WHEN f.unsupported_preview=1 THEN 1 ELSE 0 END) AS unsupported,
            SUM(CASE WHEN f.status='error' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN f.parser_error IS NOT NULL AND f.parser_error!='' THEN 1 ELSE 0 END) AS dependency_missing
        FROM files f
        LEFT JOIN features fe ON fe.file_id=f.id
        WHERE instr(f.filename, '.') > 0
        GROUP BY ext
        ORDER BY found DESC
        """
    ).fetchall()
    conn.close()

    # Merge alias extensions into canonical buckets.
    merged: dict[str, dict] = {}
    for r in rows:
        key = _canon_ext(r["ext"])
        bucket = merged.setdefault(
            key,
            {
                "format": key,
                "found": 0,
                "indexed": 0,
                "searchable": 0,
                "missing": 0,
                "preview_ok": 0,
                "thumbnail_ok": 0,
                "metadata_ok": 0,
                "ai_analyzed": 0,
                "embedding_ok": 0,
                "unsupported": 0,
                "failed": 0,
                "dependency_missing": 0,
            },
        )
        for field in (
            "found",
            "indexed",
            "searchable",
            "missing",
            "preview_ok",
            "thumbnail_ok",
            "metadata_ok",
            "ai_analyzed",
            "embedding_ok",
            "unsupported",
            "failed",
            "dependency_missing",
        ):
            bucket[field] += int(r[field] or 0)

    # Preferred display order from supported set (canonicalized).
    preferred: list[str] = []
    seen: set[str] = set()
    for ext in sorted(SUPPORTED_EXTENSIONS, key=lambda e: e.lstrip(".")):
        key = _canon_ext(ext)
        if key in seen:
            continue
        seen.add(key)
        preferred.append(key)

    out: list[dict] = []
    for key in preferred:
        row = merged.get(key, {})
        out.append(
            {
                "format": key,
                "found": int(row.get("found", 0)),
                "indexed": int(row.get("indexed", 0)),
                "preview_ok": int(row.get("preview_ok", 0)),
                "thumbnail_ok": int(row.get("thumbnail_ok", 0)),
                "metadata_ok": int(row.get("metadata_ok", 0)),
                "searchable": int(row.get("searchable", 0)),
                "missing": int(row.get("missing", 0)),
                "embedding_ok": int(row.get("embedding_ok", 0)),
                "ai_analyzed": int(row.get("ai_analyzed", 0)),
                "unsupported": int(row.get("unsupported", 0)),
                "failed": int(row.get("failed", 0)),
                "dependency_missing": int(row.get("dependency_missing", 0)),
            }
        )
    return out


def count_missing_by_format(db_path: str) -> dict[str, int]:
    """Convenience: format → missing status count (path may be stale)."""
    return {r["format"]: int(r.get("missing") or 0) for r in audit_format_counts(db_path)}
