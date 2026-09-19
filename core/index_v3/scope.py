"""Index Engine V3 — shared source scope SSOT (UI + worker aynı çözüm)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScopeResolution:
    """selected=[] → tüm arşiv = yalnız aktif kayıtlı kaynaklar (orphan HARİÇ)."""

    source_ids: list[int]
    selected_source_ids: list[int]
    active_registry_ids: list[int]
    orphan_source_ids: list[int]
    orphan_file_counts: dict[int, int] = field(default_factory=dict)
    mode: str = "all_archive"  # selected | all_archive

    @property
    def source_count_display(self) -> int:
        """UI 'N kaynak': seçili modda seçim sayısı; tüm arşivde aktif registry."""
        if self.mode == "selected":
            return len(self.source_ids)
        return len(self.active_registry_ids)

    @property
    def orphan_file_total(self) -> int:
        return int(sum(self.orphan_file_counts.values()))


def _active_registry_ids(db: Any) -> list[int]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id FROM sources WHERE is_active=1 ORDER BY id"
        ).fetchall()
    return [int(r["id"] if hasattr(r, "keys") else r[0]) for r in rows]


def _registry_ids(db: Any) -> set[int]:
    with db.connect() as conn:
        rows = conn.execute("SELECT id FROM sources").fetchall()
    return {int(r["id"] if hasattr(r, "keys") else r[0]) for r in rows}


def _file_source_ids(db: Any) -> dict[int, int]:
    """source_id → active file count (not excluded/missing)."""
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT source_id, COUNT(*) AS n
            FROM files
            WHERE status NOT IN ('excluded_internal','missing')
            GROUP BY source_id
            """
        ).fetchall()
    out: dict[int, int] = {}
    for r in rows:
        sid = int(r["source_id"] if hasattr(r, "keys") else r[0])
        n = int(r["n"] if hasattr(r, "keys") else r[1])
        out[sid] = n
    return out


def resolve_index_scope(
    db: Any,
    selected_source_ids: list[int] | None,
) -> ScopeResolution:
    """UI ve V3 worker aynı fonksiyonu kullanmalı.

    Policy:
    - selected dolu → yalnız seçilen id'ler (index/kuyruk)
    - selected boş → tüm arşiv = YALNIZ aktif kayıtlı sources
    - Orphan (sources satırı olmayan file.source_id) arşive/kuyruğa GİRMEZ
    - Orphan yalnız diagnostic (orphan_file_total / rapor)
    """
    selected = [int(x) for x in (selected_source_ids or []) if int(x) > 0]
    active = _active_registry_ids(db)
    registry = _registry_ids(db)
    file_counts = _file_source_ids(db)
    orphan_ids = sorted(sid for sid in file_counts if sid not in registry)
    orphan_counts = {sid: file_counts[sid] for sid in orphan_ids}

    if selected:
        return ScopeResolution(
            source_ids=list(selected),
            selected_source_ids=list(selected),
            active_registry_ids=active,
            orphan_source_ids=orphan_ids,
            orphan_file_counts=orphan_counts,
            mode="selected",
        )

    # Tüm arşiv: yalnız aktif registry — orphan / pasif file_counts YOK
    return ScopeResolution(
        source_ids=list(active),
        selected_source_ids=[],
        active_registry_ids=active,
        orphan_source_ids=orphan_ids,
        orphan_file_counts=orphan_counts,
        mode="all_archive",
    )


def orphan_source_report(db: Any) -> list[dict[str, Any]]:
    """Orphan source_id raporu — purge YOK; path örnekleri."""
    scope = resolve_index_scope(db, [])
    out: list[dict[str, Any]] = []
    for sid in scope.orphan_source_ids:
        with db.connect() as conn:
            sample = conn.execute(
                """
                SELECT id, path, status FROM files
                WHERE source_id=? AND status NOT IN ('excluded_internal','missing')
                ORDER BY id LIMIT 5
                """,
                (int(sid),),
            ).fetchall()
            roots = conn.execute(
                """
                SELECT DISTINCT
                  CASE
                    WHEN instr(replace(path,'/','\\'), '\\')>0
                    THEN substr(path, 1, instr(replace(path,'/','\\'), '\\')-1)
                    ELSE path
                  END AS root_hint
                FROM files WHERE source_id=? LIMIT 10
                """,
                (int(sid),),
            ).fetchall()
        out.append(
            {
                "source_id": int(sid),
                "file_count": int(scope.orphan_file_counts.get(sid, 0)),
                "in_sources_table": False,
                "sample_paths": [
                    str(r["path"] if hasattr(r, "keys") else r[1]) for r in sample
                ],
                "path_prefixes": [
                    str(r["root_hint"] if hasattr(r, "keys") else r[0]) for r in roots
                ],
                "policy": "diagnostic_only; not_in_all_archive; no_auto_index",
            }
        )
    return out
