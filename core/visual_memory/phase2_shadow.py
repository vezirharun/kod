"""Phase 2 Visual Memory shadow store — isolated from Phase 1 and production.

Idempotent snapshot replace. Never writes patterns.db or Phase 1 VM DBs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.visual_memory.schema import STATUS_DISCOVERED, STATUS_UNKNOWN

DDL = """
CREATE TABLE IF NOT EXISTS vm2_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vm2_clusters (
    cluster_id TEXT PRIMARY KEY,
    parent_cluster_id TEXT DEFAULT '',
    family TEXT DEFAULT '',
    member_count INTEGER DEFAULT 0,
    member_ids TEXT DEFAULT '[]',
    clip_cohesion REAL DEFAULT 0,
    dino_gain REAL,
    dino_intra REAL,
    dino_inter REAL,
    split_reason TEXT DEFAULT '',
    status TEXT DEFAULT 'DISCOVERED',
    claimed_class TEXT DEFAULT '',
    confidence REAL DEFAULT 0,
    pattern_family_distribution TEXT DEFAULT '{}',
    object_distribution TEXT DEFAULT '{}',
    created_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_vm2_parent ON vm2_clusters(parent_cluster_id);
"""

ALLOWED_STATUS = frozenset({STATUS_UNKNOWN, STATUS_DISCOVERED})


def phase2_shadow_path(data_dir: str | Path) -> Path:
    root = Path(data_dir) / "visual_memory_phase2"
    root.mkdir(parents=True, exist_ok=True)
    return root / "_shadow.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Phase2ShadowStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(DDL)
            con.execute(
                "INSERT OR IGNORE INTO vm2_meta(key, value) VALUES('schema','1')"
            )
            con.execute(
                "INSERT OR IGNORE INTO vm2_meta(key, value) VALUES('kind','phase2_shadow')"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path))
        con.row_factory = sqlite3.Row
        return con

    def replace_snapshot(
        self,
        clusters: list[dict[str, Any]],
        *,
        splits: list[dict[str, Any]] | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Idempotent: full snapshot replace. Same IDs on rerun → no duplicates."""
        now = _now()
        rows = []
        for c in clusters:
            status = str(c.get("status") or STATUS_DISCOVERED)
            if status not in ALLOWED_STATUS:
                status = STATUS_DISCOVERED
            members = c.get("members") or []
            ids = [int(m.get("file_id") or m) for m in members] if members else list(
                c.get("representative_ids") or []
            )
            family = str(c.get("family") or "")
            if not family:
                famd = c.get("pattern_family_distribution") or {}
                if len(famd) == 1:
                    family = next(iter(famd))
            rows.append(
                (
                    str(c["cluster_id"]),
                    str(c.get("parent_cluster_id") or ""),
                    family,
                    int(c.get("member_count") or len(ids)),
                    json.dumps(ids, ensure_ascii=False),
                    float(c.get("cohesion") or c.get("clip_cohesion") or 0),
                    c.get("split_gain"),
                    c.get("dino_intra_similarity"),
                    c.get("dino_inter_similarity"),
                    str(c.get("split_reason") or ""),
                    status,
                    "",
                    float(c.get("confidence") or 0),
                    json.dumps(c.get("pattern_family_distribution") or {}, ensure_ascii=False),
                    json.dumps(c.get("object_distribution") or {}, ensure_ascii=False),
                    now,
                    now,
                )
            )
        with self._connect() as con:
            con.execute("DELETE FROM vm2_clusters")
            con.executemany(
                """
                INSERT INTO vm2_clusters(
                    cluster_id, parent_cluster_id, family, member_count, member_ids,
                    clip_cohesion, dino_gain, dino_intra, dino_inter, split_reason,
                    status, claimed_class, confidence,
                    pattern_family_distribution, object_distribution,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            con.execute(
                "INSERT OR REPLACE INTO vm2_meta(key, value) VALUES('updated_at', ?)",
                (now,),
            )
            con.execute(
                "INSERT OR REPLACE INTO vm2_meta(key, value) VALUES('cluster_n', ?)",
                (str(len(rows)),),
            )
            if before is not None:
                con.execute(
                    "INSERT OR REPLACE INTO vm2_meta(key, value) VALUES('phase1_metrics', ?)",
                    (json.dumps(before, ensure_ascii=False),),
                )
            if after is not None:
                con.execute(
                    "INSERT OR REPLACE INTO vm2_meta(key, value) VALUES('phase2_metrics', ?)",
                    (json.dumps(after, ensure_ascii=False),),
                )
            if splits is not None:
                con.execute(
                    "INSERT OR REPLACE INTO vm2_meta(key, value) VALUES('splits', ?)",
                    (json.dumps(splits, ensure_ascii=False),),
                )
        return self.stats()

    def list_clusters(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM vm2_clusters ORDER BY member_count DESC, cluster_id"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["member_ids"] = json.loads(d.get("member_ids") or "[]")
            d["pattern_family_distribution"] = json.loads(
                d.get("pattern_family_distribution") or "{}"
            )
            d["object_distribution"] = json.loads(d.get("object_distribution") or "{}")
            out.append(d)
        return out

    def stats(self) -> dict[str, Any]:
        clusters = self.list_clusters()
        claimed = sum(1 for c in clusters if c.get("claimed_class"))
        children = [c for c in clusters if c.get("parent_cluster_id")]
        return {
            "path": str(self.path),
            "clusters": len(clusters),
            "children": len(children),
            "parents_split": len({c["parent_cluster_id"] for c in children if c["parent_cluster_id"]}),
            "claimed_class_n": claimed,
            "ids": [c["cluster_id"] for c in clusters],
        }
