"""Per-customer Visual Memory SQLite store.

Never opens production patterns.db for writes. Customer keys cannot read
each other's files.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.visual_memory.schema import ALL_STATUSES, AUTO_STATUSES, DDL, STATUS_UNKNOWN

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def customer_key(customer: str) -> str:
    raw = (customer or "").strip() or "_default"
    key = _SAFE.sub("_", raw).strip(".")
    if not key or key in {"-", "_"}:
        key = "_default"
    return key[:80]


def memory_db_path(root: str | Path, customer: str) -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{customer_key(customer)}.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def centroid_to_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr.tobytes()


def blob_to_centroid(blob: bytes | None, dim: int = 0) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim and arr.size != dim:
        return None
    return arr.copy()


class VisualMemoryStore:
    def __init__(self, path: str | Path, *, customer: str):
        self.path = Path(path)
        self.customer = customer_key(customer)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.executescript(DDL)
            con.execute(
                "INSERT OR IGNORE INTO vm_meta(key, value) VALUES('customer', ?)",
                (self.customer,),
            )
            con.execute(
                "INSERT OR IGNORE INTO vm_meta(key, value) VALUES('schema', '1')",
            )
            stored = con.execute(
                "SELECT value FROM vm_meta WHERE key='customer'"
            ).fetchone()
            if stored and stored["value"] != self.customer:
                raise ValueError(
                    f"visual memory customer mismatch: {stored['value']} != {self.customer}"
                )

    def replace_clusters(self, clusters: Iterable[dict[str, Any]]) -> int:
        rows = list(clusters)
        with self._connect() as con:
            con.execute("DELETE FROM vm_members")
            con.execute("DELETE FROM vm_clusters")
            for c in rows:
                self._upsert_cluster(con, c)
        return len(rows)

    def _upsert_cluster(self, con: sqlite3.Connection, c: dict[str, Any]) -> None:
        status = str(c.get("status") or STATUS_UNKNOWN)
        if status not in ALL_STATUSES:
            raise ValueError(f"invalid cluster status: {status}")
        cid = str(c["cluster_id"])
        now = c.get("updated_at") or _now()
        created = c.get("created_at") or now
        centroid = c.get("embedding_centroid")
        blob = None
        dim = int(c.get("embedding_dim") or 0)
        if isinstance(centroid, np.ndarray):
            blob = centroid_to_blob(centroid)
            dim = int(centroid.size)
        elif isinstance(centroid, (bytes, bytearray)):
            blob = bytes(centroid)
        con.execute(
            """
            INSERT INTO vm_clusters(
                cluster_id, embedding_centroid, embedding_dim, embedding_kind,
                member_count, representative_ids, visual_dna_summary,
                object_distribution, color_distribution, texture_distribution,
                pattern_family_distribution, confidence, status, claimed_class,
                evidence_sources, cohesion, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                cid,
                blob,
                dim,
                str(c.get("embedding_kind") or ""),
                int(c.get("member_count") or 0),
                json.dumps(list(c.get("representative_ids") or []), ensure_ascii=False),
                json.dumps(c.get("visual_dna_summary") or {}, ensure_ascii=False),
                json.dumps(c.get("object_distribution") or {}, ensure_ascii=False),
                json.dumps(c.get("color_distribution") or {}, ensure_ascii=False),
                json.dumps(c.get("texture_distribution") or {}, ensure_ascii=False),
                json.dumps(c.get("pattern_family_distribution") or {}, ensure_ascii=False),
                float(c.get("confidence") or 0),
                status,
                str(c.get("claimed_class") or ""),
                json.dumps(list(c.get("evidence_sources") or []), ensure_ascii=False),
                float(c.get("cohesion") or 0),
                created,
                now,
            ),
        )
        for m in c.get("members") or []:
            con.execute(
                """
                INSERT OR REPLACE INTO vm_members(
                    file_id, cluster_id, sim_to_centroid, source_id, evidence, created_at
                ) VALUES (?,?,?,?,?,?)
                """,
                (
                    int(m["file_id"]),
                    cid,
                    float(m.get("sim_to_centroid") or 0),
                    int(m.get("source_id") or 0),
                    json.dumps(m.get("evidence") or {}, ensure_ascii=False),
                    now,
                ),
            )

    def list_clusters(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM vm_clusters ORDER BY member_count DESC, cluster_id"
            ).fetchall()
            return [self._cluster_from_row(r) for r in rows]

    def get_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM vm_clusters WHERE cluster_id=?", (cluster_id,)
            ).fetchone()
            if not row:
                return None
            data = self._cluster_from_row(row)
            members = con.execute(
                "SELECT * FROM vm_members WHERE cluster_id=? ORDER BY sim_to_centroid DESC",
                (cluster_id,),
            ).fetchall()
            data["members"] = [dict(m) for m in members]
            return data

    def cluster_for_file(self, file_id: int) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT cluster_id FROM vm_members WHERE file_id=?",
                (int(file_id),),
            ).fetchone()
        if not row:
            return None
        return self.get_cluster(row["cluster_id"])

    def record_feedback(
        self,
        *,
        file_id: int,
        action: str,
        label: str = "",
        cluster_id: str = "",
        note: str = "",
    ) -> int:
        """Phase 1: persist only. Does not promote cluster status or claimed_class."""
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO vm_feedback(file_id, cluster_id, label, action, applied, note, created_at)
                VALUES (?,?,?,?,0,?,?)
                """,
                (int(file_id), cluster_id, label, action, note, _now()),
            )
            return int(cur.lastrowid)

    def feedback_count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM vm_feedback").fetchone()[0])

    def stats(self) -> dict[str, Any]:
        with self._connect() as con:
            n = int(con.execute("SELECT COUNT(*) FROM vm_clusters").fetchone()[0])
            members = int(con.execute("SELECT COUNT(*) FROM vm_members").fetchone()[0])
            claimed = int(
                con.execute(
                    "SELECT COUNT(*) FROM vm_clusters WHERE claimed_class<>''"
                ).fetchone()[0]
            )
            auto = int(
                con.execute(
                    f"SELECT COUNT(*) FROM vm_clusters WHERE status IN ({','.join('?' * len(AUTO_STATUSES))})",
                    tuple(AUTO_STATUSES),
                ).fetchone()[0]
            )
        return {
            "customer": self.customer,
            "path": str(self.path),
            "clusters": n,
            "members": members,
            "claimed_class_n": claimed,
            "auto_status_n": auto,
        }

    def _cluster_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in (
            "representative_ids",
            "visual_dna_summary",
            "object_distribution",
            "color_distribution",
            "texture_distribution",
            "pattern_family_distribution",
            "evidence_sources",
        ):
            raw = d.get(key) or ("[]" if key in ("representative_ids", "evidence_sources") else "{}")
            try:
                d[key] = json.loads(raw)
            except Exception:
                d[key] = [] if key in ("representative_ids", "evidence_sources") else {}
        d["embedding_centroid"] = blob_to_centroid(
            d.get("embedding_centroid"), int(d.get("embedding_dim") or 0)
        )
        return d
