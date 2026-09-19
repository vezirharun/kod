"""Probe search_memory concept positives (read-only)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "search_memory.db"


def main() -> None:
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out: dict = {"tables": tables}
    if "concept_registry" in tables:
        rows = c.execute(
            "SELECT id, canonical FROM concept_registry"
        ).fetchall()
        out["concepts"] = {r["canonical"]: r["id"] for r in rows}
    if "concept_examples" in tables:
        cols = [r[1] for r in c.execute("PRAGMA table_info(concept_examples)")]
        out["example_cols"] = cols
        role_col = "role" if "role" in cols else None
        sql = (
            "SELECT cr.canonical, COUNT(*) AS n "
            "FROM concept_examples ce JOIN concept_registry cr ON cr.id=ce.concept_id "
        )
        if role_col:
            sql += "WHERE ce.role='positive' "
        sql += "GROUP BY cr.canonical ORDER BY n DESC"
        out["positives"] = {
            r["canonical"]: r["n"] for r in c.execute(sql).fetchall()
        }
    if "learning_events" in tables:
        out["learning_events"] = c.execute(
            "SELECT COUNT(*) AS n FROM learning_events"
        ).fetchone()["n"]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    c.close()


if __name__ == "__main__":
    main()
