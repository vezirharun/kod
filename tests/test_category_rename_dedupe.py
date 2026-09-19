"""Category memory: case dedupe + rename misspellings."""
from __future__ import annotations

from pathlib import Path

from core.category_memory import (
    _key,
    dedupe_category_roots,
    dynamic_parents,
    register_root_category,
    rename_root_category,
)
from core.category_tree import parent_categories
from core.search_memory import memory_db_path


def test_animal_animal_case_deduped_in_list(tmp_path: Path):
    db = str(tmp_path / "patterns.db")
    assert register_root_category(db, "animal")
    assert register_root_category(db, "Animal")
    parents = dynamic_parents(db)
    animalish = [p for p in parents if _key(p) == _key("animal")]
    assert len(animalish) == 1


def test_rename_moti_to_motif(tmp_path: Path):
    db = str(tmp_path / "patterns.db")
    assert register_root_category(db, "moti")
    stats = rename_root_category(db, "moti", "Motif")
    assert stats["ok"]
    parents = dynamic_parents(db)
    assert "Motif" in parents
    assert "moti" not in parents
    # alias kept for resolution
    from core.category_memory import dynamic_root_rows

    rows = {r["label"]: r for r in dynamic_root_rows(db)}
    assert "Motif" in rows
    aliases = [str(a).lower() for a in rows["Motif"].get("aliases") or []]
    assert "moti" in aliases


def test_parent_categories_no_case_dupes(tmp_path: Path):
    db = str(tmp_path / "patterns.db")
    register_root_category(db, "figuratif")
    register_root_category(db, "Figuratif")
    roots = parent_categories(db)
    keys = [_key(x) for x in roots]
    assert keys.count(_key("figuratif")) == 1


def test_dedupe_existing_roots(tmp_path: Path):
    db = str(tmp_path / "patterns.db")
    mem = memory_db_path(db)
    # Bypass register to force two case variants in SQLite
    import json
    import sqlite3
    from datetime import datetime, timezone

    from core.category_memory import _conn, _write_db_path

    write = _write_db_path(db)
    conn = _conn(write)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO category_memory_roots(label,aliases,source,created_at,updated_at) VALUES(?,?,?,?,?)",
        ("animal", "[]", "user", now, now),
    )
    conn.execute(
        "INSERT INTO category_memory_roots(label,aliases,source,created_at,updated_at) VALUES(?,?,?,?,?)",
        ("Animal", "[]", "user", now, now),
    )
    conn.commit()
    conn.close()
    stats = dedupe_category_roots(db)
    assert stats["merged"] >= 1
    assert len([p for p in dynamic_parents(db) if _key(p) == "animal"]) == 1
